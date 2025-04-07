# inference_nuke_converter.py
import os
import argparse
import logging
import torch
import torch.nn as nn
import time # For the sleep warning
from types import SimpleNamespace # Add import

# --- Helper to convert nested dict to nested SimpleNamespace ---
#
def dict_to_namespace(d):
    if isinstance(d,dict):
        safe_d={};
        for k,v in d.items(): safe_key=k.replace('-','_'); safe_d[safe_key]=dict_to_namespace(v)
        return SimpleNamespace(**safe_d)
    elif isinstance(d,list): return [dict_to_namespace(item) for item in d]
    else: return d

# --- Model Definition ---
# 
class DoubleConv(nn.Module):
    # ... (Same as training/inference code) ...
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        mid_channels = max(1, mid_channels) # Ensure mid_channels is at least 1
        self.d = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True) )
    def forward(self, x): return self.d(x)
class Down(nn.Module):
    # ... (Same as training/inference code) ...
    def __init__(self, in_channels, out_channels):
        super().__init__(); self.m = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.m(x)
class Up(nn.Module):
    # ... (Same as training/inference code) ...
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__(); self.bilinear = bilinear; conv_in_channels=0; up_out_channels=0;
        if bilinear: self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True); conv_in_channels = in_channels + skip_channels
        else: up_out_channels = in_channels // 2; self.up = nn.ConvTranspose2d(in_channels, up_out_channels, kernel_size=2, stride=2); conv_in_channels = up_out_channels + skip_channels
        self.conv = DoubleConv(conv_in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1); diffY = x2.size(2) - x1.size(2); diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0: x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1); return self.conv(x)
class OutConv(nn.Module):
    # ... (Same as training/inference code) ...
    def __init__(self, in_channels, out_channels): super().__init__(); self.c = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x): return self.c(x)

# UNet class using model_size_dims
class UNet(nn.Module):
    def __init__(self, config, n_ch=3, n_cls=3, bilinear=True):
        super().__init__(); self.n_ch = n_ch; self.n_cls = n_cls;
        self.hidden_size = config.model.model_size_dims # Use new key
        self.bilinear = bilinear
        h = self.hidden_size; chs = {'enc1': h, 'enc2': h*2, 'enc3': h*4, 'enc4': h*8, 'bottle': h*16}
        self.inc = DoubleConv(n_ch, chs['enc1']); self.down1 = Down(chs['enc1'], chs['enc2']); self.down2 = Down(chs['enc2'], chs['enc3']); self.down3 = Down(chs['enc3'], chs['enc4']); self.down4 = Down(chs['enc4'], chs['bottle'])
        self.up1 = Up(chs['bottle'], chs['enc4'], chs['enc4'], bilinear); self.up2 = Up(chs['enc4'], chs['enc3'], chs['enc3'], bilinear); self.up3 = Up(chs['enc3'], chs['enc2'], chs['enc2'], bilinear); self.up4 = Up(chs['enc2'], chs['enc1'], chs['enc1'], bilinear)
        self.outc = OutConv(chs['enc1'], n_cls)
    def forward(self, x):
        x1=self.inc(x); x2=self.down1(x1); x3=self.down2(x2); x4=self.down3(x3); x5=self.down4(x4)
        x=self.up1(x5, x4); x=self.up2(x, x3); x=self.up3(x, x2); x=self.up4(x, x1); return self.outc(x)

# --- Normalization Wrapper ---
# 
class NormalizedUNet(nn.Module):
    def __init__(self, unet_model):
        super().__init__()
        self.unet = unet_model
        self.register_buffer('mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
    def forward(self, x):
        normalized_x = (x - self.mean) / self.std
        unet_output = self.unet(normalized_x)
        denormalized_output = (unet_output * self.std) + self.mean
        clamped_output = torch.clamp(denormalized_output, 0.0, 1.0)
        return clamped_output

# --- Modified load_model_for_conversion ---
# 
def load_model_for_conversion(checkpoint_path, device='cpu'):
    """Loads TuNet checkpoint, wraps it for normalization."""
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Determine Config Source
    if 'config' in checkpoint:
        logging.info("Detected 'config' key (YAML-based training).")
        config_source = checkpoint['config']
        if isinstance(config_source, dict): config_source = dict_to_namespace(config_source)
        is_new_format = True
    elif 'args' in checkpoint:
        logging.info("Detected 'args' key (argparse-based training).")
        config_source = checkpoint['args']
        if isinstance(config_source, argparse.Namespace): config_source = SimpleNamespace(**vars(config_source))
        is_new_format = False
    else:
        raise ValueError("Checkpoint missing configuration ('config' or 'args').")

    # Extract Parameters Safely (Handles model_size_dims and unet_hidden_size)
    default_hidden_size = 64; default_loss = 'l1'; default_resolution = 512; default_bilinear = True
    model_size_saved = default_hidden_size
    if is_new_format:
        model_config = getattr(config_source, 'model', SimpleNamespace())
        model_size_saved = getattr(model_config, 'model_size_dims', getattr(model_config, 'unet_hidden_size', default_hidden_size))
        loss_mode = getattr(getattr(config_source, 'training', SimpleNamespace()), 'loss', default_loss)
        resolution = getattr(getattr(config_source, 'data', SimpleNamespace()), 'resolution', default_resolution)
        bilinear_mode = default_bilinear # Assuming bilinear 
    else: # Old format (args)
        quality_legacy = getattr(config_source, 'quality', None)
        if quality_legacy == 'HQ': loss_mode = 'l1+lpips'
        elif quality_legacy == 'LQ': loss_mode = 'l1'
        else: loss_mode = getattr(config_source, 'loss', default_loss)
        model_size_saved = getattr(config_source, 'model_size_dims', getattr(config_source, 'unet_hidden_size', default_hidden_size))
        resolution = getattr(config_source, 'resolution', default_resolution)
        bilinear_mode = getattr(config_source, 'bilinear', default_bilinear)
    logging.info(f"Checkpoint params: Saved Model Size={model_size_saved}, Loss='{loss_mode}', Res={resolution}, Bilinear={bilinear_mode}")

    # Calculate Effective Hidden Size
    hq_default_bump_size = 96; default_size_for_bump = 64
    effective_model_size = model_size_saved
    if loss_mode == 'l1+lpips' and model_size_saved == default_size_for_bump:
        effective_model_size = hq_default_bump_size
        logging.info(f"Applied model size bump logic: Effective Size = {effective_model_size}")
    else:
        logging.info(f"Effective Model Size = {effective_model_size}")

    # Instantiate the ORIGINAL UNet using a minimal config with model_size_dims
    minimal_unet_config = SimpleNamespace(model=SimpleNamespace(model_size_dims=effective_model_size))
    base_model = UNet(config=minimal_unet_config, n_ch=3, n_cls=3, bilinear=bilinear_mode)

    # Load state dict into original UNet
    state_dict = checkpoint['model_state_dict']
    if all(key.startswith('module.') for key in state_dict):
        logging.info("Removing 'module.' prefix from state_dict keys."); state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    try: base_model.load_state_dict(state_dict)
    except RuntimeError as e: logging.error(f"Error loading state_dict: {e}\nCheck model definition matches checkpoint."); raise

    # Instantiate and setup the WRAPPER
    wrapped_model = NormalizedUNet(base_model)
    wrapped_model.to(device)
    wrapped_model.eval()

    logging.info("Model loaded, wrapped for normalization, and set to evaluation mode.")
    return wrapped_model, resolution

# --- Function to Generate Nuke Script ---
def generate_nuke_script(pt_file_path, cat_file_path, nk_file_path):
    """Generates a Nuke script (.nk) file with pre-configured nodes."""

    # Ensure paths use forward slashes for Nuke
    pt_file_path_nuke = pt_file_path.replace('\\', '/')
    cat_file_path_nuke = cat_file_path.replace('\\', '/')

    # Extract a base name for node labels (optional, makes it clearer)
    base_name = os.path.splitext(os.path.basename(pt_file_path))[0]

    nuke_script_content = f"""#! C:/Program Files/Nuke15.1v3/Nuke15.1.exe -nx
# Nuke script generated by inference_nuke_converter.py
# Use this script to load the converted model and create the .cat file.

set cut_paste_input [stack 0]
version 15.1 v3
push $cut_paste_input
# --- Input Processing (Assumes scene_linear input) ---
OCIOColorSpace {{
 in_colorspace scene_linear
 out_colorspace "Output - Rec.709"
 name OCIOColorSpace_In_{base_name}
 xpos 0
 ypos -100
}}
# --- The Inference Node (Uses the .cat file) ---
Inference {{
  modelFile "{cat_file_path_nuke}"
  serialiseKnob {{}} # Keep empty unless specific knobs need saving
  name Inference_{base_name}
  xpos 0
  ypos -45
}}
# --- Output Processing (Converts back to scene_linear) ---
OCIOColorSpace {{
 in_colorspace "Output - Rec.709"
 out_colorspace scene_linear
 name OCIOColorSpace_Out_{base_name}
 selected true
 xpos 0
 ypos 16
}}

# --- CatFileCreator (Creates the .cat file from .pt) ---
# Connect this node's output (optional) if you want to see the model info
# Execute this node once (e.g., by viewing it) to generate the .cat file
# The Inference node above requires the .cat file to exist.
CatFileCreator {{
 inputs 0
 torchScriptFile "{pt_file_path_nuke}"
 catFile "{cat_file_path_nuke}"
 channelsIn "rgba.red, rgba.green, rgba.blue"
 channelsOut "rgba.red, rgba.green, rgba.blue"
 modelId 1
 name CatFileCreator_{base_name}
 xpos 250 # Position it to the side
 ypos -77
 postage_stamp false
}}
"""
    try:
        nk_dir = os.path.dirname(nk_file_path)
        if nk_dir: os.makedirs(nk_dir, exist_ok=True) # Ensure directory exists
        with open(nk_file_path, 'w') as f:
            f.write(nuke_script_content)
        logging.info(f"Nuke script generated successfully at: {nk_file_path}")
        logging.info(f" -> Note: Open this .nk script in Nuke and execute 'CatFileCreator_{base_name}' node once to generate the required '{os.path.basename(cat_file_path)}' file.")
    except Exception as e:
        logging.error(f"Failed to write Nuke script file: {e}", exc_info=True)


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.warning("="*60 + "\nIMPORTANT: Nuke's ML nodes require specific, often older, PyTorch versions.\n" + f"You are using PyTorch {torch.__version__}.\nCheck Foundry docs for the recommended version for YOUR Nuke version.\n" + "You may need a separate Python environment for this conversion.\n" + "="*60)
    time.sleep(2)
    device = torch.device('cpu') # Convert to TorchScript on CPU

    try:
        model, resolution = load_model_for_conversion(args.checkpoint_pth, device)
    except Exception as e:
        logging.error(f"Failed to load and wrap model: {e}", exc_info=True); exit(1)

    # --- Determine Output .PT Path ---
    if args.output_pt is None:
        checkpoint_dir = os.path.dirname(args.checkpoint_pth)
        checkpoint_basename_no_ext = os.path.splitext(os.path.basename(args.checkpoint_pth))[0]
        # Ensure the base name doesn't somehow end in .pth if splitext failed unusually
        if checkpoint_basename_no_ext.endswith('.pth'):
            checkpoint_basename_no_ext = checkpoint_basename_no_ext[:-4]
        final_output_pt_path = os.path.join(checkpoint_dir, f"{checkpoint_basename_no_ext}.pt")
        logging.info(f"--output_pt not specified, defaulting to: {final_output_pt_path}")
    else:
        final_output_pt_path = args.output_pt
        logging.info(f"Using specified output .pt path: {final_output_pt_path}")

    # --- Determine Output .NK and required .CAT Path (if generating NK) ---
    final_output_nk_path = None
    final_output_cat_path = None # Path the CatFileCreator node will write to
    if args.generate_nk:
        # Base path without extension for nk and cat
        pt_dir = os.path.dirname(final_output_pt_path)
        pt_basename_no_ext = os.path.splitext(os.path.basename(final_output_pt_path))[0]

        # Determine .cat path (will be created by Nuke's CatFileCreator)
        final_output_cat_path = os.path.join(pt_dir, f"{pt_basename_no_ext}.cat")
        logging.info(f"Nuke script will configure CatFileCreator to output: {final_output_cat_path}")

        # Determine .nk path
        if args.output_nk is None:
            final_output_nk_path = os.path.join(pt_dir, f"{pt_basename_no_ext}.nk")
            logging.info(f"--output_nk not specified, defaulting to: {final_output_nk_path}")
        else:
            final_output_nk_path = args.output_nk
            logging.info(f"Using specified output .nk path: {final_output_nk_path}")

    # --- Tracing/Scripting Logic ---
    scripted_model = None
    conversion_method = args.method.lower()

    if conversion_method == 'script':
        logging.info("Attempting conversion using torch.jit.script...")
        try:
            scripted_model = torch.jit.script(model)
            logging.info("torch.jit.script successful!")
        except Exception as e:
            logging.error(f"torch.jit.script failed: {e}", exc_info=True)
            logging.warning("Try 'trace' method? Scripting often fails with complex control flow/modules.")
            exit(1)
    elif conversion_method == 'trace':
        logging.info("Attempting conversion using torch.jit.trace...")
        try:
            dummy_input = torch.rand(1, 3, resolution, resolution, device=device)
            logging.info(f"Using dummy input shape for tracing: {list(dummy_input.shape)} (Range [0,1])")
            scripted_model = torch.jit.trace(model, dummy_input, strict=False)
            logging.info("torch.jit.trace successful!")
        except Exception as e:
            logging.error(f"torch.jit.trace failed: {e}", exc_info=True)
            logging.warning("If model uses control flow internally that tracing missed, behavior might differ.")
            exit(1)
    else:
        logging.error(f"Invalid conversion method: {args.method}"); exit(1)

    # --- Saving Logic (.pt and optionally .nk) ---
    if scripted_model:
        try:
            # Save the .pt model
            output_dir = os.path.dirname(final_output_pt_path)
            if output_dir: os.makedirs(output_dir, exist_ok=True)
            scripted_model.save(final_output_pt_path)
            logging.info(f"TorchScript model saved to: {final_output_pt_path}")
            logging.info("Use this .pt file with Nuke's CatFileCreator.")

            # Generate the .nk script if requested
            if args.generate_nk:
                if final_output_nk_path and final_output_cat_path:
                    generate_nuke_script(final_output_pt_path, final_output_cat_path, final_output_nk_path)
                else:
                    # This should not happen if logic above is correct, but as a safeguard:
                    logging.error("Could not determine paths required for .nk generation. Skipping.")

        except Exception as e:
            logging.error(f"Failed during saving process: {e}", exc_info=True); exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert trained TuNet .pth checkpoint to TorchScript .pt for Nuke (with internal normalization) and optionally generate a basic Nuke .nk script.')
    parser.add_argument('--checkpoint_pth', type=str, required=True, help='Path to input .pth checkpoint (e.g., tunet_latest.pth)')
    parser.add_argument('--output_pt', type=str, default=None, required=False, help='Path to save output TorchScript .pt file. Defaults to same name/dir as checkpoint with .pt extension.')
    parser.add_argument('--method', type=str, default='script', choices=['script', 'trace'], help="Conversion method ('script' or 'trace')")
    # --- New Arguments ---
    parser.add_argument('--generate_nk', action='store_true', help='Generate a basic Nuke (.nk) script alongside the .pt file.')
    parser.add_argument('--output_nk', type=str, default=None, required=False, help='Optional path to save generated Nuke script (.nk) file. Defaults to same name/dir as the .pt file with .nk extension. Only used if --generate_nk is specified.')

    args = parser.parse_args()
    main(args)

import os
import argparse
import json
import logging
import torch
import torch.nn as nn
import torch.onnx
from types import SimpleNamespace # Add import

# --- Helper to convert nested dict to nested SimpleNamespace ---
# (Copied from train.py)
def dict_to_namespace(d):
    if isinstance(d,dict):
        safe_d={};
        for k,v in d.items(): safe_key=k.replace('-','_'); safe_d[safe_key]=dict_to_namespace(v)
        return SimpleNamespace(**safe_d)
    elif isinstance(d,list): return [dict_to_namespace(item) for item in d]
    else: return d

# --- UNet Model Definition ---
class DoubleConv(nn.Module):
    # ... ...
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        mid_channels = max(1, mid_channels) # Ensure mid_channels is at least 1
        self.d = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True) )
    def forward(self, x): return self.d(x)

class Down(nn.Module):
    # ... ...
    def __init__(self, in_channels, out_channels):
        super().__init__(); self.m = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.m(x)

class Up(nn.Module):
    # ... ...
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__(); self.bilinear = bilinear
        if bilinear: self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True); conv_in_channels = in_channels + skip_channels
        else: up_out_channels = in_channels // 2; self.up = nn.ConvTranspose2d(in_channels, up_out_channels, kernel_size=2, stride=2); conv_in_channels = up_out_channels + skip_channels
        self.conv = DoubleConv(conv_in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1); diffY = x2.size(2) - x1.size(2); diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0: x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1); return self.conv(x)

class OutConv(nn.Module):
    # ... ...
    def __init__(self, in_channels, out_channels): super().__init__(); self.c = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x): return self.c(x)

# <-- CHANGED: UNet class uses model_size_dims -->
class UNet(nn.Module):
    def __init__(self, config, n_ch=3, n_cls=3, bilinear=True):
        super().__init__(); self.n_ch = n_ch; self.n_cls = n_cls;
        # Get model size from the passed config object using the new key
        self.hidden_size = config.model.model_size_dims # new key
        self.bilinear = bilinear
        h = self.hidden_size; chs = {'enc1': h, 'enc2': h*2, 'enc3': h*4, 'enc4': h*8, 'bottle': h*16}
        self.inc = DoubleConv(n_ch, chs['enc1']); self.down1 = Down(chs['enc1'], chs['enc2']); self.down2 = Down(chs['enc2'], chs['enc3']); self.down3 = Down(chs['enc3'], chs['enc4']); self.down4 = Down(chs['enc4'], chs['bottle'])
        self.up1 = Up(chs['bottle'], chs['enc4'], chs['enc4'], bilinear); self.up2 = Up(chs['enc4'], chs['enc3'], chs['enc3'], bilinear); self.up3 = Up(chs['enc3'], chs['enc2'], chs['enc2'], bilinear); self.up4 = Up(chs['enc2'], chs['enc1'], chs['enc1'], bilinear)
        self.outc = OutConv(chs['enc1'], n_cls)
    def forward(self, x):
        x1=self.inc(x); x2=self.down1(x1); x3=self.down2(x2); x4=self.down3(x3); x5=self.down4(x4)
        x=self.up1(x5, x4); x=self.up2(x, x3); x=self.up3(x, x2); x=self.up4(x, x1); return self.outc(x)


# --- Normalization Wrapper ---
# ... (same) ...
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

# --- Modified load_model_for_export ---
def load_model_for_export(checkpoint_path, device):
    """Loads TuNet, wraps it for normalization, detecting model size.""" # description
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    # Load onto CPU first might be safer if checkpoint is large or from untrusted source
    # Set weights_only=False as we need the config/args object.
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

    # Extract Parameters Safely
    default_hidden_size = 64; default_loss = 'l1'; default_resolution = 512; default_bilinear = True

    # <-- CHANGED: Extract model size with fallback -->
    model_size_saved = default_hidden_size
    if is_new_format:
        model_config = getattr(config_source, 'model', SimpleNamespace())
        model_size_saved = getattr(model_config, 'model_size_dims', getattr(model_config, 'unet_hidden_size', default_hidden_size))
        loss_mode = getattr(getattr(config_source, 'training', SimpleNamespace()), 'loss', default_loss)
        resolution = getattr(getattr(config_source, 'data', SimpleNamespace()), 'resolution', default_resolution)
        bilinear_mode = default_bilinear
    else: # Old format (args)
        model_size_saved = getattr(config_source, 'model_size_dims', getattr(config_source, 'unet_hidden_size', default_hidden_size))
        loss_mode = getattr(config_source, 'loss', default_loss)
        resolution = getattr(config_source, 'resolution', default_resolution)
        bilinear_mode = getattr(config_source, 'bilinear', default_bilinear)

    logging.info(f"Checkpoint params: Saved Model Size={model_size_saved}, Loss='{loss_mode}', Res={resolution}, Bilinear={bilinear_mode}")

    # Calculate Effective Hidden Size
    hq_default_bump_size = 96; default_size_for_bump = 64 # Match training script
    effective_model_size = model_size_saved
    if loss_mode == 'l1+lpips' and model_size_saved == default_size_for_bump:
        effective_model_size = hq_default_bump_size
        logging.info(f"Applied model size bump logic: Effective Size = {effective_model_size}")
    else:
        logging.info(f"Effective Model Size = {effective_model_size}")

    # Instantiate the ORIGINAL UNet using a minimal config
    # <-- CHANGED: Create minimal config with the NEW key name -->
    minimal_unet_config = SimpleNamespace(
        model=SimpleNamespace(model_size_dims=effective_model_size)
    )
    base_model = UNet(config=minimal_unet_config, n_ch=3, n_cls=3, bilinear=bilinear_mode)

    # Load State Dict into base model
    state_dict = checkpoint['model_state_dict']
    if all(key.startswith('module.') for key in state_dict):
        logging.info("Removing 'module.' prefix from state_dict keys.")
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    base_model.load_state_dict(state_dict)

    # Instantiate the WRAPPER
    wrapped_model = NormalizedUNet(base_model)
    wrapped_model.to(device) # Move wrapped model (including base) to device
    wrapped_model.eval()     # Set wrapper to eval mode

    logging.info("Model loaded, wrapped for normalization, and set to evaluation mode.")
    return wrapped_model, resolution

# --- JSON Generation Function ---
# ... (sa............me) ...
def generate_flame_json(onnx_base_name, resolution, output_json_path, model_name=None, model_desc=""):
    if model_name is None: model_name = onnx_base_name
    input_name = "input_image"; output_name = "output_image"
    flame_data = {
        "ModelDescription": {
            "MinimumVersion": "2025.1", "Name": model_name,
            "Description": model_desc + " (Internal Norm: [0,1] -> [-1,1] -> [0,1])",
            "SupportsSceneLinear": False, "KeepAspectRatio": False, "Padding": 1,
            "Inputs": [{ "Name": input_name, "Description": "Source Image ([0,1] Range)", "Type": "Front", "Gain": 1.0, "Channels": "RGB" }],
            "Outputs": [{ "Name": output_name, "Description": "Processed Image ([0,1] Range)", "Type": "Result", "InverseGain": 1.0, "ScalingFactor": 1.0, "Channels": "RGB" }]
        } }
    try:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f: json.dump(flame_data, f, indent=4)
        logging.info(f"Generated Flame JSON configuration (Internal Norm): {output_json_path}")
    except Exception as e: logging.error(f"Failed to write JSON file: {e}")

# --- Main Conversion Function ---
# ... (remains the same) ...
def convert(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    if not os.path.exists(args.checkpoint): logging.error(f"Checkpoint file not found: {args.checkpoint}"); return
    checkpoint_dir = os.path.dirname(args.checkpoint); checkpoint_basename = os.path.basename(args.checkpoint)
    base_name = os.path.splitext(checkpoint_basename)[0]
    output_dir = args.output_dir if args.output_dir else checkpoint_dir
    output_onnx_path = args.output_onnx if args.output_onnx else os.path.join(output_dir, f"{base_name}.onnx")
    output_json_path = args.output_json if args.output_json else os.path.join(output_dir, f"{base_name}.json")
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    logging.info(f"Input Checkpoint: {args.checkpoint}")
    logging.info(f"Output ONNX Path: {output_onnx_path}")
    logging.info(f"Output JSON Path: {output_json_path}")

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    logging.info(f"Using device: {device}")

    try:
        model, resolution = load_model_for_export(args.checkpoint, device)
    except Exception as e:
        logging.error(f"Failed to load and wrap model: {e}", exc_info=True); return

    dummy_input = torch.rand(1, 3, resolution, resolution, device=device)
    input_names = ["input_image"]; output_names = ["output_image"]
    logging.info(f"Starting ONNX export (with internal normalization) to: {output_onnx_path}")
    try:
        torch.onnx.export(
            model, dummy_input, output_onnx_path, export_params=True, opset_version=args.opset,
            do_constant_folding=True, input_names=input_names, output_names=output_names,
            dynamic_axes={ input_names[0]: {0: 'batch_size'}, output_names[0]: {0: 'batch_size'} } if args.dynamic_batch else None )
        logging.info("ONNX export completed successfully.")

        json_base_name = os.path.splitext(os.path.basename(output_onnx_path))[0]
        generate_flame_json( json_base_name, resolution, output_json_path, args.model_name, args.model_desc )
    except Exception as e:
        logging.error(f"ONNX export or JSON generation failed: {e}", exc_info=True)


# --- Main Execution ---
# ... (remains the same) ...
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PyTorch TuNet checkpoint to ONNX for Flame (with internal normalization).') # Updated description
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the PyTorch checkpoint (.pth)')
    parser.add_argument('--output_onnx', type=str, default=None, help='Path to save the output ONNX model (.onnx). Defaults to same name/dir as checkpoint.')
    parser.add_argument('--output_json', type=str, default=None, help='Path to save the Flame JSON sidecar file (.json). Defaults to same name/dir as checkpoint.')
    parser.add_argument('--output_dir', type=str, default=None, help='Optional directory to save outputs if not specified explicitly. Overrides checkpoint directory.')
    parser.add_argument('--opset', type=int, default=14, help='ONNX opset version (e.g., 11, 12, 14, 17)')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for loading model if available (recommended)')
    parser.add_argument('--dynamic_batch', action='store_true', help='Allow dynamic batch size in the ONNX model')
    parser.add_argument('--model_name', type=str, default=None, help='Name for the model in Flame UI (defaults to ONNX filename base)')
    parser.add_argument('--model_desc', type=str, default="TuNet model converted from PyTorch.", help='Description for the model in Flame UI') # Updated default desc
    args = parser.parse_args()
    convert(args)

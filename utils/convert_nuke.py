# converter.py (Revised for Cross-Platform support - Minimal Change)

import os
import argparse
import logging
import torch
import torch.nn as nn
import time # For the sleep warning
from types import SimpleNamespace # Ensure this is imported

# --- dict direto rapido SimpleNamespace ---
def dict_to_namespace(d):
    """Converts a dictionary potentially containing nested dictionaries
       into a SimpleNamespace potentially containing nested SimpleNamespaces.
       Replaces hyphens in keys with underscores.
    """
    if isinstance(d, dict):
        safe_d = {}
        for key, value in d.items():
            # Replace hyphens, ensure identifier validity (basic check)
            safe_key = str(key).replace('-', '_')
            if not safe_key.isidentifier():
                 logging.warning(f"Config key '{key}' -> '{safe_key}' might not be a valid identifier.")
            safe_d[safe_key] = dict_to_namespace(value)
        return SimpleNamespace(**safe_d)
    elif isinstance(d, (list, tuple)):
        # Recursively convert items in lists/tuples
        return type(d)(dict_to_namespace(item) for item in d)
    else:
        # Return non-dict/list/tuple items as is
        return d

# --- Model Definition ---

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        mid_channels = max(1, mid_channels) # Ensure mid_channels is at least 1
        self.d_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.d_block(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.m_block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x): return self.m_block(x)

class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv_in_channels = in_channels + skip_channels
        else:
            in_channels_safe = max(1, in_channels)
            up_out_channels = max(1, in_channels_safe // 2)
            self.up = nn.ConvTranspose2d(in_channels_safe, up_out_channels, kernel_size=2, stride=2)
            conv_in_channels = up_out_channels + skip_channels
        conv_in_channels = max(1, conv_in_channels)
        out_channels = max(1, out_channels)
        self.conv = DoubleConv(conv_in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            pad_left = diffX // 2
            pad_right = diffX - pad_left
            pad_top = diffY // 2
            pad_bottom = diffY - pad_top
            if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
                 # --- IMPORTANT ---
                 # logging.warning(...) call removed/commented below because
                 # torch.jit.script cannot compile functions with *args/**kwargs
                 # like those in the standard logging library.
                 # Tracing (--method trace) usually avoids this issue.
                 # logging.warning(f"Negative padding calculated in Up block: L{pad_left},R{pad_right},T{pad_top},B{pad_bottom}. Skipping padding.")
                 pass # Just skip padding if dimensions are negative
            else:
                 x1 = nn.functional.pad(x1, [pad_left, pad_right, pad_top, pad_bottom])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Ensure positive channel numbers
        in_channels_safe = max(1, in_channels)
        out_channels_safe = max(1, out_channels)
        self.c_block = nn.Conv2d(in_channels_safe, out_channels_safe, kernel_size=1)
    def forward(self, x): return self.c_block(x)

# class definition matching 
# Takes hidden_size directly
class UNet(nn.Module):
    def __init__(self, n_ch=3, n_cls=3, hidden_size=64, bilinear=True):
        super().__init__()
        if n_ch <= 0 or n_cls <= 0 or hidden_size <= 0: raise ValueError("n_ch, n_cls, hidden_size must be positive")
        self.n_ch, self.n_cls, self.hidden_size, self.bilinear = n_ch, n_cls, hidden_size, bilinear
        h = hidden_size # Use the passed hidden_size

        # Ensure all channel counts are at least 1
        chs = {'enc1': max(1, h), 'enc2': max(1, h*2), 'enc3': max(1, h*4), 'enc4': max(1, h*8), 'bottle': max(1, h*16)}
        bottle_in_ch = chs['bottle']

        # Use max(1, n_ch) for input channels safety
        self.inc = DoubleConv(max(1, n_ch), chs['enc1'])
        self.down1 = Down(chs['enc1'], chs['enc2'])
        self.down2 = Down(chs['enc2'], chs['enc3'])
        self.down3 = Down(chs['enc3'], chs['enc4'])
        self.down4 = Down(chs['enc4'], bottle_in_ch) # Input to bottle neck is enc4 output

        # Ensure skip connection channel sizes match corresponding down levels
        self.up1 = Up(bottle_in_ch, chs['enc4'], chs['enc4'], bilinear)
        self.up2 = Up(chs['enc4'],   chs['enc3'], chs['enc3'], bilinear)
        self.up3 = Up(chs['enc3'],   chs['enc2'], chs['enc2'], bilinear)
        self.up4 = Up(chs['enc2'],   chs['enc1'], chs['enc1'], bilinear)
        self.outc = OutConv(chs['enc1'], max(1, n_cls)) # Ensure output channels are positive

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# --- Normalization Wrapper ---
class NormalizedUNet(nn.Module):
    def __init__(self, unet_model):
        super().__init__()
        self.unet = unet_model
        # Use buffers for constants that should be part of the state_dict
        # and moved to the correct device automatically.
        self.register_buffer('mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    def forward(self, x):
        # Normalize: [0, 1] -> [-1, 1]
        normalized_x = (x - self.mean) / self.std
        # Pass through the underlying UNet
        unet_output = self.unet(normalized_x)
        # Denormalize: [-1, 1] -> [0, 1]
        denormalized_output = (unet_output * self.std) + self.mean
        # Clamp the final output to [0, 1] range
        clamped_output = torch.clamp(denormalized_output, 0.0, 1.0)
        return clamped_output

# --- To DO load_model_for_conversion ---
def load_model_for_conversion(checkpoint_path, device='cpu'):
    """Loads TuNet checkpoint, determines effective model size from metadata,
       instantiates the model, loads weights, and wraps it for normalization.
       Handles both new ('config') and old ('args') metadata formats.
    """
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    # Load onto CPU first, ensure weights_only=False to access metadata
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # --- Determine Config Source (New vs Old Format) ---
    config_source_obj = None
    is_new_format = False
    config_format_detected = "Unknown"
    if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
        config_format_detected = "New (dict under 'config' key)"
        logging.info(f"Detected format: {config_format_detected}")
        try:
            config_source_obj = dict_to_namespace(checkpoint['config'])
            is_new_format = True
            logging.debug("Successfully converted config dict to SimpleNamespace.")
        except Exception as e:
            logging.error(f"Failed to convert config dict to SimpleNamespace: {e}", exc_info=True)
            raise ValueError("Checkpoint contains 'config' dict, but conversion failed.")
    elif 'args' in checkpoint and isinstance(checkpoint['args'], argparse.Namespace):
        config_format_detected = "Old (argparse.Namespace under 'args' key)"
        logging.warning(f"Detected format: {config_format_detected}. Attempting compatibility.")
        # Convert argparse.Namespace to SimpleNamespace for consistent access
        config_source_obj = SimpleNamespace(**vars(checkpoint['args']))
        is_new_format = False # Mark as old format
    else:
        # Handle case where neither 'config' (dict) nor 'args' (Namespace) is found
        config_key_present = 'config' in checkpoint
        args_key_present = 'args' in checkpoint
        err_msg = "Checkpoint missing required configuration metadata. "
        if config_key_present: err_msg += f"Found 'config' key, but it's not a dictionary (type: {type(checkpoint.get('config'))}). "
        if args_key_present: err_msg += f"Found 'args' key, but it's not an argparse.Namespace (type: {type(checkpoint.get('args'))}). "
        if not config_key_present and not args_key_present: err_msg += "Neither 'config' (dict) nor 'args' (Namespace) key found. "
        logging.error(err_msg)
        raise ValueError(err_msg + "Cannot determine model parameters.")

    if config_source_obj is None:
         raise ValueError("Configuration object could not be loaded or created from checkpoint.")

    # --- Extract Parameters Safely using the loaded config_source_obj ---
    logging.info("Extracting parameters from checkpoint metadata...")
    default_hidden_size = 64
    default_loss = 'l1'
    default_resolution = 512
    default_bilinear = True # Bilinear is usually True by default in UNet

    # Use getattr for safe access with defaults
    model_config = getattr(config_source_obj, 'model', SimpleNamespace())
    training_config = getattr(config_source_obj, 'training', SimpleNamespace())
    data_config = getattr(config_source_obj, 'data', SimpleNamespace())

    # Get model size: Prioritize 'model_size_dims', fallback to 'unet_hidden_size'
    model_size_saved = getattr(model_config, 'model_size_dims', default_hidden_size)
    size_source = "'model.model_size_dims'"
    if model_size_saved == default_hidden_size: # If primary key wasn't found or was default, try fallback
        legacy_size = getattr(model_config, 'unet_hidden_size', default_hidden_size)
        # Only use legacy if it's different from default AND primary was default
        if legacy_size != default_hidden_size:
             model_size_saved = legacy_size
             size_source = "'model.unet_hidden_size' (legacy fallback)"
        else:
             size_source += " (or fallback, using default)"
    logging.info(f"  - Saved Model Size: {model_size_saved} (from {size_source})")


    # Get loss mode: Prioritize 'training.loss'. Handle legacy 'quality' for old format.
    loss_mode = getattr(training_config, 'loss', None)
    loss_source = "'training.loss'"
    if loss_mode is None and not is_new_format: # Only check legacy if new format key missing AND it's old format
        quality_legacy = getattr(config_source_obj, 'quality', None) # Check top-level 'quality' in old args
        if quality_legacy == 'HQ':
            loss_mode = 'l1+lpips'
            loss_source = "'quality'=='HQ' (legacy)"
        elif quality_legacy == 'LQ':
            loss_mode = 'l1'
            loss_source = "'quality'=='LQ' (legacy)"
    # If still None, use default
    if loss_mode is None:
        loss_mode = default_loss
        loss_source += " (not found, using default)"
    logging.info(f"  - Loss Mode: '{loss_mode}' (from {loss_source})")

    # Get resolution: Prioritize 'data.resolution'
    resolution = getattr(data_config, 'resolution', default_resolution)
    res_source = "'data.resolution'"
    if resolution == default_resolution: # If primary key wasn't found or was default, try top-level fallback (old)
        legacy_res = getattr(config_source_obj, 'resolution', default_resolution)
        if legacy_res != default_resolution:
            resolution = legacy_res
            res_source = "'resolution' (legacy fallback)"
        else:
            res_source += " (or fallback, using default)"
    logging.info(f"  - Resolution: {resolution} (from {res_source})")


    # Get bilinear mode (less likely to be explicitly set, default is common)
    bilinear_mode = getattr(model_config, 'bilinear', default_bilinear)
    bilinear_source = "'model.bilinear'"
    if bilinear_mode == default_bilinear: # Try top-level fallback (old)
         legacy_bilinear = getattr(config_source_obj, 'bilinear', default_bilinear)
         if legacy_bilinear != default_bilinear:
             bilinear_mode = legacy_bilinear
             bilinear_source = "'bilinear' (legacy fallback)"
         else:
             bilinear_source += " (or fallback, using default)"
    logging.info(f"  - Bilinear Mode: {bilinear_mode} (from {bilinear_source})")

    # --- Calculate Effective Hidden Size (Replicating train.py logic) ---
    hq_default_bump_size = 96  # Size to bump to for LPIPS if starting at default
    default_size_for_bump = 64 # Default size that triggers the bump with LPIPS

    effective_model_size = model_size_saved
    logging.info(f"Calculating effective model size (Base: {model_size_saved}, Loss: '{loss_mode}')...")
    if loss_mode == 'l1+lpips' and model_size_saved == default_size_for_bump:
        effective_model_size = hq_default_bump_size
        logging.info(f" -> LPIPS loss detected with base size {default_size_for_bump}. Applied model size bump.")
    else:
        logging.info(f" -> No size bump applied.")
    logging.info(f"Effective Model Size for Instantiation: {effective_model_size}")

    # --- Instantiate the Base UNet Model ---
    # Instantiate the UNet defined *in this script*, passing the calculated effective size directly
    logging.info(f"Instantiating UNet with effective_hidden_size={effective_model_size}, bilinear={bilinear_mode}")
    try:
        base_model = UNet(
            n_ch=3,
            n_cls=3,
            hidden_size=effective_model_size, # Pass the calculated effective size
            bilinear=bilinear_mode
        )
    except Exception as e:
        logging.error(f"Failed to instantiate UNet model: {e}", exc_info=True)
        raise RuntimeError(f"UNet instantiation failed with effective size {effective_model_size}") from e


    # --- Load State Dict into base model ---
    if 'model_state_dict' not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")

    state_dict = checkpoint['model_state_dict']

    # Check for and remove 'module.' prefix if checkpoint was saved from DDP
    if all(key.startswith('module.') for key in state_dict):
        logging.info("Detected 'module.' prefix in state_dict keys (likely DDP checkpoint). Removing prefix.")
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    elif any(key.startswith('module.') for key in state_dict):
         logging.warning("Mixed state_dict keys found (some with 'module.', some without). Attempting to load anyway...")

    logging.info("Loading state dict into the base UNet model (strict=True)...")
    try:
        base_model.load_state_dict(state_dict, strict=True)
        logging.info("State dict loaded successfully (strict=True).")
    except RuntimeError as e:
        logging.error(f"Failed to load state_dict with strict=True: {e}")
        logging.info("Attempting to load state_dict with strict=False...")
        try:
             # Fallback to strict=False
             incompatible_keys = base_model.load_state_dict(state_dict, strict=False)
             if incompatible_keys.missing_keys:
                 logging.warning(f"Missing keys found during non-strict load: {incompatible_keys.missing_keys}")
             if incompatible_keys.unexpected_keys:
                 logging.warning(f"Unexpected keys found during non-strict load: {incompatible_keys.unexpected_keys}")
             logging.warning("State dict loaded with strict=False. Model might not behave as expected if keys were incompatible.")
        except Exception as e_nonstrict:
             logging.error(f"Failed to load state_dict even with strict=False: {e_nonstrict}", exc_info=True)
             raise RuntimeError("Could not load model state_dict.") from e_nonstrict
    except Exception as e:
        logging.error(f"An unexpected error occurred during state_dict loading: {e}", exc_info=True)
        raise RuntimeError("Could not load model state_dict.") from e


    # --- Instantiate the WRAPPER ---
    logging.info("Wrapping UNet model with Normalization layer...")
    wrapped_model = NormalizedUNet(base_model)

    # Move the entire wrapped model (including the base model) to the target device
    logging.info(f"Moving wrapped model to device: {device}")
    wrapped_model.to(device)

    # Set the wrapper (and underlying model) to evaluation mode
    wrapped_model.eval()
    logging.info("Model loaded, wrapped for normalization, moved to device, and set to evaluation mode.")

    # Return the wrapped model and the resolution determined from the config
    return wrapped_model, resolution


# --- Function to Generate Nuke Script (remains the same) ---
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
# --- Input Processing (Rec.709 for model) ---
OCIOColorSpace {{
 in_colorspace scene_linear
 out_colorspace "Output - Rec.709"
 name OCIOColorSpace_In_{base_name}
 label "scene\\_linear -> Rec.709"
 xpos 0
 ypos -100
}}
# --- The Inference Node (Uses the .cat file generated below) ---
Inference {{
  modelFile "{cat_file_path_nuke}"
  serialiseKnob {{}} # Keep empty unless specific knobs need saving
  name Inference_{base_name}
  label "{os.path.basename(cat_file_path_nuke)}"
  xpos 0
  ypos -45
}}
# --- Output Processing (Rec.709 -> scene_linear) ---
OCIOColorSpace {{
 in_colorspace "Output - Rec.709"
 out_colorspace scene_linear
 name OCIOColorSpace_Out_{base_name}
 label "Rec.709 -> scene\\_linear"
 selected true
 xpos 0
 ypos 16
}}

# --- CatFileCreator (For TuNet model by tpo) ---
# Connect this node's output (optional) if you want to see the model info
# Execute this node once (e.g., by viewing it or right-click -> Execute)
# to generate the .cat file at the path specified below.
# The Inference node above requires this .cat file to exist.
CatFileCreator {{
 inputs 0
 torchScriptFile "{pt_file_path_nuke}"
 catFile "{cat_file_path_nuke}"
 channelsIn "rgba.red, rgba.green, rgba.blue"
 channelsOut "rgba.red, rgba.green, rgba.blue"
 modelId 1
 name CatFileCreator_TuNet by tpo_{base_name}
 label "TuNet by tpo\\nInput: {os.path.basename(pt_file_path_nuke)}\\nOutput: {os.path.basename(cat_file_path_nuke)}"
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
        logging.info(f" -> Open this .nk script in Nuke.")
        logging.info(f" -> Locate the 'CatFileCreator_{base_name}' node.")
        logging.info(f" -> Create .cat file.")
        logging.info(f" -> This will generate the required '{os.path.basename(cat_file_path)}' file.")
        logging.info(f" -> The 'Inference_{base_name}' node can then use the .cat file.")
    except Exception as e:
        logging.error(f"Failed to write Nuke script file: {e}", exc_info=True)


# --- Main Conversion Function ---
def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.warning("="*60)
    logging.warning("IMPORTANT: Nuke's ML nodes require specific PyTorch versions.")
    logging.warning(f"You are using PyTorch {torch.__version__}.")
    logging.warning("Check Foundry documentation for the recommended version for YOUR Nuke version.")
    logging.warning("Conversion might fail or the model may not load in Nuke if versions mismatch.")
    logging.warning("Consider using a dedicated Python environment matching Nuke's requirements.")
    logging.warning("="*60)
    time.sleep(3) # Give user time to read warning

    # Convert to TorchScript on CPU is generally safer and recommended
    device = torch.device('cpu')
    logging.info(f"Using device for model loading (pre-scripting): {device}")

    # Load the model using the updated function
    try:
        model, resolution = load_model_for_conversion(args.checkpoint_pth, device)
        logging.info(f"Model loaded successfully. Detected resolution for trace (if used): {resolution}x{resolution}")
    except Exception as e:
        logging.error(f"Fatal error during model loading: {e}", exc_info=True)
        exit(1) # Exit if model loading fails

    # --- Determine Output .PT Path ---
    if args.output_pt is None:
        checkpoint_dir = os.path.dirname(args.checkpoint_pth)
        checkpoint_basename = os.path.basename(args.checkpoint_pth)
        checkpoint_basename_no_ext = os.path.splitext(checkpoint_basename)[0]
        # Ensure the base name doesn't somehow end in .pth (e.g. if input was already .pt)
        if checkpoint_basename_no_ext.endswith('.pth'):
            checkpoint_basename_no_ext = checkpoint_basename_no_ext[:-4]
        final_output_pt_path = os.path.join(checkpoint_dir, f"{checkpoint_basename_no_ext}.pt")
        logging.info(f"--output_pt not specified, defaulting to: {final_output_pt_path}")
    else:
        final_output_pt_path = args.output_pt
        logging.info(f"Using specified output .pt path: {final_output_pt_path}")

    # --- Determine Output .NK and required .CAT Path (if generating NK) ---
    final_output_nk_path = None
    final_output_cat_path = None # Path the CatFileCreator node will WRITE to
    if args.generate_nk:
        # Base path without extension for nk and cat, derived from the final .pt path
        pt_dir = os.path.dirname(final_output_pt_path)
        pt_basename_no_ext = os.path.splitext(os.path.basename(final_output_pt_path))[0]

        # Determine .cat path (this is the file Nuke's node will create)
        final_output_cat_path = os.path.join(pt_dir, f"{pt_basename_no_ext}.cat")
        logging.info(f"Nuke script will be configured with CatFileCreator outputting to: {final_output_cat_path}")

        # Determine .nk path (the script file we generate)
        if args.output_nk is None:
            final_output_nk_path = os.path.join(pt_dir, f"{pt_basename_no_ext}.nk")
            logging.info(f"--output_nk not specified, defaulting .nk script path to: {final_output_nk_path}")
        else:
            final_output_nk_path = args.output_nk
            logging.info(f"Using specified output .nk script path: {final_output_nk_path}")

    # --- Tracing/Scripting Logic ---
    scripted_model = None
    conversion_method = args.method.lower()
    logging.info(f"Starting TorchScript conversion using method: '{conversion_method}'...")

    # Ensure model is in eval mode before scripting/tracing
    model.eval()

    if conversion_method == 'script':
        logging.info("Attempting conversion using torch.jit.script...")
        try:
            scripted_model = torch.jit.script(model)
            logging.info("torch.jit.script successful!")
        except Exception as e:
            logging.error(f"torch.jit.script failed: {e}", exc_info=True)
            logging.error("Scripting often fails with complex control flow or third-party modules.")
            logging.error("Consider trying the 'trace' method instead.")
            exit(1)
    elif conversion_method == 'trace':
        logging.info("Attempting conversion using torch.jit.trace...")
        try:
            # Create dummy input on the same device the model is on (CPU in this case)
            # Shape: (batch_size, channels, height, width)
            dummy_input = torch.rand(1, 3, resolution, resolution, device=device)
            logging.info(f"Using dummy input shape for tracing: {list(dummy_input.shape)} (Range [0,1])")
            # strict=False is often needed for complex models or wrappers
            scripted_model = torch.jit.trace(model, dummy_input, strict=False)
            logging.info("torch.jit.trace successful!")
        except Exception as e:
            logging.error(f"torch.jit.trace failed: {e}", exc_info=True)
            logging.error("Tracing might miss internal control flow (if/loops).")
            logging.error("Ensure the dummy input matches the expected input format.")
            exit(1)
    else:
        # This case should not be reachable due to argparse choices
        logging.error(f"Invalid conversion method specified: {args.method}")
        exit(1)

    # --- Saving Logic (.pt and optionally .nk) ---
    if scripted_model:
        logging.info(f"Saving converted TorchScript model to: {final_output_pt_path}")
        try:
            # Ensure the output directory for the .pt file exists
            output_dir = os.path.dirname(final_output_pt_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Save the scripted/traced model
            scripted_model.save(final_output_pt_path)
            logging.info(f"TorchScript model (.pt) saved successfully.")
            logging.info(" -> This .pt file is needed for Nuke's CatFileCreator node.")

            # Generate the .nk script if requested AND paths were determined
            if args.generate_nk:
                if final_output_nk_path and final_output_cat_path:
                    logging.info("Generating Nuke script...")
                    generate_nuke_script(final_output_pt_path, final_output_cat_path, final_output_nk_path)
                else:
                    # Safeguard: This should not happen if path logic above is correct
                    logging.error("Could not determine final paths required for .nk script generation. Skipping NK script.")

            logging.info("Conversion process finished.")

        except Exception as e:
            logging.error(f"Failed during saving process (.pt or .nk): {e}", exc_info=True)
            # Attempt cleanup? Maybe not necessary unless file is huge.
            exit(1)
    else:
        # Should not happen if script/trace blocks exit on failure
        logging.error("Conversion resulted in no model object. Cannot save.")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert trained TuNet .pth checkpoint to TorchScript .pt for Nuke (handles new config format) and optionally generate a helper Nuke .nk script.')
    parser.add_argument('--checkpoint_pth', type=str, required=True, help='Path to input .pth checkpoint (e.g., tunet_latest.pth) from the UPDATED training script.')
    parser.add_argument('--output_pt', type=str, default=None, required=False, help='Path to save output TorchScript .pt file. Defaults to same name/dir as checkpoint with .pt extension.')
    parser.add_argument('--method', type=str, default='script', choices=['script', 'trace'], help="Conversion method ('script' or 'trace'). 'script' is generally preferred if it works, 'trace' is often more compatible but less flexible. Default: script")
    # --- Nuke Script Generation Args ---
    parser.add_argument('--generate_nk', action='store_true', help='Generate a basic Nuke (.nk) script alongside the .pt file to help set up CatFileCreator and Inference nodes.')
    parser.add_argument('--output_nk', type=str, default=None, required=False, help='Optional path to save the generated Nuke script (.nk) file. Defaults to same name/dir as the final .pt file with .nk extension. Only used if --generate_nk is specified.')

    args = parser.parse_args()
    main(args)

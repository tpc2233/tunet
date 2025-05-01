# converter.py (Revised for Cross-Platform support - Minimal Change)

import os
import argparse
import json
import logging
import torch
import torch.nn as nn
import torch.onnx
from types import SimpleNamespace

# --- Helper rapido dict_to_namespace 
def dict_to_namespace(d):
    if isinstance(d, dict):
        safe_d = {}
        for key, value in d.items():
            safe_key = str(key).replace('-', '_')
            if not safe_key.isidentifier():
                 # Use logging if available, otherwise print
                 log_func = logging.warning if logging.getLogger().hasHandlers() else print
                 log_func(f"Config key '{key}' -> '{safe_key}' might not be a valid identifier.")
            safe_d[safe_key] = dict_to_namespace(value)
        return SimpleNamespace(**safe_d)
    elif isinstance(d, (list, tuple)):
        return type(d)(dict_to_namespace(item) for item in d)
    else:
        return d

# --- Model Definition ---
# Ensure these definitions exactly match the ones used during training


class DoubleConv(nn.Module):
    # ... definition remains the same ...
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        mid_channels = max(1, mid_channels)
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
            pad_left = diffX // 2; pad_right = diffX - pad_left
            pad_top = diffY // 2; pad_bottom = diffY - pad_top
            if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0: pass
            else: x1 = nn.functional.pad(x1, [pad_left, pad_right, pad_top, pad_bottom])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        in_channels_safe = max(1, in_channels); out_channels_safe = max(1, out_channels)
        self.c_block = nn.Conv2d(in_channels_safe, out_channels_safe, kernel_size=1)
    def forward(self, x): return self.c_block(x)


# Takes hidden_size directly
class UNet(nn.Module):
    def __init__(self, n_ch=3, n_cls=3, hidden_size=64, bilinear=True):
        super().__init__()
        if n_ch <= 0 or n_cls <= 0 or hidden_size <= 0: raise ValueError("n_ch, n_cls, hidden_size must be positive")
        self.n_ch, self.n_cls, self.hidden_size, self.bilinear = n_ch, n_cls, hidden_size, bilinear
        h = hidden_size
        chs = {'enc1': max(1, h), 'enc2': max(1, h*2), 'enc3': max(1, h*4), 'enc4': max(1, h*8), 'bottle': max(1, h*16)}
        bottle_in_ch = chs['bottle']
        self.inc = DoubleConv(max(1, n_ch), chs['enc1'])
        self.down1 = Down(chs['enc1'], chs['enc2'])
        self.down2 = Down(chs['enc2'], chs['enc3'])
        self.down3 = Down(chs['enc3'], chs['enc4'])
        self.down4 = Down(chs['enc4'], bottle_in_ch)
        self.up1 = Up(bottle_in_ch, chs['enc4'], chs['enc4'], bilinear)
        self.up2 = Up(chs['enc4'],   chs['enc3'], chs['enc3'], bilinear)
        self.up3 = Up(chs['enc3'],   chs['enc2'], chs['enc2'], bilinear)
        self.up4 = Up(chs['enc2'],   chs['enc1'], chs['enc1'], bilinear)
        self.outc = OutConv(chs['enc1'], max(1, n_cls))

    def forward(self, x):
        x1=self.inc(x); x2=self.down1(x1); x3=self.down2(x2); x4=self.down3(x3); x5=self.down4(x4)
        x=self.up1(x5, x4); x=self.up2(x, x3); x=self.up3(x, x2); x=self.up4(x, x1); return self.outc(x)


# --- Normalization Wrapper ---
class NormalizedUNet(nn.Module):
    # ... definition remains the same ...
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
    """Loads TuNet checkpoint, determines model size *directly from metadata*,
       instantiates the model, loads weights, and wraps it for normalization.
    """
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # --- Determine Config Source (New vs Old Format) ---
    config_source_obj = None; is_new_format = False; config_format_detected = "Unknown"
    if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
        config_format_detected = "New (dict under 'config' key)"
        logging.info(f"Detected format: {config_format_detected}")
        try: config_source_obj = dict_to_namespace(checkpoint['config']); is_new_format = True
        except Exception as e: raise ValueError(f"Config dict conversion failed: {e}") from e
    elif 'args' in checkpoint and isinstance(checkpoint['args'], argparse.Namespace):
        config_format_detected = "Old (argparse.Namespace under 'args' key)"
        logging.warning(f"Detected format: {config_format_detected}. Attempting compatibility.")
        config_source_obj = SimpleNamespace(**vars(checkpoint['args'])); is_new_format = False
    else: raise ValueError("Checkpoint missing required configuration metadata ('config' dict or 'args' namespace).")
    if config_source_obj is None: raise ValueError("Conf obj could not be loaded.")

    # --- Extract Parameters Safely ---
    logging.info("Extracting parameters from checkpoint metadata...")
    default_hidden_size = 64; default_resolution = 512; default_bilinear = True
    model_config = getattr(config_source_obj, 'model', SimpleNamespace())
    data_config = getattr(config_source_obj, 'data', SimpleNamespace())

    # Get model size DIRECTLY from checkpoint metadata, using fallbacks
    model_size_saved = getattr(model_config, 'model_size_dims', default_hidden_size); size_source = "'model.model_size_dims'"
    if model_size_saved == default_hidden_size:
        legacy_size = getattr(model_config, 'unet_hidden_size', default_hidden_size)
        if legacy_size != default_hidden_size: model_size_saved = legacy_size; size_source = "'model.unet_hidden_size' (legacy fallback)"
        else: size_source += " (or fallback, using default)"
    logging.info(f"  - Using Model Size from Checkpoint: {model_size_saved} (loaded via {size_source})")

    # Extract resolution (needed for dummy input)
    resolution = getattr(data_config, 'resolution', default_resolution); res_source = "'data.resolution'"
    if resolution == default_resolution:
        legacy_res = getattr(config_source_obj, 'resolution', default_resolution)
        if legacy_res != default_resolution: resolution = legacy_res; res_source = "'resolution' (legacy fallback)"
        else: res_source += " (or fallback, using default)"
    logging.info(f"  - Resolution: {resolution} (from {res_source})")

    # Extract bilinear mode
    bilinear_mode = getattr(model_config, 'bilinear', default_bilinear); bilinear_source = "'model.bilinear'"
    if bilinear_mode == default_bilinear:
         legacy_bilinear = getattr(config_source_obj, 'bilinear', default_bilinear)
         if legacy_bilinear != default_bilinear: bilinear_mode = legacy_bilinear; bilinear_source = "'bilinear' (legacy fallback)"
         else: bilinear_source += " (or fallback, using default)"
    logging.info(f"  - Bilinear Mode: {bilinear_mode} (from {bilinear_source})")

    # Extract loss mode for logging purposes (not used for size calculation here)
    training_config = getattr(config_source_obj, 'training', SimpleNamespace())
    loss_mode = getattr(training_config, 'loss', 'l1'); loss_source = "'training.loss'"
    if loss_mode == 'l1' and not is_new_format:
        quality_legacy = getattr(config_source_obj, 'quality', None)
        if quality_legacy == 'HQ': loss_mode = 'l1+lpips'; loss_source = "'quality'=='HQ' (legacy)"
        elif quality_legacy == 'LQ': loss_mode = 'l1'; loss_source = "'quality'=='LQ' (legacy)"
    logging.info(f"  - Loss Mode (for info): '{loss_mode}' (from {loss_source})")

    # --- Instantiate the Base UNet Model ---
    logging.info(f"Instantiating UNet with hidden_size={model_size_saved} (from checkpoint metadata), bilinear={bilinear_mode}")
    try:
        base_model = UNet(n_ch=3, n_cls=3, hidden_size=model_size_saved, bilinear=bilinear_mode)
    except Exception as e:
        logging.error(f"Failed to instantiate UNet model: {e}", exc_info=True)
        raise RuntimeError(f"UNet instantiation failed with size {model_size_saved}") from e

    # --- Load State Dict ---
    if 'model_state_dict' not in checkpoint: raise KeyError("Checkpoint missing 'model_state_dict'.")
    state_dict = checkpoint['model_state_dict']
    if all(key.startswith('module.') for key in state_dict):
        logging.info("Removing 'module.' prefix from state_dict keys."); state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    elif any(key.startswith('module.') for key in state_dict): logging.warning("Mixed state_dict keys ('module.' prefix). Loading anyway...")
    logging.info("Loading state dict into UNet model (strict=True)...")
    try:
        base_model.load_state_dict(state_dict, strict=True)
        logging.info("State dict loaded successfully (strict=True).")
    except RuntimeError as e:
        logging.error(f"Failed state_dict load (strict=True): {e}")
        logging.info("Attempting load with strict=False...")
        try:
            incompatible_keys = base_model.load_state_dict(state_dict, strict=False)
            # <<< --- Rever correcao rapida --- >>>
            if incompatible_keys.missing_keys:
                logging.warning(f"Missing keys (strict=False): {incompatible_keys.missing_keys}")
            if incompatible_keys.unexpected_keys:
                logging.warning(f"Unexpected keys (strict=False): {incompatible_keys.unexpected_keys}")
            logging.warning("State dict loaded (strict=False). Check warnings.")
            # <<< --- TO DO rever --- >>>
        except Exception as e_nonstrict:
            logging.error(f"Failed load (strict=False): {e_nonstrict}", exc_info=True)
            raise RuntimeError("Could not load model state_dict.") from e_nonstrict
    except Exception as e:
        logging.error(f"Unexpected state_dict loading error: {e}", exc_info=True)
        raise RuntimeError("Could not load model state_dict.") from e

    # --- Instantiate the WRAPPER ---
    logging.info("Wrapping UNet model with Normalization layer...")
    wrapped_model = NormalizedUNet(base_model)
    logging.info(f"Moving wrapped model to device: {device}")
    wrapped_model.to(device)
    wrapped_model.eval()
    logging.info("Model ready (loaded, wrapped, device, eval mode).")

    return wrapped_model, resolution # Return resolution needed for dummy input

# --- JSON Generation Function  ---
def generate_flame_json(onnx_base_name, resolution, output_json_path, model_name=None, model_desc=""):
    if model_name is None: model_name = onnx_base_name
    input_name = "input_image"; output_name = "output_image"
    flame_data = { "ModelDescription": { "MinimumVersion": "2025.1", "Name": model_name, "Description": model_desc + " (Normalized: [0,1] input/output)", "SupportsSceneLinear": False, "KeepAspectRatio": False, "Padding": 1, "Inputs": [{ "Name": input_name, "Description": "Source Image ([0,1] Range)", "Type": "Front", "Gain": 1.0, "Channels": "RGB" }], "Outputs": [{ "Name": output_name, "Description": "Processed Image ([0,1] Range)", "Type": "Result", "InverseGain": 1.0, "ScalingFactor": 1.0, "Channels": "RGB" }] } }
    try:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f: json.dump(flame_data, f, indent=4)
        logging.info(f"Generated Flame JSON configuration: {output_json_path}")
    except Exception as e: logging.error(f"Failed to write Flame JSON file '{output_json_path}': {e}", exc_info=True)


# --- Main Conversion Function  ---
def convert(args):
    # ... Setup, path determination, device selection ... 
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if not os.path.isfile(args.checkpoint): logging.error(f"Checkpoint file not found: {args.checkpoint}"); return 1
    checkpoint_dir = os.path.dirname(args.checkpoint); checkpoint_basename = os.path.basename(args.checkpoint)
    base_name = os.path.splitext(checkpoint_basename)[0]
    output_dir = args.output_dir if args.output_dir else checkpoint_dir
    output_onnx_path = args.output_onnx if args.output_onnx else os.path.join(output_dir, f"{base_name}.onnx")
    output_json_path = args.output_json if args.output_json else os.path.join(output_dir, f"{base_name}.json")
    try: os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True); os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    except OSError as e: logging.error(f"Error creating output directories: {e}"); return 1
    logging.info(f"--- Conversion Settings ---"); logging.info(f"Input Checkpoint: {args.checkpoint}"); logging.info(f"Output ONNX Path: {output_onnx_path}"); logging.info(f"Output JSON Path: {output_json_path}"); logging.info(f"ONNX Opset: {args.opset}"); logging.info(f"Use GPU if available: {args.use_gpu}"); logging.info(f"Dynamic Batch Size: {args.dynamic_batch}"); logging.info(f"-------------------------")
    use_cuda = torch.cuda.is_available() and args.use_gpu; device = torch.device('cuda' if use_cuda else 'cpu')
    logging.info(f"Using device: {device}")

    # Load model using the revised function
    try:
        model, resolution = load_model_for_export(args.checkpoint, device)
        logging.info(f"Model loaded. Using resolution from checkpoint: {resolution}x{resolution}")
    except Exception as e: logging.error(f"Fatal error during model loading: {e}", exc_info=True); return 1

    # Create dummy input based on loaded resolution
    try: dummy_input = torch.rand(1, 3, resolution, resolution, device=device); logging.info(f"Created dummy input tensor: {list(dummy_input.shape)}")
    except Exception as e: logging.error(f"Failed to create dummy input (Res: {resolution}): {e}", exc_info=True); return 1

    # Define ONNX export parameters
    input_names = ["input_image"]; output_names = ["output_image"]
    dynamic_axes_config = None
    if args.dynamic_batch:
        dynamic_axes_config = { input_names[0]: {0: 'batch_size'}, output_names[0]: {0: 'batch_size'} }; logging.info("Dynamic batch size enabled for ONNX export.")

    # Export to ONNX
    logging.info(f"Starting ONNX export to: {output_onnx_path}")
    try:
        model.eval() # Ensure eval mode
        torch.onnx.export( model, dummy_input, output_onnx_path, export_params=True, opset_version=args.opset, do_constant_folding=True, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes_config )
        logging.info("ONNX export completed successfully.")

        # Generate JSON Sidecar
        json_base_name = os.path.splitext(os.path.basename(output_onnx_path))[0]
        generate_flame_json( onnx_base_name=json_base_name, resolution=resolution, output_json_path=output_json_path, model_name=args.model_name, model_desc=args.model_desc )
        logging.info("Conversion process finished successfully.")
        return 0
    except torch.onnx.errors.CheckerError as ce: logging.error(f"ONNX CheckerError during export: {ce}\nModel may be invalid.", exc_info=True); return 1
    except Exception as e:
        logging.error(f"ONNX export or JSON generation failed: {e}", exc_info=True)
        # --- Attempt to clean up potentially incomplete ONNX file ---
        if os.path.exists(output_onnx_path):
            try:
                os.remove(output_onnx_path)
                logging.info(f"Removed potentially incomplete ONNX file: {output_onnx_path}")
            except OSError as remove_e: # Catch specific OS errors during removal
                logging.warning(f"Could not remove incomplete ONNX file '{output_onnx_path}': {remove_e}")
        return 1

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PyTorch TuNet checkpoint to ONNX for Flame (with internal normalization). Handles new/old config formats.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the PyTorch checkpoint (.pth)')
    parser.add_argument('--output_onnx', type=str, default=None, help='Path to save the output ONNX model (.onnx). Defaults to same name/dir.')
    parser.add_argument('--output_json', type=str, default=None, help='Path to save the Flame JSON sidecar file (.json). Defaults to same name/dir.')
    parser.add_argument('--output_dir', type=str, default=None, help='Optional directory to save outputs if specific paths not given.')
    parser.add_argument('--opset', type=int, default=14, help='ONNX opset version. Default: 14')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for loading model if available.')
    parser.add_argument('--dynamic_batch', action='store_true', help='Enable dynamic batch size in the exported ONNX model.')
    parser.add_argument('--model_name', type=str, default=None, help='Name for the model in Flame UI (defaults to ONNX filename base).')
    parser.add_argument('--model_desc', type=str, default="TuNet by tpo, converted model.", help='Description for the model in Flame UI.')
    args = parser.parse_args()

    exit_code = convert(args)
    exit(exit_code)

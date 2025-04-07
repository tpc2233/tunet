import os
import argparse
import math
from glob import glob
from PIL import Image
import logging
import time
from types import SimpleNamespace # <-- Add import

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from torch.amp import autocast

# --- Helper to convert nested dict to nested SimpleNamespace ---
# (Copied from train.py)
def dict_to_namespace(d):
    if isinstance(d,dict):
        safe_d={};
        for k,v in d.items(): safe_key=k.replace('-','_'); safe_d[safe_key]=dict_to_namespace(v)
        return SimpleNamespace(**safe_d)
    elif isinstance(d,list): return [dict_to_namespace(item) for item in d]
    else: return d

# --- Corrected UNet Model Definition ---
# !! Must be IDENTICAL to the training script !!
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        mid_channels = max(1, mid_channels)
        self.d = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(True) )
    def forward(self, x): return self.d(x)
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__(); self.m = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.m(x)
class Up(nn.Module):
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
    def __init__(self, in_channels, out_channels): super().__init__(); self.c = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x): return self.c(x)

# <-- CHANGED: UNet class uses model_size_dims -->
class UNet(nn.Module):
    def __init__(self, config, n_ch=3, n_cls=3, bilinear=True):
        super().__init__(); self.n_ch = n_ch; self.n_cls = n_cls;
        # Get model size from the passed config object using the new key
        self.hidden_size = config.model.model_size_dims # <-- Use new key
        self.bilinear = bilinear
        h = self.hidden_size; chs = {'enc1': h, 'enc2': h*2, 'enc3': h*4, 'enc4': h*8, 'bottle': h*16}
        self.inc = DoubleConv(n_ch, chs['enc1']); self.down1 = Down(chs['enc1'], chs['enc2']); self.down2 = Down(chs['enc2'], chs['enc3']); self.down3 = Down(chs['enc3'], chs['enc4']); self.down4 = Down(chs['enc4'], chs['bottle'])
        self.up1 = Up(chs['bottle'], chs['enc4'], chs['enc4'], bilinear); self.up2 = Up(chs['enc4'], chs['enc3'], chs['enc3'], bilinear); self.up3 = Up(chs['enc3'], chs['enc2'], chs['enc2'], bilinear); self.up4 = Up(chs['enc2'], chs['enc1'], chs['enc1'], bilinear)
        self.outc = OutConv(chs['enc1'], n_cls)
    def forward(self, x):
        x1=self.inc(x); x2=self.down1(x1); x3=self.down2(x2); x4=self.down3(x3); x5=self.down4(x4)
        x=self.up1(x5, x4); x=self.up2(x, x3); x=self.up3(x, x2); x=self.up4(x, x1); return self.outc(x)

# --- Helper Functions ---
NORM_MEAN = [0.5, 0.5, 0.5]; NORM_STD = [0.5, 0.5, 0.5]
def denormalize(tensor):
    mean = torch.tensor(NORM_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)

# --- Updated load_model_and_config ---
def load_model_and_config(checkpoint_path, device):
    """Loads TuNet, automatically detecting config from checkpoint.""" # Changed description
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

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
    model_size_saved = default_hidden_size # Start with default
    if is_new_format:
        # Try new key first, then old key within model section
        model_config = getattr(config_source, 'model', SimpleNamespace())
        model_size_saved = getattr(model_config, 'model_size_dims', getattr(model_config, 'unet_hidden_size', default_hidden_size))
        loss_mode = getattr(getattr(config_source, 'training', SimpleNamespace()), 'loss', default_loss)
        resolution = getattr(getattr(config_source, 'data', SimpleNamespace()), 'resolution', default_resolution)
        bilinear_mode = default_bilinear # Use class default
    else: # Old format (args)
        # Try new key first (in case args somehow had it), then old key
        model_size_saved = getattr(config_source, 'model_size_dims', getattr(config_source, 'unet_hidden_size', default_hidden_size))
        loss_mode = getattr(config_source, 'loss', default_loss)
        resolution = getattr(config_source, 'resolution', default_resolution)
        bilinear_mode = getattr(config_source, 'bilinear', default_bilinear)

    logging.info(f"Checkpoint parameters: Saved Model Size={model_size_saved}, Loss Mode='{loss_mode}', Resolution={resolution}, Bilinear={bilinear_mode}")

    # Calculate Effective Hidden Size (using extracted model_size_saved)
    hq_default_bump_size = 96 # Match training script
    effective_model_size = model_size_saved # Use the correctly extracted size
    default_size_for_bump = 64 # Match training script condition
    if loss_mode == 'l1+lpips' and model_size_saved == default_size_for_bump:
        effective_model_size = hq_default_bump_size
        logging.info(f"Applying model size bump logic: Effective Size = {effective_model_size}")
    else:
        logging.info(f"Effective Model Size = {effective_model_size}")

    # Instantiate Model
    # <-- CHANGED: Create minimal config with the NEW key name -->
    minimal_unet_config = SimpleNamespace(
        model=SimpleNamespace(model_size_dims=effective_model_size)
    )
    model = UNet(config=minimal_unet_config, n_ch=3, n_cls=3, bilinear=bilinear_mode)

    # Load State Dict
    state_dict = checkpoint['model_state_dict']
    if all(key.startswith('module.') for key in state_dict):
        logging.info("Removing 'module.' prefix from state_dict keys.")
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logging.info("Model loaded successfully and set to evaluation mode.")

    return model, resolution

# --- create_blend_mask ---
# ... (remains the same) ...
def create_blend_mask(resolution, device):
    hann_1d = torch.hann_window(resolution, periodic=False, device=device)
    hann_2d = hann_1d.unsqueeze(1) * hann_1d.unsqueeze(0)
    return hann_2d.view(1, 1, resolution, resolution)

# --- Main Inference Function ---
# ... (process_image remains the same) ...
def process_image(model, image_path, output_path, resolution, stride, device, batch_size, transform, denormalize_fn, use_amp):
    logging.info(f"Processing: {os.path.basename(image_path)}")
    start_time = time.time()
    try:
        img = Image.open(image_path).convert('RGB')
        img_width, img_height = img.size
        if img_width < resolution or img_height < resolution:
            logging.warning(f"Image smaller than resolution {resolution}. Skipping.")
            return
    except Exception as e:
        logging.error(f"Failed to load: {image_path}: {e}")
        return
    slice_coords = []
    possible_y = list(range(0, img_height - resolution, stride)) + ([img_height - resolution] if img_height > resolution else [0])
    possible_x = list(range(0, img_width - resolution, stride)) + ([img_width - resolution] if img_width > resolution else [0])
    unique_y = sorted(list(set(possible_y)))
    unique_x = sorted(list(set(possible_x)))
    for y in unique_y:
        for x in unique_x:
            slice_coords.append((x, y, x + resolution, y + resolution))
    num_slices = len(slice_coords)
    if num_slices == 0:
        logging.warning(f"No slices generated for {os.path.basename(image_path)}.")
        return
    logging.info(f"Generated {num_slices} slices.")
    output_canvas_cpu = torch.zeros(1, 3, img_height, img_width, dtype=torch.float32)
    weight_map_cpu = torch.zeros(1, 1, img_height, img_width, dtype=torch.float32)
    blend_mask = create_blend_mask(resolution, device)
    blend_mask_cpu = blend_mask.cpu()[0]
    processed_count = 0
    device_type = device.type
    with torch.no_grad():
        for i in range(0, num_slices, batch_size):
            batch_coords = slice_coords[i:min(i + batch_size, num_slices)]
            batch_src_list = [img.crop(coords) for coords in batch_coords]
            batch_src_tensor_list = [transform(src_pil) for src_pil in batch_src_list]
            batch_src_tensor = torch.stack(batch_src_tensor_list).to(device, non_blocking=True)
            with autocast(device_type=device_type, enabled=use_amp):
                batch_output_tensor = model(batch_src_tensor)
            batch_output_tensor_cpu = batch_output_tensor.cpu().float()
            for j, coords in enumerate(batch_coords):
                x, y, _, _ = coords
                output_slice_cpu = denormalize_fn(batch_output_tensor_cpu[j].unsqueeze(0))[0]
                output_canvas_cpu[0, :, y:y + resolution, x:x + resolution] += output_slice_cpu * blend_mask_cpu
                weight_map_cpu[0, :, y:y + resolution, x:x + resolution] += blend_mask_cpu
            processed_count += len(batch_coords)
            if processed_count % (batch_size * 10) == 0 or processed_count == num_slices:
                logging.info(f"  Processed {processed_count}/{num_slices} slices...")
    weight_map_cpu = torch.clamp(weight_map_cpu, min=1e-8)
    final_image_tensor = output_canvas_cpu / weight_map_cpu
    try:
        final_pil_image = to_pil_image(torch.clamp(final_image_tensor[0], 0, 1))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_pil_image.save(output_path)
        end_time = time.time()
        logging.info(f"Saved result: {output_path} ({end_time - start_time:.2f} sec)")
    except Exception as e:
        logging.error(f"Failed to save {output_path}: {e}")


# --- Main Execution ---
# ... (remains the same) ...
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference using trained TuNet with tiled processing.') # Updated description
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing source images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed images')
    parser.add_argument('--overlap_factor', type=float, default=0.5, help='Overlap factor for slices (0.0 to <1.0). Default: 0.5')
    parser.add_argument('--batch_size', type=int, default=1, help='Slice batch size (adjust based on GPU memory)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device for inference')
    parser.add_argument('--use_amp', action='store_true', help='Enable AMP for inference')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()],
        force=True )

    if not os.path.exists(args.checkpoint): logging.error(f"Checkpoint not found: {args.checkpoint}"); exit(1)
    if not os.path.isdir(args.input_dir): logging.error(f"Input dir not found: {args.input_dir}"); exit(1)
    if not (0.0 <= args.overlap_factor < 1.0): logging.error("overlap_factor must be [0.0, 1.0)"); exit(1)
    if args.batch_size <= 0: logging.error("batch_size must be positive."); exit(1)

    selected_device = args.device
    if selected_device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Falling back to CPU.")
        selected_device = 'cpu'
    device = torch.device(selected_device)
    logging.info(f"Using device: {device}")
    if args.use_amp and selected_device == 'cpu':
        logging.warning("AMP disabled on CPU.")
        args.use_amp = False

    try:
        model, resolution = load_model_and_config(args.checkpoint, device)
    except Exception as e:
        logging.error(f"Failed to load model: {e}", exc_info=True)
        exit(1)

    transform = T.Compose([ T.ToTensor(), T.Normalize(mean=NORM_MEAN, std=NORM_STD) ])
    overlap_pixels = int(resolution * args.overlap_factor)
    stride = max(1, resolution - overlap_pixels)
    logging.info(f"Inference Parameters: Resolution={resolution}, Overlap Factor={args.overlap_factor}, Stride={stride}")

    img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.webp']
    input_files = sorted([f for ext in img_extensions for f in glob(os.path.join(args.input_dir, ext))])
    if not input_files:
        logging.error(f"No supported image files found in {args.input_dir}")
        exit(1)
    logging.info(f"Found {len(input_files)} images to process.")
    os.makedirs(args.output_dir, exist_ok=True)

    total_start_time = time.time()
    for i, img_path in enumerate(input_files):
        logging.info(f"--- Processing image {i+1}/{len(input_files)} ---")
        basename = os.path.splitext(os.path.basename(img_path))[0]
        output_filename = f"{basename}_processed.png"
        output_path = os.path.join(args.output_dir, output_filename)
        process_image(model, img_path, output_path, resolution, stride, device,
                      args.batch_size, transform, denormalize, args.use_amp)

    total_end_time = time.time()
    logging.info(f"--- Inference finished ---")
    logging.info(f"Processed {len(input_files)} images.")
    logging.info(f"Total time: {total_end_time - total_start_time:.2f} seconds")
    logging.info(f"Average time per image: {(total_end_time - total_start_time)/len(input_files) if input_files else 0:.2f} seconds")

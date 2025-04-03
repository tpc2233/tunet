import os
import argparse
import math
from glob import glob
from PIL import Image
import logging
import time

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image

# --- UNet Model Definition ---
# !! Important: This definition MUST match the one used during training !!
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

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
        logits = self.outc(x)
        return logits

# --- Helper Functions ---

# !! Important: Match these values to your T.Normalize values used during training !!
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

def denormalize(tensor):
    """Reverses the normalization applied by T.Normalize."""
    mean = torch.tensor(NORM_MEAN).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(NORM_STD).view(1, 3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1) # Clamp to valid image range [0, 1]

def load_model(checkpoint_path, device):
    """Loads the UNet model from a checkpoint."""
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # --- Extract training arguments ---
    if 'args' not in checkpoint:
        raise ValueError("Checkpoint does not contain 'args'. Cannot determine training resolution.")
    train_args = checkpoint['args']

    # --- Determine resolution ---
    if not hasattr(train_args, 'resolution'):
         raise ValueError("Training 'args' in checkpoint missing 'resolution' attribute.")
    resolution = train_args.resolution
    logging.info(f"Model trained with resolution: {resolution}x{resolution}")

    # --- Instantiate the model ---
    # Assuming 3 input channels (RGB) and 3 output channels
    model = UNet(n_channels=3, n_classes=3, bilinear=True) # Ensure bilinear matches training if used

    # --- Load the state dict ---
    state_dict = checkpoint['model_state_dict']
    # Handle potential 'module.' prefix if saved from DDP
    if all(key.startswith('module.') for key in state_dict):
        logging.info("Removing 'module.' prefix from state_dict keys (DDP checkpoint).")
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval() # Set to evaluation mode
    logging.info("Model loaded successfully and set to evaluation mode.")

    return model, resolution, train_args # Return args too, might be useful

def create_blend_mask(resolution, device):
    """Creates a 2D Hann window for blending"""
    # Generates a 1D Hann window and expands it to 2D
    hann_1d = torch.hann_window(resolution, periodic=False, device=device)
    hann_2d = hann_1d.unsqueeze(1) * hann_1d.unsqueeze(0)
    # Reshape to (1, 1, H, W) for broadcasting with image tensors (C=1 as it's applied per channel)
    return hann_2d.view(1, 1, resolution, resolution)

# --- Main Inference Function ---

def process_image(model, image_path, output_path, resolution, stride, device, batch_size, transform, denormalize_fn, use_amp):
    """Processes a single large image using tiled inference."""
    logging.info(f"Processing: {os.path.basename(image_path)}")
    start_time = time.time()

    try:
        img = Image.open(image_path).convert('RGB')
        img_width, img_height = img.size
    except Exception as e:
        logging.error(f"Failed to load image {image_path}: {e}")
        return

    # --- Calculate slice coordinates ---
    slice_coords = []
    possible_y = list(range(0, img_height - resolution, stride)) + [img_height - resolution]
    possible_x = list(range(0, img_width - resolution, stride)) + [img_width - resolution]
    unique_y = sorted(list(set(possible_y)))
    unique_x = sorted(list(set(possible_x)))

    for y in unique_y:
        for x in unique_x:
            slice_coords.append((x, y, x + resolution, y + resolution)) # left, upper, right, lower

    num_slices = len(slice_coords)
    logging.info(f"Generated {num_slices} slices.")

    # --- Prepare output canvas and weight map ---
    output_canvas = torch.zeros(1, 3, img_height, img_width, dtype=torch.float32, device=device)
    weight_map = torch.zeros(1, 1, img_height, img_width, dtype=torch.float32, device=device) # Single channel for weights
    blend_mask = create_blend_mask(resolution, device)

    # --- Process slices in batches ---
    processed_count = 0
    with torch.no_grad(): # Disable gradient calculations
        for i in range(0, num_slices, batch_size):
            batch_coords = slice_coords[i:min(i + batch_size, num_slices)]
            batch_src_slices = []

            # --- Prepare batch ---
            for coords in batch_coords:
                src_slice_pil = img.crop(coords)
                # Apply transforms (ToTensor, Normalize)
                src_tensor = transform(src_slice_pil)
                batch_src_slices.append(src_tensor)

            # Stack tensors and move to device
            batch_src_tensor = torch.stack(batch_src_slices).to(device)

            # --- Inference ---
            with torch.cuda.amp.autocast(enabled=use_amp):
                batch_output_tensor = model(batch_src_tensor)

            # Move results to CPU *before* potential large memory operations like denormalizing or stitching
            batch_output_tensor = batch_output_tensor.cpu().float() # Ensure float for denorm/stitching

            # --- Stitch results back ---
            for j, coords in enumerate(batch_coords):
                x, y, _, _ = coords
                # Denormalize the single slice output
                output_slice = denormalize_fn(batch_output_tensor[j].unsqueeze(0)) # Add batch dim back
                output_slice = output_slice.to(device) # Move back to device for tensor operations

                # Add slice to canvas, weighted by blend mask
                output_canvas[0, :, y:y+resolution, x:x+resolution] += output_slice[0] * blend_mask[0]
                # Add blend mask weights to the weight map
                weight_map[0, :, y:y+resolution, x:x+resolution] += blend_mask[0]

            processed_count += len(batch_coords)
            if processed_count % (batch_size * 5) == 0: # Log progress occasionally
                 logging.info(f"  Processed {processed_count}/{num_slices} slices...")

    # --- Average overlapping regions ---
    # Add a small epsilon to weight_map to prevent division by zero in areas with no overlap (shouldn't happen with this logic but safe)
    weight_map = torch.clamp(weight_map, min=1e-8)
    final_image_tensor = output_canvas / weight_map
    final_image_tensor = final_image_tensor.cpu() # Ensure on CPU before converting to PIL

    # --- Convert to PIL and Save ---
    try:
        # Clamp just in case, then convert (expects CHW)
        final_pil_image = to_pil_image(torch.clamp(final_image_tensor[0], 0, 1))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_pil_image.save(output_path)
        end_time = time.time()
        logging.info(f"Saved result to: {output_path} (took {end_time - start_time:.2f} seconds)")
    except Exception as e:
        logging.error(f"Failed to save image {output_path}: {e}")


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference using trained UNet with tiled processing.')

    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint (.pth file)')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing large source images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the processed images')
    parser.add_argument('--overlap_factor', type=float, default=0.5, help='Overlap factor for slices during inference (0.0 to <1.0). Default: 0.5 (50%%)')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of slices to process in a batch (adjust based on GPU memory)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for inference')
    parser.add_argument('--use_amp', action='store_true', help='Enable Automatic Mixed Precision (AMP) for inference')

    args = parser.parse_args()

    # --- Setup Logging ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.StreamHandler()])

    # --- Validate Args ---
    if not os.path.exists(args.checkpoint):
        logging.error(f"Checkpoint file not found: {args.checkpoint}")
        exit(1)
    if not os.path.isdir(args.input_dir):
        logging.error(f"Input directory not found: {args.input_dir}")
        exit(1)
    if not (0.0 <= args.overlap_factor < 1.0):
        logging.error(f"overlap_factor ({args.overlap_factor}) must be between 0.0 and < 1.0")
        exit(1)
    if args.batch_size <= 0:
        logging.error("batch_size must be positive.")
        exit(1)

    # --- Setup Device ---
    if args.device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    logging.info(f"Using device: {device}")
    if args.use_amp and args.device == 'cpu':
         logging.warning("AMP requested but device is CPU. AMP will not be used.")
         args.use_amp = False


    # --- Load Model ---
    try:
        model, resolution, _ = load_model(args.checkpoint, device) # We get resolution from checkpoint
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        exit(1)

    # --- Define Transforms (must match training) ---
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    # --- Calculate Stride ---
    overlap_pixels = int(resolution * args.overlap_factor)
    stride = max(1, resolution - overlap_pixels)
    logging.info(f"Using inference Resolution={resolution}, Overlap Factor={args.overlap_factor}, Stride={stride}")


    # --- Find Input Images ---
    img_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
    input_files = []
    for ext in img_extensions:
        input_files.extend(glob(os.path.join(args.input_dir, ext)))

    if not input_files:
        logging.error(f"No supported image files found in {args.input_dir}")
        exit(1)

    logging.info(f"Found {len(input_files)} images to process.")

    # --- Create Output Directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Process Each Image ---
    for img_path in input_files:
        basename = os.path.basename(img_path)
        output_path = os.path.join(args.output_dir, basename)
        process_image(
            model=model,
            image_path=img_path,
            output_path=output_path,
            resolution=resolution,
            stride=stride,
            device=device,
            batch_size=args.batch_size,
            transform=transform,
            denormalize_fn=denormalize,
            use_amp=args.use_amp
        )

    logging.info("Inference finished.")

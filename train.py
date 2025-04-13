# Merged train and train_multigpu

import os
import argparse
import math
import glob
#from glob import glob as VERY_UNIQUE_GLOB_ALIAS
from PIL import Image
import logging
import time
import random
import yaml
import copy
from types import SimpleNamespace
import itertools # For infinite dataloader cycle
import signal # For graceful shutdown signal handling
import re # For checkpoint pruning regex
import numpy as np # Added for potential use in augmentations/datasets

import torch
import torch.nn as nn # Import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
# <<< Novos imports needed by create_augmentations >>>
# Example: import albumentations as A
# Example: from torchvision.transforms import InterpolationMode
# <<< -------------------------------------------------- >>>
from torchvision.utils import make_grid
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
import lpips
import albumentations as A
from albumentations.pytorch import ToTensorV2 # Talvez para YAML config

# --- DDP Setup ---
# (Setup functions: AQUI)
def setup_ddp():
    if not dist.is_initialized():
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "29500")
        rank = int(os.environ.get("RANK", os.environ.get("OMPI_COMM_WORLD_RANK", "0")))
        world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE", "1")))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        if "MASTER_ADDR" not in os.environ: os.environ["MASTER_ADDR"] = master_addr
        if "MASTER_PORT" not in os.environ: os.environ["MASTER_PORT"] = master_port
        if "RANK" not in os.environ: os.environ["RANK"] = str(rank)
        if "WORLD_SIZE" not in os.environ: os.environ["WORLD_SIZE"] = str(world_size)

        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        try:
            dist.init_process_group(
                backend=backend,
                init_method=f'tcp://{master_addr}:{master_port}',
                world_size=world_size,
                rank=rank
            )
            torch.cuda.set_device(local_rank)
            if get_rank() == 0:
                logging.info(f"DDP Initialized: Rank {rank}/{world_size} on device cuda:{local_rank} (Backend: {backend})")
        except Exception as e:
            print(f"Error initializing DDP: {e}. Check DDP environment variables (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, LOCAL_RANK) and NCCL/Gloo.")
            raise

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_rank():
    if not dist.is_available() or not dist.is_initialized(): return 0
    return dist.get_rank()

def get_world_size():
    if not dist.is_available() or not dist.is_initialized(): return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

# --- UNet Model & Components ---
# (component classes: DoubleConv, Down, Up, OutConv)
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        mid_channels = max(1, mid_channels)
        self.d = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels), # Converted to SyncBatchNorm if needed
            nn.ReLU(True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), # Converted to SyncBatchNorm if needed
            nn.ReLU(True)
        )
    def forward(self, x): return self.d(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.m = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.m(x)

class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            conv_in_channels = in_channels + skip_channels
        else:
            up_out_channels = in_channels // 2
            self.up = nn.ConvTranspose2d(in_channels, up_out_channels, kernel_size=2, stride=2)
            conv_in_channels = up_out_channels + skip_channels
        self.conv = DoubleConv(conv_in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2); diffX = x2.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x): return self.c(x)

# (UNet class definition)
class UNet(nn.Module):
    def __init__(self, n_ch=3, n_cls=3, hidden_size=64, bilinear=True):
        super().__init__()
        self.n_ch = n_ch; self.n_cls = n_cls; self.hidden_size = hidden_size; self.bilinear = bilinear
        h = hidden_size
        chs = {
            'enc1': max(1, h), 'enc2': max(1, h*2), 'enc3': max(1, h*4),
            'enc4': max(1, h*8), 'bottle': max(1, h*16)
        }
        self.inc = DoubleConv(n_ch, chs['enc1'])
        self.down1 = Down(chs['enc1'], chs['enc2'])
        self.down2 = Down(chs['enc2'], chs['enc3'])
        self.down3 = Down(chs['enc3'], chs['enc4'])
        self.down4 = Down(chs['enc4'], chs['bottle'])
        self.up1 = Up(chs['bottle'], chs['enc4'], chs['enc4'], bilinear)
        self.up2 = Up(chs['enc4'],   chs['enc3'], chs['enc3'], bilinear)
        self.up3 = Up(chs['enc3'],   chs['enc2'], chs['enc2'], bilinear)
        self.up4 = Up(chs['enc2'],   chs['enc1'], chs['enc1'], bilinear)
        self.outc = OutConv(chs['enc1'], n_cls)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# --- Data Loading and Augmentation ---

# <<< --- START: Copied/Adapted from train.py --- >>>

# --- Augmentation Creation Function ---
# IMPORTANT: Make sure this function is identical to the one in your working train.py
# It might require specific imports like `albumentations as A` or others.
# <<< --- Add Albumentations import at the top --- >>>
import albumentations as A
from albumentations.pytorch import ToTensorV2 # Import if you use it in your YAML config

# --- Augmentation Creation Function ---
def create_augmentations(augmentation_list):
    """
    Creates an albumentations Compose object from a list of augmentation configs.
    Each config must have '_target_' key specifying the transform class
    and other keys as parameters. Handles torchvision and albumentations.
    NOTE: Assumes Albumentations is the primary library if mixed.
          Returns A.Compose if any albumentations are used.
    """
    transforms = []
    uses_albumentations = False # Flag to track if we need A.Compose

    if not isinstance(augmentation_list, list):
        logging.warning(f"Augmentation list is not a list, but {type(augmentation_list)}. No augmentations applied.")
        return None

    for aug_config in augmentation_list:
        if not isinstance(aug_config, SimpleNamespace) or not hasattr(aug_config, '_target_'):
            logging.error(f"Invalid augmentation config format: {aug_config}. Missing '_target_'. Skipping.")
            continue

        try:
            target_str = aug_config._target_
            params = {k: v for k, v in vars(aug_config).items() if k != '_target_'}

            target_cls = None
            is_albumentation = False

            # --- Resolve target string ---
            if target_str.startswith('torchvision.transforms.'):
                cls_name = target_str.split('.')[-1]
                target_cls = getattr(T, cls_name, None)
                # Note: Mixing torchvision PIL transforms with Albumentations numpy transforms
                # usually requires careful handling or separate Compose pipelines.
                # This basic version assumes primarily one library is used per pipeline (src/dst/shared).

            # <<< --- Added Albumentations Handling --- >>>
            elif target_str.startswith('albumentations.'):
                # Handle potential nested classes like pytorch.ToTensorV2
                parts = target_str.split('.')
                if len(parts) == 3 and parts[1] == 'pytorch': # e.g., albumentations.pytorch.ToTensorV2
                     cls_name = parts[-1]
                     target_cls = getattr(A.pytorch, cls_name, None)
                elif len(parts) == 2: # e.g., albumentations.HorizontalFlip
                     cls_name = parts[-1]
                     target_cls = getattr(A, cls_name, None)
                else:
                     logging.error(f"Unsupported albumentations target format: {target_str}. Skipping.")
                     continue

                if target_cls:
                    is_albumentation = True
                    uses_albumentations = True # Mark that we need A.Compose
            # <<< --- End Albumentations Handling --- >>>

            else:
                 logging.error(f"Unsupported augmentation target library root: {target_str}. Skipping.")
                 continue

            if target_cls is None:
                logging.error(f"Could not find augmentation class: {target_str}. Skipping.")
                continue

            # --- Instantiate the transform ---
            try:
                 # Albumentations doesn't typically need interpolation conversion like torchvision
                 # Parameters like 'p' (probability) are usually handled directly.
                 transform_instance = target_cls(**params)
                 transforms.append(transform_instance)
                 logging.debug(f"Added augmentation: {target_str} with params {params}")
            except TypeError as e:
                 logging.error(f"TypeError instantiating {target_str} with params {params}: {e}. Skipping.")
            except Exception as e:
                 logging.error(f"Error instantiating {target_str} with params {params}: {e}. Skipping.")

        except Exception as e:
            logging.error(f"Failed to process augmentation config {aug_config}: {e}. Skipping.", exc_info=True)

    if not transforms:
        return None

    # --- Return appropriate Compose object ---
    if uses_albumentations:
        logging.info("Creating Albumentations Compose pipeline.")
        # Note: Albumentations Compose expects numpy arrays typically
        return A.Compose(transforms)
    else:
        # If only torchvision transforms were used
        logging.info("Creating Torchvision Compose pipeline.")
        return T.Compose(transforms)


# --- Augmented Dataset Class ---
# --- Augmented Dataset Class (with glob fix and Albumentations handling) ---
class AugmentedImagePairSlicingDataset(Dataset):
    """
    Dataset class that slices image pairs and applies separate/shared augmentations
    followed by a final transformation (e.g., ToTensor, Normalize).
    Handles Albumentations pipelines.
    """
    def __init__(self, src_dir, dst_dir, resolution, overlap_factor=0.0,
                 src_transforms=None, dst_transforms=None, shared_transforms=None,
                 final_transform=None):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.resolution = resolution
        self.overlap_factor = overlap_factor
        self.src_transforms = src_transforms      # Can be T.Compose or A.Compose or None
        self.dst_transforms = dst_transforms      # Can be T.Compose or A.Compose or None
        self.shared_transforms = shared_transforms # Can be T.Compose or A.Compose or None
        self.final_transform = final_transform   # Should be T.Compose including ToTensor/Normalize

        self.slice_info = []
        self.skipped_count = 0
        self.processed_files = 0
        self.total_slices_generated = 0
        self.skipped_paths = [] # Store tuples of (path, reason)

        if not os.path.isdir(src_dir): raise FileNotFoundError(f"Source directory not found: {src_dir}")
        if not os.path.isdir(dst_dir): raise FileNotFoundError(f"Destination directory not found: {dst_dir}")
        if not (0.0 <= overlap_factor < 1.0): raise ValueError("overlap_factor must be [0.0, 1.0)")

        overlap_pixels = int(resolution * overlap_factor)
        self.stride = max(1, resolution - overlap_pixels)

        # --- correct like this ---
        src_files = sorted(glob.glob(os.path.join(src_dir, '*.*')))
        # ---  ---

        if not src_files: raise FileNotFoundError(f"No source images found in {src_dir}")
        # Use is_main_process() for logging to avoid spam from multiple ranks
        if is_main_process():
            logging.info(f"Found {len(src_files)} potential source files.")
        processed_count_log_interval = max(1, len(src_files) // 10) # Log roughly 10 times

        for i, src_path in enumerate(src_files):
            # Log progress only from the main process
            #if is_main_process() and (i + 1) % processed_count_log_interval == 0:
            #     logging.info(f"  Dataset Init: Processed {i+1}/{len(src_files)} potential pairs...") #Remove por hora

            base_name = os.path.basename(src_path)
            dst_path = os.path.join(dst_dir, base_name)

            if not os.path.exists(dst_path):
                self.skipped_count += 1
                # Only store skip paths on rank 0 to save memory? Or keep for debugging? Let's keep for now.
                self.skipped_paths.append((src_path, "Destination image missing"))
                continue

            try:
                # Get image dimensions without loading full image initially
                with Image.open(src_path) as img_s, Image.open(dst_path) as img_d:
                    w_s, h_s = img_s.size
                    w_d, h_d = img_d.size

                if (w_s, h_s) != (w_d, h_d):
                    self.skipped_count += 1
                    self.skipped_paths.append((src_path, f"Dimension mismatch: Src({w_s}x{h_s}) vs Dst({w_d}x{h_d})"))
                    continue

                if w_s < resolution or h_s < resolution:
                    self.skipped_count += 1
                    self.skipped_paths.append((src_path, f"Image too small ({w_s}x{h_s} vs {resolution}x{resolution})"))
                    continue

                # Calculate slice coordinates using stride
                num_slices_for_file = 0
                y_coords = list(range(0, h_s - resolution, self.stride)) + [h_s - resolution] # Include last possible position
                x_coords = list(range(0, w_s - resolution, self.stride)) + [w_s - resolution] # Include last possible position
                unique_y = sorted(list(set(y_coords)))
                unique_x = sorted(list(set(x_coords)))

                for y in unique_y:
                    for x in unique_x:
                        # Define crop box: (left, upper, right, lower)
                        crop_box = (x, y, x + resolution, y + resolution)
                        self.slice_info.append({'src_path': src_path, 'dst_path': dst_path, 'crop_box': crop_box})
                        num_slices_for_file += 1

                if num_slices_for_file > 0:
                    self.processed_files += 1
                    self.total_slices_generated += num_slices_for_file

            except FileNotFoundError: # Handles case where file disappears between glob and open
                 self.skipped_count += 1
                 self.skipped_paths.append((src_path, "File not found during processing"))
                 continue
            except Exception as e: # Catch other potential errors (PIL issues, etc.)
                self.skipped_count += 1
                self.skipped_paths.append((src_path, f"Error during processing: {e}"))
                # Log error details only from main process to avoid spam
                if is_main_process():
                    logging.warning(f"Skipping {base_name} due to error: {e}", exc_info=False)
                continue

        # Log final counts and skip reasons from main process
        if is_main_process():
            logging.info(f"Dataset initialization finished. Processed: {self.processed_files} pairs. Generated: {self.total_slices_generated} slices. Skipped: {self.skipped_count} pairs.")
            if self.skipped_count > 0:
                 limit = 10 # Log first few reasons
                 logged = 0
                 for path, reason in self.skipped_paths:
                      if logged < limit:
                           logging.warning(f"  - Skip reason: {os.path.basename(path)} -> {reason}")
                           logged += 1
                      else:
                           logging.warning(f"  ... (logged {limit}/{self.skipped_count} skip reasons)")
                           break

        if len(self.slice_info) == 0 and self.processed_files == 0:
             # Make this error more prominent as it stops training
             msg = "CRITICAL: Dataset initialization resulted in 0 usable slices. Check input directories, image formats, dimensions, and skip reasons in logs."
             logging.error(msg)
             # Raise error only on main process to ensure clean exit message?
             # Or let all ranks raise to ensure termination? Let's let all raise.
             raise ValueError(msg)

    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, idx):
        # <<< The __getitem__ method remains the same as the previous version >>>
        # It handles applying Albumentations/Torchvision transforms correctly
        try:
            info = self.slice_info[idx]
            src_path = info['src_path']
            dst_path = info['dst_path']
            crop_box = info['crop_box']

            src_img = Image.open(src_path).convert('RGB')
            dst_img = Image.open(dst_path).convert('RGB')

            src_slice_pil = src_img.crop(crop_box)
            dst_slice_pil = dst_img.crop(crop_box)

            src_slice_np = np.array(src_slice_pil)
            dst_slice_np = np.array(dst_slice_pil)

            if isinstance(self.shared_transforms, A.Compose):
                augmented = self.shared_transforms(image=src_slice_np, mask=dst_slice_np)
                src_slice_np = augmented['image']; dst_slice_np = augmented['mask']
            elif isinstance(self.shared_transforms, T.Compose):
                 seed = random.randint(0, 2**32 - 1); random.seed(seed); torch.manual_seed(seed)
                 src_slice_pil_aug = self.shared_transforms(src_slice_pil)
                 random.seed(seed); torch.manual_seed(seed)
                 dst_slice_pil_aug = self.shared_transforms(dst_slice_pil)
                 src_slice_np = np.array(src_slice_pil_aug); dst_slice_np = np.array(dst_slice_pil_aug)

            if isinstance(self.src_transforms, A.Compose):
                src_slice_np = self.src_transforms(image=src_slice_np)['image']
            elif isinstance(self.src_transforms, T.Compose):
                 src_slice_pil = Image.fromarray(src_slice_np)
                 src_slice_pil = self.src_transforms(src_slice_pil)
                 src_slice_np = np.array(src_slice_pil)

            if isinstance(self.dst_transforms, A.Compose):
                dst_slice_np = self.dst_transforms(image=dst_slice_np)['image']
            elif isinstance(self.dst_transforms, T.Compose):
                 dst_slice_pil = Image.fromarray(dst_slice_np)
                 dst_slice_pil = self.dst_transforms(dst_slice_pil)
                 dst_slice_np = np.array(dst_slice_pil)

            if self.final_transform:
                 src_final_input = Image.fromarray(src_slice_np) # Assume final is T.Compose(ToTensor, Normalize)
                 dst_final_input = Image.fromarray(dst_slice_np)
                 src_tensor = self.final_transform(src_final_input)
                 dst_tensor = self.final_transform(dst_final_input)
            else:
                 src_tensor = T.functional.to_tensor(Image.fromarray(src_slice_np))
                 dst_tensor = T.functional.to_tensor(Image.fromarray(dst_slice_np))

            return src_tensor, dst_tensor
        except Exception as e:
            # Log less verbose error here
            info = self.slice_info[idx]
            logging.error(f"Error processing item idx {idx} (Src: {os.path.basename(info['src_path'])}, Box: {info['crop_box']}): {e}", exc_info=False)
            return None # Return None for collate_fn to handle

# <<< --- END: Copied/Adapted from train.py --- >>>


# --- Helper Functions ---
NORM_MEAN = [0.5, 0.5, 0.5]; NORM_STD = [0.5, 0.5, 0.5]
def denormalize(tensor):
    mean = torch.tensor(NORM_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)

def cycle(iterable):
    while True:
        for x in iterable: yield x

# --- Preview Capture/Saving (Adapted from train.py) ---
def save_previews(model, fixed_src_batch, fixed_dst_batch, output_dir, current_epoch, global_step, device, preview_save_count, preview_refresh_rate):
    # This function is now identical to the one in train.py provided earlier
    # It takes the batch (potentially augmented by capture_preview_batch) and runs inference
    if not is_main_process() or fixed_src_batch is None or fixed_dst_batch is None: return
    num_grid_cols=3;
    if fixed_src_batch.size(0)<num_grid_cols: return # Need enough samples

    # Ensure batches are on CPU before manipulation
    src_select=fixed_src_batch[:num_grid_cols].cpu(); dst_select=fixed_dst_batch[:num_grid_cols].cpu()

    model.eval(); device_type=device.type
    with torch.no_grad(), autocast(device_type=device_type, enabled=torch.is_autocast_enabled()): # Use autocast setting from config
        src_dev=src_select.to(device);
        # Handle DDP model
        model_module=model.module if isinstance(model,DDP) else model;
        predicted_batch=model_module(src_dev)
    model.train() # Return model to train mode

    pred_select=predicted_batch.cpu().float()

    # Denormalize the potentially augmented batches for preview
    src_denorm=denormalize(src_select); pred_denorm=denormalize(pred_select); dst_denorm=denormalize(dst_select)
    combined=[item for i in range(num_grid_cols) for item in [src_denorm[i],dst_denorm[i],pred_denorm[i]]]
    if not combined: return

    grid_tensor=torch.stack(combined); grid=make_grid(grid_tensor,nrow=num_grid_cols,padding=2,normalize=False)
    img_pil=T.functional.to_pil_image(grid); preview_filename=os.path.join(output_dir,"training_preview.jpg")
    try:
        img_pil.save(preview_filename,"JPEG",quality=95)
        log_msg = f"Saved preview to {preview_filename} (Epoch {current_epoch+1}, Step {global_step}, Save #{preview_save_count})"
        # Log refresh status correctly
        refreshed_this_time = preview_refresh_rate > 0 and preview_save_count > 0 and (preview_save_count % preview_refresh_rate == 0)
        # Log only if refreshed or first time
        if refreshed_this_time or preview_save_count == 0:
             logging.info(log_msg + (" - Refreshed Batch" if refreshed_this_time else ""))
    except Exception as e: logging.error(f"Failed to save preview image: {e}")

# --- Preview Capture (Adapted from train.py to handle augmented dataset) ---
def capture_preview_batch(config, src_transforms, dst_transforms, shared_transforms, standard_transform):
    # This function is now identical to the one in train.py provided earlier
    if not is_main_process(): return None, None
    num_preview_samples = 3
    logging.info(f"Capturing/Refreshing fixed batch ({num_preview_samples} samples) for previews (using augmentation settings)...")
    try:
        # Use the Augmented Dataset class and pass the *actual* transform pipelines
        preview_dataset = AugmentedImagePairSlicingDataset(
            config.data.src_dir, config.data.dst_dir, config.data.resolution,
            overlap_factor=config.data.overlap_factor,
            src_transforms=src_transforms,      # Pass pipelines used in training
            dst_transforms=dst_transforms,
            shared_transforms=shared_transforms,
            final_transform=standard_transform  # Pass final ToTensor/Normalize
        )

        if len(preview_dataset) == 0:
             logging.warning("Preview dataset has 0 slices. Cannot capture preview batch.")
             return None, None

        num_samples_to_load = min(num_preview_samples, len(preview_dataset))
        if num_samples_to_load < num_preview_samples:
            logging.warning(f"Preview dataset has only {len(preview_dataset)} slices, capturing {num_samples_to_load}.")

        # Use a temporary dataloader *without* DDP sampler
        # Shuffle=True ensures random samples each time capture is called
        preview_loader = DataLoader(preview_dataset, batch_size=num_samples_to_load, shuffle=True, num_workers=0)
        fixed_src_slices, fixed_dst_slices = next(iter(preview_loader)) # Get one batch

        # Check if the loader actually returned the expected number (or fewer if dataset was small)
        if fixed_src_slices is not None and fixed_src_slices.size(0) > 0:
            logging.info(f"Captured new batch of size {fixed_src_slices.size(0)} for previews (augmentations applied).")
            # Return on CPU
            return fixed_src_slices.cpu(), fixed_dst_slices.cpu()
        else:
             logging.error("Preview DataLoader returned empty batch during capture.")
             return None, None

    except StopIteration:
        logging.error("Preview DataLoader yielded no batches during capture.")
        return None, None
    except Exception as e:
        logging.exception(f"Error capturing preview batch: {e}")
        return None, None


# --- Signal Handler for Graceful Shutdown ---
# (Signal handling: shutdown_requested, handle_signal - remain the same)
shutdown_requested = False
def handle_signal(signum, frame):
    global shutdown_requested
    if not shutdown_requested:
        print(f"\n[Rank {get_rank()}] Received signal {signum}. Requesting graceful shutdown...")
        logging.warning(f"Received signal {signum}. Requesting graceful shutdown...")
        shutdown_requested = True
    else:
        print(f"[Rank {get_rank()}] Shutdown already requested. Terminating forcefully.")
        logging.warning("Shutdown already requested. Terminating forcefully.")
        exit(1)

# --- Custom Collate Function to Skip None ---
# (collate_skip_none remains the same)
def collate_skip_none(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# --- Training Function ---
def train(config):
    global shutdown_requested

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    setup_ddp(); rank = get_rank(); world_size = get_world_size()
    device = torch.device(f"cuda:{rank}"); device_type = device.type

    # --- Logging Setup ---
    # (Logging setup remains the same)
    log_level = logging.INFO if is_main_process() else logging.WARNING
    log_format = f'%(asctime)s [RK{rank}][%(levelname)s] %(message)s'
    handlers = [logging.StreamHandler()]
    if is_main_process():
        os.makedirs(config.data.output_dir, exist_ok=True)
        log_file = os.path.join(config.data.output_dir, 'training.log')
        handlers.append(logging.FileHandler(log_file, mode='a'))
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers, force=True)

    if is_main_process():
        logging.info("="*50)
        logging.info("Starting training run (runs indefinitely until stopped):")
        try: logging.info(f"\n{yaml.dump(config_to_dict(config), indent=2, default_flow_style=False)}")
        except Exception: logging.info(f"Config: {config}")
        logging.info("="*50)
        logging.info(">>> Press Ctrl+C to request graceful shutdown (saves checkpoint). <<<")

    # --- Create Standard Transform (used after augmentations) --- #
    standard_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    # --- Create Augmentations (Adapted from train.py) --- #
    src_transforms = None
    dst_transforms = None
    shared_transforms = None

    if hasattr(config, 'dataloader') and hasattr(config.dataloader, 'datasets'):
        datasets_config = config.dataloader.datasets
        src_augs_list = getattr(datasets_config, 'src_augs', [])
        dst_augs_list = getattr(datasets_config, 'dst_augs', [])
        shared_augs_list = getattr(datasets_config, 'shared_augs', [])

        # Ensure they are lists
        src_augs_list = src_augs_list if isinstance(src_augs_list, list) else []
        dst_augs_list = dst_augs_list if isinstance(dst_augs_list, list) else []
        shared_augs_list = shared_augs_list if isinstance(shared_augs_list, list) else []

        if is_main_process():
             if not any([src_augs_list, dst_augs_list, shared_augs_list]):
                 logging.info("No augmentations specified in config.dataloader.datasets.")
             else:
                 logging.info("Augmentation config found. Creating pipelines...")
                 logging.info(f"  Source Augs: {len(src_augs_list)} steps configured.")
                 logging.info(f"  Dest Augs: {len(dst_augs_list)} steps configured.")
                 logging.info(f"  Shared Augs: {len(shared_augs_list)} steps configured.")

        try:
            src_transforms = create_augmentations(src_augs_list)
            dst_transforms = create_augmentations(dst_augs_list)
            shared_transforms = create_augmentations(shared_augs_list)
            if is_main_process() and any([src_transforms, dst_transforms, shared_transforms]):
                logging.info("Augmentation pipelines created successfully.")
        except Exception as e:
            logging.error(f"FATAL: Failed to create augmentations: {e}. Exiting.", exc_info=True)
            if dist.is_initialized(): cleanup_ddp()
            return # Exit if aug creation fails

    else:
         if is_main_process():
             logging.warning("Config section 'dataloader.datasets' not found. No augmentations will be used.")


    # --- Dataset & DataLoader (Using Augmented Dataset) --- #
    dataset = None
    try:
        # Use the new Augmented Dataset class
        dataset = AugmentedImagePairSlicingDataset(
            config.data.src_dir, config.data.dst_dir, config.data.resolution,
            overlap_factor=config.data.overlap_factor,
            src_transforms=src_transforms,         # Pass created augmentation pipelines
            dst_transforms=dst_transforms,
            shared_transforms=shared_transforms,
            final_transform=standard_transform     # Pass the standard transform
        )
        # Logging now uses properties exposed by AugmentedImagePairSlicingDataset
        if is_main_process():
            logging.info(f"Augmented Dataset: Res={dataset.resolution}, Overlap={dataset.overlap_factor:.2f}, Stride={dataset.stride}")
            if dataset.skipped_count > 0:
                logging.warning(f"Skipped {dataset.skipped_count} pairs during dataset init.")
            if dataset.processed_files > 0:
                avg_slices = dataset.total_slices_generated / dataset.processed_files if dataset.processed_files > 0 else 0
                logging.info(f"Found {dataset.total_slices_generated} usable slices from {dataset.processed_files} valid pairs (avg {avg_slices:.1f}).")
            else:
                 logging.error(f"Dataset processed 0 valid image pairs. Total skipped: {dataset.skipped_count}.")
                 # Log skip reasons if available
                 if dataset.skipped_paths:
                      limit=20; logged=0
                      for path, reason in dataset.skipped_paths:
                           if logged < limit: logging.error(f"  - Skip: {os.path.basename(path)} -> {reason}"); logged+=1
                           else: logging.error(f"  ... (logged {limit}/{len(dataset.skipped_paths)} reasons)"); break
                 raise ValueError("Dataset initialization failed: Processed 0 valid image pairs.")

            if len(dataset) == 0:
                 raise ValueError("Dataset created but contains 0 slices. Check slicing logic or image dimensions.")

            logging.info(f"Total dataset slices: {len(dataset)}")
            est_batches_per_pass = math.ceil(len(dataset) / (config.training.batch_size * world_size))
            logging.info(f"Est. batches per dataset pass (global): {est_batches_per_pass}")
            logging.info(f"Logical Epoch = {config.training.iterations_per_epoch} iterations (for sampler reset).")


    except (FileNotFoundError, ValueError, Exception) as e:
        logging.error(f"Dataset initialization failed: {e}", exc_info=True)
        if dist.is_initialized(): cleanup_ddp();
        return # Exit if dataset fails

    if len(dataset) < world_size:
        logging.error(f"FATAL: Dataset size ({len(dataset)}) is smaller than world size ({world_size}). Cannot use DistributedSampler with drop_last=True.")
        if dist.is_initialized(): cleanup_ddp();
        return

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, sampler=sampler,
                            num_workers=config.dataloader.num_workers, pin_memory=True,
                            prefetch_factor=config.dataloader.prefetch_factor if config.dataloader.num_workers > 0 else None,
                            persistent_workers=True if config.dataloader.num_workers > 0 else False,
                            collate_fn=collate_skip_none) # Use custom collate
    dataloader_iter = cycle(dataloader)

    # --- Model Size Calculation ---
    # (Model size calculation remains the same)
    effective_model_size = config.model.model_size_dims
    use_lpips = False; loss_fn_lpips = None; default_hidden_size = 64; bumped_hidden_size = 96
    if config.training.loss == 'l1+lpips':
        use_lpips = True
        if config.model.model_size_dims == default_hidden_size: effective_model_size = bumped_hidden_size
        if is_main_process(): logging.info(f"LPIPS Enabled. Effective UNet hidden size: {effective_model_size}, Lambda={config.training.lambda_lpips}.")
        # Initialize LPIPS (remains the same, handling broadcast)
        lpips_init_success = [False];
        if is_main_process():
             try: loss_fn_lpips = lpips.LPIPS(net='vgg').to(device).eval(); [p.requires_grad_(False) for p in loss_fn_lpips.parameters()]; lpips_init_success[0]=True; logging.info("LPIPS OK (Rank 0)")
             except Exception as e: logging.error(f"LPIPS Fail (Rank 0): {e}. Disabling.", exc_info=True); use_lpips=False; lpips_init_success[0]=False
        if world_size > 1:
             dist.broadcast_object_list(lpips_init_success, src=0); use_lpips = lpips_init_success[0]
             if rank > 0 and use_lpips:
                  try: loss_fn_lpips = lpips.LPIPS(net='vgg').to(device).eval(); [p.requires_grad_(False) for p in loss_fn_lpips.parameters()]
                  except Exception as e: logging.error(f"LPIPS Fail (Rank {rank}): {e}. May be inconsistent.", exc_info=True)
        if not use_lpips and is_main_process(): logging.warning("LPIPS loss is DISABLED.")
    else:
        if is_main_process(): logging.info(f"L1 loss only: UNet hidden size = {effective_model_size}.")

    # --- Model Instantiation, SyncBN conversion, Resume, DDP Wrap ---
    # (This block containing SyncBN conversion, resume, DDP wrap remains the same as provided in the previous answer)
    # --- Instantiate Model ---
    model = UNet(n_ch=3, n_cls=3, hidden_size=effective_model_size).to(device)
    if is_main_process(): logging.info(f"Base model instantiated with hidden_size={effective_model_size} on device {device}.")
    # --- Convert to SyncBatchNorm if using DDP ---
    if world_size > 1:
        if is_main_process(): logging.info(f"Converting BatchNorm layers to SyncBatchNorm for {world_size} GPUs.")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if is_main_process(): logging.info("SyncBatchNorm conversion complete.")
    else:
         if is_main_process(): logging.info("Running on single GPU (or DDP disabled), standard BatchNorm will be used.")
    # --- Optimizer, Scaler, Loss ---
    optimizer = optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=1e-5)
    scaler = GradScaler(enabled=config.training.use_amp)
    criterion_l1 = nn.L1Loss()
    # --- Resume Logic ---
    start_epoch = 0; start_step = 0
    latest_checkpoint_path = os.path.join(config.data.output_dir, 'tunet_latest.pth')
    resume_flag = [False];
    if is_main_process(): resume_flag[0] = os.path.exists(latest_checkpoint_path)
    if world_size > 1: dist.broadcast_object_list(resume_flag, src=0)
    if resume_flag[0]:
        try:
            if is_main_process(): logging.info(f"Attempting resume from checkpoint: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, map_location='cpu')
            # --- Compatibility Check ---
            saved_config_data = checkpoint.get('config');
            if not saved_config_data: raise ValueError("Checkpoint missing 'config' dictionary.")
            ckpt_config_sns = dict_to_sns(saved_config_data)
            ckpt_loss = getattr(getattr(ckpt_config_sns, 'training', SimpleNamespace()), 'loss', 'l1')
            ckpt_base_model_dims = getattr(getattr(ckpt_config_sns, 'model', SimpleNamespace()), 'model_size_dims', default_hidden_size)
            ckpt_effective_hidden_size = ckpt_base_model_dims
            if ckpt_loss == 'l1+lpips' and ckpt_base_model_dims == default_hidden_size: ckpt_effective_hidden_size = bumped_hidden_size
            current_loss = config.training.loss
            if ckpt_loss != current_loss: raise ValueError(f"Loss mismatch: Checkpoint='{ckpt_loss}', Current='{current_loss}'")
            if ckpt_effective_hidden_size != effective_model_size: raise ValueError(f"Effective model size mismatch: Checkpoint={ckpt_effective_hidden_size}, Current={effective_model_size}")
            if is_main_process(): logging.info("Checkpoint configuration compatible.")
            # --- Load State Dicts ---
            state_dict = checkpoint['model_state_dict']
            if all(key.startswith('module.') for key in state_dict):
                if is_main_process(): logging.info("Removing 'module.' prefix from checkpoint state_dict keys.")
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            if is_main_process(): logging.info("Loaded model state_dict.")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if config.training.use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            else:
                 if config.training.use_amp and ('scaler_state_dict' not in checkpoint or checkpoint['scaler_state_dict'] is None):
                      logging.warning("Resuming with AMP enabled, but no scaler state found in checkpoint. Initializing new scaler state.")
            if is_main_process(): logging.info("Loaded optimizer and scaler state_dict.")
            # --- Load Epoch and Step ---
            start_step = checkpoint.get('global_step', 0)
            start_epoch = start_step // config.training.iterations_per_epoch
            if is_main_process(): logging.info(f"Resuming from Global Step {start_step} (Logical Epoch {start_epoch}).")
            del checkpoint; torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Failed load/resume checkpoint: {e}", exc_info=True)
            logging.warning("Starting training from scratch."); start_epoch = 0; start_step = 0
            optimizer = optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=1e-5) # Reset optimizer
            scaler = GradScaler(enabled=config.training.use_amp) # Reset scaler
        if world_size > 1: dist.barrier()
    else:
        if is_main_process(): logging.info(f"No checkpoint found at {latest_checkpoint_path}. Starting training from scratch.")
        start_epoch = 0; start_step = 0
    # --- Wrap model with DDP ---
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
        if is_main_process(): logging.info("Model wrapped with DDP.")


    # --- Training Loop Variables ---
    global_step = start_step
    iterations_per_epoch = config.training.iterations_per_epoch
    fixed_src_slices = fixed_dst_slices = None; preview_save_count = 0
    epoch_l1_loss_accum = 0.0; epoch_lpips_loss_accum = 0.0; epoch_step_count = 0
    batch_iter_start_time = time.time()

    if is_main_process():
         logging.info(f"Starting training loop (runs indefinitely).")
         logging.info(f"Iterations per Logical Epoch (Sampler Reset): {iterations_per_epoch}")
         logging.info(f"Starting from Global Step: {global_step}, Initial Logical Epoch: {start_epoch}")

    # --- Main Training Loop (Infinite) ---
    model.train()
    try:
        while True:
            if shutdown_requested:
                if is_main_process(): logging.info(f"Shutdown requested at step {global_step}. Exiting training loop.")
                if world_size > 1: dist.barrier()
                break

            current_epoch = global_step // iterations_per_epoch
            is_new_epoch_start = (global_step % iterations_per_epoch == 0)

            if is_new_epoch_start:
                sampler.set_epoch(current_epoch) # Essential for shuffling
                if is_main_process() and (global_step > 0 or start_step > 0): # Avoid logging epoch 0 start unless resuming
                    logging.info(f"--- Starting Logical Epoch {current_epoch + 1} (Step {global_step}) ---")
                epoch_l1_loss_accum = 0.0; epoch_lpips_loss_accum = 0.0; epoch_step_count = 0
                batch_iter_start_time = time.time() # Reset timer at epoch start

            # --- Fetch Batch ---
            data_load_start_time = time.time()
            try:
                batch_data = next(dataloader_iter)
                if batch_data is None:
                    if is_main_process(): logging.warning(f"Step {global_step}: Skipped batch due to data loading error(s).")
                    global_step += 1 # Ensure progress
                    continue
                src_slices, dst_slices = batch_data
            except StopIteration: # Should not happen with cycle()
                logging.error("Infinite dataloader stopped unexpectedly. Re-creating."); dataloader_iter = cycle(dataloader); continue
            except Exception as e:
                if is_main_process(): logging.error(f"Step {global_step}: Unexpected batch loading error: {e}, skipping.", exc_info=True)
                global_step += 1; continue
            data_load_end_time = time.time()

            # --- Process Batch ---
            src_slices = src_slices.to(device, non_blocking=True)
            dst_slices = dst_slices.to(device, non_blocking=True)
            iter_prep_end_time = time.time()

            optimizer.zero_grad(set_to_none=True)
            compute_start_time = time.time()
            with autocast(device_type=device_type, enabled=config.training.use_amp):
                outputs = model(src_slices)
                l1_loss = criterion_l1(outputs, dst_slices)
                lpips_loss = torch.tensor(0.0, device=device)
                if use_lpips and loss_fn_lpips is not None:
                    try: lpips_loss = loss_fn_lpips(outputs, dst_slices).mean()
                    except Exception as e:
                        if is_main_process() and global_step % (config.logging.log_interval * 10) == 1:
                            logging.warning(f"LPIPS calc failed step {global_step}: {e}.", exc_info=False)
                        lpips_loss = torch.tensor(0.0, device=device)
                loss = l1_loss + config.training.lambda_lpips * lpips_loss

            # --- Backprop and Optimize ---
            if torch.isnan(loss) or torch.isinf(loss):
                if is_main_process(): logging.error(f"Step {global_step}: NaN/Inf loss! Skipping step.")
                global_step += 1; batch_iter_start_time = time.time(); continue # Skip update

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            compute_end_time = time.time()

            # --- Loss Aggregation and Logging ---
            batch_l1_loss_reduced = l1_loss.detach().clone()
            batch_lpips_loss_reduced = lpips_loss.detach().clone() if use_lpips else torch.tensor(0.0, device=device)
            if world_size > 1:
                 dist.all_reduce(batch_l1_loss_reduced, op=dist.ReduceOp.AVG)
                 if use_lpips: dist.all_reduce(batch_lpips_loss_reduced, op=dist.ReduceOp.AVG)
            epoch_l1_loss_accum += batch_l1_loss_reduced.item()
            epoch_lpips_loss_accum += batch_lpips_loss_reduced.item()
            epoch_step_count += 1

            global_step += 1
            current_logical_epoch_display = global_step // iterations_per_epoch + 1

            # Logging (only on main process)
            if is_main_process() and global_step % config.logging.log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                avg_epoch_l1 = epoch_l1_loss_accum / epoch_step_count if epoch_step_count > 0 else 0.0
                avg_epoch_lpips = epoch_lpips_loss_accum / epoch_step_count if use_lpips and epoch_step_count > 0 else 0.0
                data_time = data_load_end_time - data_load_start_time
                prep_time = iter_prep_end_time - data_load_end_time
                compute_time = compute_end_time - compute_start_time
                total_step_time = time.time() - batch_iter_start_time
                steps_in_epoch = global_step % iterations_per_epoch if global_step % iterations_per_epoch != 0 else iterations_per_epoch

                log_msg = (f'Epoch[{current_logical_epoch_display}] Step[{global_step}] ({steps_in_epoch}/{iterations_per_epoch}), '
                           f'L1:{batch_l1_loss_reduced.item():.4f}(Avg:{avg_epoch_l1:.4f})')
                if use_lpips: log_msg += (f', LPIPS:{batch_lpips_loss_reduced.item():.4f}(Avg:{avg_epoch_lpips:.4f})')
                log_msg += (f', LR:{current_lr:.1e}, T/Step:{total_step_time:.3f}s (D:{data_time:.3f}, P:{prep_time:.3f} C:{compute_time:.3f})')
                logging.info(log_msg)
                batch_iter_start_time = time.time() # Reset timer for next interval

            # --- Preview Generation (main process, using capture_preview_batch that understands augmentations) ---
            preview_interval = config.logging.preview_batch_interval
            if is_main_process() and preview_interval > 0 and global_step % preview_interval == 0:
                preview_refresh_rate = config.logging.preview_refresh_rate
                refresh_preview = (fixed_src_slices is None) or \
                                  (preview_refresh_rate > 0 and preview_save_count > 0 and \
                                   (preview_save_count % preview_refresh_rate == 0))

                if refresh_preview:
                     # Capture batch using the *training* augmentation pipelines
                     new_src, new_dst = capture_preview_batch(
                         config, src_transforms, dst_transforms, shared_transforms, standard_transform
                     )
                     if new_src is not None and new_dst is not None:
                         fixed_src_slices, fixed_dst_slices = new_src, new_dst # Update batch (on CPU)
                     elif fixed_src_slices is None:
                          logging.warning(f"Step {global_step}: Failed to capture initial preview batch.")
                     else:
                          logging.warning(f"Step {global_step}: Failed to refresh preview batch, using previous one.")

                if fixed_src_slices is not None and fixed_dst_slices is not None:
                    # Pass necessary args to save_previews
                    save_previews(model, fixed_src_slices, fixed_dst_slices, config.data.output_dir,
                                  current_epoch, global_step, device,
                                  preview_save_count, preview_refresh_rate)
                    preview_save_count += 1
                elif preview_save_count == 0: # Log only if we never got a batch
                     logging.warning(f"Step {global_step}: Skipping preview generation (no valid batch available).")


            # --- Periodic Checkpointing & Pruning (main process) ---
            # (Checkpointing logic remains the same - uses model.module.state_dict() correctly)
            save_interval = config.saving.save_iterations_interval
            save_interval_now = (save_interval > 0 and global_step % save_interval == 0)
            save_epoch_end_now = (global_step % iterations_per_epoch == 0) and global_step > 0
            if is_main_process() and (save_interval_now or save_epoch_end_now or shutdown_requested):
                 ckpt_epoch_num = global_step // iterations_per_epoch
                 epoch_checkpoint_path = os.path.join(config.data.output_dir, f'tunet_epoch_{ckpt_epoch_num:09d}.pth') if save_epoch_end_now else None
                 latest_path = os.path.join(config.data.output_dir, 'tunet_latest.pth')
                 config_dict = config_to_dict(config)
                 model_state_to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                 checkpoint_data = { 'epoch': ckpt_epoch_num, 'global_step': global_step, 'model_state_dict': model_state_to_save,
                                     'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict() if config.training.use_amp else None,
                                     'config': config_dict }
                 try:
                     log_save_reason = "interval" if save_interval_now else "epoch end" if save_epoch_end_now else "shutdown"
                     torch.save(checkpoint_data, latest_path)
                     logging.info(f"Saved latest checkpoint ({log_save_reason}): {latest_path} (Step {global_step}, Epoch {ckpt_epoch_num})")
                     if epoch_checkpoint_path:
                          torch.save(checkpoint_data, epoch_checkpoint_path)
                          logging.info(f"Saved epoch checkpoint: {os.path.basename(epoch_checkpoint_path)}")
                          if hasattr(config.saving, 'keep_last_checkpoints') and config.saving.keep_last_checkpoints >= 0:
                              prune_checkpoints(config.data.output_dir, config.saving.keep_last_checkpoints)
                 except Exception as e: logging.error(f"Failed to save checkpoint at step {global_step}: {e}", exc_info=True)

            # --- Barrier ---
            if world_size > 1 and (global_step % iterations_per_epoch == 0):
                 dist.barrier()

    # --- End of Training Loop ---
    except KeyboardInterrupt:
        if is_main_process(): logging.warning("KeyboardInterrupt caught in main loop. Exiting gracefully.")
    except Exception as train_loop_error:
        logging.error("Unexpected error occurred during training loop:", exc_info=True)
        if is_main_process(): logging.info("Attempting to save final state due to error...")
        shutdown_requested = True
    finally:
        # --- Final Save on Shutdown/Error ---
        # (Final save logic remains the same)
        if shutdown_requested and is_main_process():
            logging.info("Performing final checkpoint save on shutdown/error...")
            try:
                 ckpt_epoch_num = global_step // iterations_per_epoch
                 latest_path = os.path.join(config.data.output_dir, 'tunet_latest.pth')
                 config_dict = config_to_dict(config)
                 model_state_to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                 checkpoint_data = { 'epoch': ckpt_epoch_num, 'global_step': global_step, 'model_state_dict': model_state_to_save,
                                     'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict() if config.training.use_amp else None,
                                     'config': config_dict }
                 torch.save(checkpoint_data, latest_path)
                 logging.info(f"Final latest checkpoint saved: {latest_path} (Step {global_step})")
            except Exception as e: logging.error(f"Failed to save final checkpoint: {e}", exc_info=True)
        # --- Cleanup DDP ---
        if is_main_process(): logging.info(f"Training loop finished/terminated at Global Step {global_step}.")
        cleanup_ddp()
        if is_main_process(): logging.info("DDP Cleaned up. Script finished.")


# --- Config Helper Functions ---
# (config_to_dict, dict_to_sns, merge_configs)
def config_to_dict(sns):
    if isinstance(sns, SimpleNamespace): return {k: config_to_dict(v) for k, v in sns.__dict__.items()}
    elif isinstance(sns, (list, tuple)): return [config_to_dict(item) for item in sns]
    else: return sns
def dict_to_sns(d):
    if isinstance(d, dict):
        safe_d = {};
        for key, value in d.items(): safe_key = key.replace('-', '_'); safe_d[safe_key] = dict_to_sns(value)
        return SimpleNamespace(**safe_d)
    elif isinstance(d, (list, tuple)): return type(d)(dict_to_sns(item) for item in d)
    else: return d
def merge_configs(base, user):
    merged = copy.deepcopy(base);
    for key, value in user.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict): merged[key] = merge_configs(merged[key], value)
        else: merged[key] = value
    return merged

# --- Checkpoint Pruning Helper ---
# (prune_checkpoints)
def prune_checkpoints(output_dir, keep_last):
    if keep_last < 0: return
    try:
        ckpt_files_info = []; pattern = os.path.join(output_dir, 'tunet_epoch_*.pth')
        for f_path in glob.glob(pattern):
            basename = os.path.basename(f_path); match = re.match(r"tunet_epoch_(\d+)\.pth", basename)
            if match:
                epoch_num = int(match.group(1))
                try: mtime = os.path.getmtime(f_path); ckpt_files_info.append({'path': f_path, 'epoch': epoch_num, 'mtime': mtime})
                except OSError: logging.warning(f"Could not get mtime for {f_path}, skipping in pruning."); continue
        ckpt_files_info.sort(key=lambda x: (x['epoch'], x['mtime']), reverse=True)
        if len(ckpt_files_info) <= keep_last: return
        files_to_remove = ckpt_files_info[keep_last:]; removed_count = 0
        logging.info(f"Pruning old checkpoints (keeping last {keep_last}). Found {len(ckpt_files_info)} total.")
        for ckpt_info in files_to_remove:
            try: os.remove(ckpt_info['path']); logging.info(f"  Removed: {os.path.basename(ckpt_info['path'])}"); removed_count += 1
            except Exception as e: logging.warning(f"  Failed to remove checkpoint {ckpt_info['path']}: {e}")
        if removed_count > 0: logging.info(f"Pruned {removed_count} old checkpoint(s).")
    except Exception as e: logging.error(f"Error during checkpoint pruning: {e}", exc_info=True)


# --- Main Execution (`if __name__ == "__main__":`) ---
# (Main block remains mostly the same - uses the modified validation from the previous step if you used Option 2,
#  or relies on your updated YAMLs if you used Option 1)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train UNet indefinitely via YAML config using DDP (Aligned Augmentations & SyncBatchNorm)')
    parser.add_argument('--config', type=str, required=True, help='Path to the USER YAML configuration file')
    parser.add_argument('--training.batch_size', type=int, dest='training_batch_size', default=None, help='Override training batch size from config')
    parser.add_argument('--training.lr', type=float, dest='training_lr', default=None, help='Override training learning rate from config')
    cli_args = parser.parse_args()

    if os.environ.get("RANK", "0") == "0":
         logging.basicConfig(level=logging.INFO, format='%(asctime)s [CONFIG LOADER] %(message)s')

    # --- Load Base Config ---
    base_config_path = None; user_config_dir = os.path.dirname(os.path.abspath(cli_args.config)); script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path_options = [ os.path.join(user_config_dir, 'base', 'base.yaml'), os.path.join(script_dir, 'base', 'base.yaml') ]
    for path in base_path_options:
        if os.path.exists(path): base_config_path = path; break
    if not base_config_path: print(f"ERROR: Base configuration ('base/base.yaml') not found"); exit(1)
    try:
        with open(base_config_path, 'r') as f: base_config_dict = yaml.safe_load(f); base_config_dict = base_config_dict or {}
        logging.info(f"Loaded base config: {base_config_path}")
    except Exception as e: print(f"ERROR loading base config {base_config_path}: {e}"); exit(1)
    # --- Load User Config ---
    user_config_path = cli_args.config
    try:
        with open(user_config_path, 'r') as f: user_config_dict = yaml.safe_load(f); user_config_dict = user_config_dict or {}
        logging.info(f"Loaded user config: {user_config_path}")
    except FileNotFoundError: print(f"ERROR: User config file not found: {user_config_path}"); exit(1)
    except Exception as e: print(f"ERROR loading user config {user_config_path}: {e}"); exit(1)
    # --- Merge Configs ---
    merged_config_dict = merge_configs(base_config_dict, user_config_dict); logging.info("Base and User configs merged.")
    # --- Apply CLI Overrides ---
    cli_override_applied = False
    if cli_args.training_batch_size is not None: merged_config_dict.setdefault('training', {})['batch_size'] = cli_args.training_batch_size; logging.info(f"Applied CLI override: training.batch_size = {cli_args.training_batch_size}"); cli_override_applied = True
    if cli_args.training_lr is not None: merged_config_dict.setdefault('training', {})['lr'] = cli_args.training_lr; logging.info(f"Applied CLI override: training.lr = {cli_args.training_lr}"); cli_override_applied = True
    if cli_override_applied: logging.info("CLI overrides applied.")
    # --- Convert final dict to Namespace ---
    config = dict_to_sns(merged_config_dict)
    # --- Display Final Config ---
    if is_main_process():
         print("-" * 20 + " Final Merged Config " + "-" * 20)
         try: print(yaml.dump(config_to_dict(config), indent=2, default_flow_style=False))
         except Exception as dump_error: print(f"Could not dump config cleanly: {dump_error}\n{config}")
         print("-" * 60)

    # --- Configuration Validation & Defaulting ---
    # Using the version from Option 2 that provides defaults if keys are missing
    # --- Configuration Validation & Defaulting/Fallback ---
    missing = []; error_msgs = []
    # Define keys absolutely essential for basic operation
    required_keys = {
        'data': ['src_dir', 'dst_dir', 'output_dir', 'resolution', 'overlap_factor'],
        'model': ['model_size_dims'],
        'training': ['iterations_per_epoch', 'batch_size', 'lr', 'loss', 'lambda_lpips', 'use_amp'],
        # Keep 'keep_last_checkpoints' required as it's usually important
        'saving': ['keep_last_checkpoints']
        # Logging keys will be defaulted if missing
    }
    # --- Check required keys first ---
    for section, keys in required_keys.items():
        if hasattr(config, section):
            section_obj = getattr(config, section);
            if section_obj is None:
                error_msgs.append(f"Required config section '{section}' is null."); missing.append(section); continue
            for k in keys:
                # Check if key exists AND has a value (is not None)
                if not hasattr(section_obj, k) or getattr(section_obj, k, None) is None:
                    error_msgs.append(f"Missing or null required config value: '{section}.{k}'"); missing.append(f"{section}.{k}")
        else:
            error_msgs.append(f"Missing required config section: '{section}'"); missing.append(section)

    # --- Set Defaults OR Use Fallback Locations for Common Keys ---
    # Ensure potentially missing sections exist as namespaces
    if not hasattr(config, 'dataloader'): config.dataloader = SimpleNamespace()
    if not hasattr(config, 'saving'): config.saving = SimpleNamespace()
    if not hasattr(config, 'logging'): config.logging = SimpleNamespace()
    # Ensure training section exists (used for fallbacks) - it should exist based on required_keys, but be safe
    if not hasattr(config, 'training'): config.training = SimpleNamespace()

    # --- num_workers ---
    if not hasattr(config.dataloader, 'num_workers') or config.dataloader.num_workers is None:
        if hasattr(config.training, 'num_workers') and config.training.num_workers is not None:
            logging.info(f"[CONFIG] Using training.num_workers ({config.training.num_workers}) for dataloader.") # Use INFO
            config.dataloader.num_workers = config.training.num_workers
        else:
            logging.info("[CONFIG] Defaulting dataloader.num_workers=4") # Use INFO
            config.dataloader.num_workers = 4

    # --- prefetch_factor ---
    if not hasattr(config.dataloader, 'prefetch_factor') or config.dataloader.prefetch_factor is None:
        if hasattr(config.training, 'prefetch_factor') and config.training.prefetch_factor is not None:
            logging.info(f"[CONFIG] Using training.prefetch_factor ({config.training.prefetch_factor}) for dataloader.") # Use INFO
            config.dataloader.prefetch_factor = config.training.prefetch_factor
        else:
            pf = 2 if config.dataloader.num_workers > 0 else None;
            logging.info(f"[CONFIG] Defaulting dataloader.prefetch_factor={pf}") # Use INFO
            config.dataloader.prefetch_factor = pf

    # --- save_iterations_interval ---
    if not hasattr(config.saving, 'save_iterations_interval') or config.saving.save_iterations_interval is None:
        # <<< CHANGED THIS LINE from logging.warning to logging.info >>>
        logging.info("[CONFIG] Defaulting saving.save_iterations_interval=0 (epoch end only).")
        config.saving.save_iterations_interval = 0


    # --- Default other non-critical keys if needed (logging) ---
    # No logging needed for these defaults
    if not hasattr(config.logging, 'log_interval') or config.logging.log_interval is None: config.logging.log_interval = 50
    if not hasattr(config.logging, 'preview_batch_interval') or config.logging.preview_batch_interval is None: config.logging.preview_batch_interval = 500
    if not hasattr(config.logging, 'preview_refresh_rate') or config.logging.preview_refresh_rate is None: config.logging.preview_refresh_rate = 5


    # --- Validate specific values (now includes defaulted or fallback values) ---
    if not missing: # Only proceed if required sections/keys were present
        try:
             # Training checks
             if config.training.iterations_per_epoch <= 0: error_msgs.append("training.iterations_per_epoch must be > 0")
             if config.training.batch_size <= 0: error_msgs.append("training.batch_size must be > 0")
             if not isinstance(config.training.lr, (int, float)) or config.training.lr <= 0: error_msgs.append("training.lr must be positive number")
             if config.training.loss not in ['l1', 'l1+lpips']: error_msgs.append("training.loss must be 'l1' or 'l1+lpips'")
             if config.training.lambda_lpips < 0: error_msgs.append("training.lambda_lpips must be >= 0")
             # Data checks
             if not (0.0 <= config.data.overlap_factor < 1.0): error_msgs.append("data.overlap_factor must be in [0.0, 1.0)")
             downsample_factor = 16;
             if config.data.resolution <= 0 or config.data.resolution % downsample_factor != 0: error_msgs.append(f"data.resolution must be positive and divisible by {downsample_factor}")
             # Model checks
             if config.model.model_size_dims <= 0: error_msgs.append("model.model_size_dims must be > 0")
             # Dataloader checks (using final values)
             if config.dataloader.num_workers < 0: error_msgs.append("dataloader.num_workers must be >= 0")
             if config.dataloader.num_workers > 0 and (config.dataloader.prefetch_factor is None or config.dataloader.prefetch_factor < 2):
                 error_msgs.append("dataloader.prefetch_factor must be >= 2 when num_workers > 0")
             # Saving checks (using final values)
             if config.saving.save_iterations_interval < 0: error_msgs.append("saving.save_iterations_interval must be >= 0")
             if config.saving.keep_last_checkpoints < -1: # Allow -1 to keep all
                  error_msgs.append("saving.keep_last_checkpoints must be >= -1 (-1 means keep all)")
             # Logging checks (using final values)
             if config.logging.log_interval <= 0: error_msgs.append("logging.log_interval must be > 0")
             if config.logging.preview_batch_interval < 0: error_msgs.append("logging.preview_batch_interval must be >= 0")
             if config.logging.preview_refresh_rate < 0: error_msgs.append("logging.preview_refresh_rate must be >= 0")

        except AttributeError as e:
            error_msgs.append(f"Error accessing expected config value during validation: {e}. Check YAML structure and required keys.")
        except Exception as e:
            error_msgs.append(f"Unexpected error during config value validation: {e}")

    # --- Final Error Check ---
    if error_msgs or missing: # Check both lists
         print("\n" + "="*30 + " CONFIGURATION ERRORS " + "="*30)
         unique_errors = sorted(list(set(error_msgs))) # Combine and unique error messages
         for msg in unique_errors:
             print(f"  - {msg}")
         print("="*82)
         exit(1)
    else:
         # Log final confirmation ONLY if validation passes fully
         logging.info("[CONFIG] Configuration validated successfully.") # Simplified message
         # Log the final config used, including defaults, if desired
         if is_main_process():
             logging.info("--- Final Effective Config ---")
             try: logging.info(f"\n{yaml.dump(config_to_dict(config), indent=2, default_flow_style=False)}")
             except Exception: logging.info(f"Config obj: {config}") # Fallback print
             logging.info("----------------------------")
    # --- Check DDP Environment Variables ---
    essential_ddp_vars = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
    if not all(v in os.environ for v in essential_ddp_vars):
         print("="*40 + "\n WARNING: DDP env vars missing! Forcing single process mode.\n" + "="*40)
         os.environ["RANK"] = "0"; os.environ["WORLD_SIZE"] = "1"; os.environ["LOCAL_RANK"] = "0"
         logging.basicConfig(level=logging.INFO, format='%(asctime)s [FORCED SINGLE] %(message)s', force=True)
    # --- Start Training ---
    train(config)

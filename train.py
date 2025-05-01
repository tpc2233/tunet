# Cross-Platform support

import os
import argparse
import math
import glob
#from glob import glob as VERY_UNIQUE_GLOB_ALIAS # Mantendo por seguranca heh
from PIL import Image, UnidentifiedImageError
import logging
import time
import random
import yaml
import copy
from types import SimpleNamespace
import itertools # For infinite dataloader cycle
import signal # For graceful shutdown signal handling
import re # For checkpoint pruning regex
import numpy as np
import platform # OS detection

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
import lpips
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Detect OS ---
CURRENT_OS = platform.system()

# --- DDP Setup ---
def setup_ddp():
    """Initializes the DDP process group if needed."""
    if not dist.is_available():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        # Use print for essential info if logging isn't fully configured yet
        print(f"INFO: PyTorch distributed not available. Running in single-process mode on {CURRENT_OS}.")
        return

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    if world_size == 1:
        # Only log basic info for single process
        # Use logging.debug for details that shouldn't clutter console
        logging.info(f"Running in single-process mode (World Size = 1).")

        # --- Device Setting for Single Process ---
        # teste MPS (mac) 
        if CURRENT_OS == 'Darwin' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
             # MPS doesn't use local_rank like CUDA for device selection
             logging.debug(f"Single process mode: MPS available.")
        elif torch.cuda.is_available():
            try:
                torch.cuda.set_device(local_rank)
                logging.debug(f"Single process mode: CUDA device set to cuda:{local_rank}")
            except Exception as e:
                 # Use ERROR level for actual problems
                logging.error(f"Error setting CUDA device cuda:{local_rank} in single process mode: {e}. Check CUDA visibility/drivers.")
        else:
             logging.debug(f"Single process mode: No CUDA or MPS available.")
        # --- End Device Setting ---
        return # DDP init not needed

    # --- Initialize DDP for world_size > 1 ---
    if not dist.is_initialized():
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

        # Backend Selection: Prioritize NCCL if CUDA available, else use Gloo (for CPU/MPS DDP)
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        if CURRENT_OS == 'Darwin' and not torch.cuda.is_available():
            # Force Gloo on Mac if no CUDA (necessary for MPS DDP)
            backend = 'gloo'
            logging.debug(f"Forcing Gloo backend for DDP on macOS (No CUDA detected).")

        try:
            init_method = f'tcp://{master_addr}:{master_port}'
            dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)

            # Determine device info for logging (actual device set in train())
            if CURRENT_OS == 'Darwin' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
                 device_info = "mps"
            elif torch.cuda.is_available():
                 # local_rank is relevant for CUDA device selection
                 device_info = f"cuda:{local_rank}"
            else:
                 device_info = "cpu"

            if rank == 0: # Log DDP success only on rank 0
                logging.info(f"DDP Initialized: Rank {rank}/{world_size} using {device_info} target (Backend: {backend})")

        except RuntimeError as e:
            # Attempt Gloo fallback if NCCL fails (especially on non-Linux)
            if 'nccl' in str(e).lower() and backend == 'nccl':
                logging.warning(f"NCCL backend failed (OS: {CURRENT_OS}). Trying Gloo...") # Keep Warning
                backend = 'gloo'
                try:
                    dist.init_process_group(backend=backend, init_method=init_method, world_size=world_size, rank=rank)
                    # Determine device info again after fallback
                    if CURRENT_OS == 'Darwin' and torch.backends.mps.is_available() and torch.backends.mps.is_built(): device_info = "mps"
                    elif torch.cuda.is_available(): device_info = f"cuda:{local_rank}"
                    else: device_info = "cpu"
                    if rank == 0: logging.info(f"DDP Re-Initialized using Gloo backend: Rank {rank}/{world_size} on {device_info}") # Keep INFO
                except Exception as e_gloo:
                    logging.error(f"Gloo backend also failed on Rank {rank}: {e_gloo}", exc_info=True) # Keep Error
                    raise RuntimeError(f"DDP Initialization failed on Rank {rank} with both NCCL and Gloo.") from e_gloo
            else:
                 logging.error(f"DDP initialization failed on Rank {rank} (Backend: {backend}): {e}", exc_info=True) # Keep Error
                 raise RuntimeError(f"DDP Initialization failed on Rank {rank}.") from e
        except Exception as e:
            logging.error(f"Unexpected error during DDP initialization on Rank {rank}: {e}", exc_info=True) # Keep Error
            raise RuntimeError(f"Unexpected DDP Initialization error on Rank {rank}.") from e


# --- DDP Cleanup & Helpers ---
def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def get_rank():
    if not dist.is_available() or not dist.is_initialized(): return 0
    return dist.get_rank()

def get_world_size():
    if not dist.is_available() or not dist.is_initialized(): return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

# --- UNet Model & Components (No changes needed here) ---
class DoubleConv(nn.Module):
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
            up_out_channels = in_channels // 2
            self.up = nn.ConvTranspose2d(in_channels, up_out_channels, kernel_size=2, stride=2)
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
            x1 = nn.functional.pad(x1, [pad_left, pad_right, pad_top, pad_bottom])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c_block = nn.Conv2d(max(1, in_channels), max(1, out_channels), kernel_size=1)
    def forward(self, x): return self.c_block(x)

class UNet(nn.Module):
    def __init__(self, n_ch=3, n_cls=3, hidden_size=64, bilinear=True):
        super().__init__()
        if n_ch <= 0 or n_cls <= 0 or hidden_size <= 0: raise ValueError("n_ch, n_cls, hidden_size must be positive")
        self.n_ch, self.n_cls, self.hidden_size, self.bilinear = n_ch, n_cls, hidden_size, bilinear
        h = hidden_size
        chs = {'enc1': max(1, h), 'enc2': max(1, h*2), 'enc3': max(1, h*4), 'enc4': max(1, h*8), 'bottle': max(1, h*16)}
        bottle_in_ch = chs['bottle']
        self.inc = DoubleConv(n_ch, chs['enc1'])
        self.down1 = Down(chs['enc1'], chs['enc2'])
        self.down2 = Down(chs['enc2'], chs['enc3'])
        self.down3 = Down(chs['enc3'], chs['enc4'])
        self.down4 = Down(chs['enc4'], bottle_in_ch)
        self.up1 = Up(bottle_in_ch, chs['enc4'], chs['enc4'], bilinear)
        self.up2 = Up(chs['enc4'],   chs['enc3'], chs['enc3'], bilinear)
        self.up3 = Up(chs['enc3'],   chs['enc2'], chs['enc2'], bilinear)
        self.up4 = Up(chs['enc2'],   chs['enc1'], chs['enc1'], bilinear)
        self.outc = OutConv(chs['enc1'], n_cls)

    def forward(self, x):
        x1 = self.inc(x); x2 = self.down1(x1); x3 = self.down2(x2); x4 = self.down3(x3); x5 = self.down4(x4)
        x = self.up1(x5, x4); x = self.up2(x, x3); x = self.up3(x, x2); x = self.up4(x, x1)
        return self.outc(x)

# --- Data Loading and Augmentation ---

# --- Augmentation Creation Function ---
def create_augmentations(augmentation_list):
    transforms = []
    uses_albumentations = False
    if not isinstance(augmentation_list, list):
        # Keep warning minimal
        # logging.warning(f"Augmentation list is not a list, using defaults.")
        return None

    for i, aug_config in enumerate(augmentation_list):
        if isinstance(aug_config, SimpleNamespace):
            if not hasattr(aug_config, '_target_'): logging.error(f"Aug {i}: Skip SimpleNamespace missing _target_"); continue
            target_str = aug_config._target_; params = {k: v for k, v in vars(aug_config).items() if k != '_target_'}
        elif isinstance(aug_config, dict):
            if '_target_' not in aug_config: logging.error(f"Aug {i}: Skip dict missing _target_"); continue
            target_str = aug_config['_target_']; params = {k: v for k, v in aug_config.items() if k != '_target_'}
        else: logging.error(f"Aug {i}: Skip invalid type {type(aug_config)}"); continue

        try:
            target_cls, is_torchvision, is_albumentation = None, False, False
            if target_str.startswith('torchvision.transforms.'):
                parts = target_str.split('.');
                if len(parts) == 3: cls_name = parts[-1]; target_cls = getattr(T, cls_name, None); is_torchvision = True
                else: logging.error(f"Aug {i}: Skip bad torchvision format {target_str}"); continue
                # Use DEBUG for mixing warning
                if is_torchvision and uses_albumentations: logging.debug(f"Aug {i}: Mixing Torchvision {cls_name} with Albumentations.")
            elif target_str.startswith('albumentations.'):
                parts = target_str.split('.')
                if len(parts) == 3 and parts[1] == 'pytorch': mod, cls_name = A.pytorch, parts[-1]
                elif len(parts) == 2: mod, cls_name = A, parts[-1]
                else: logging.error(f"Aug {i}: Skip bad albumentations format {target_str}"); continue
                target_cls = getattr(mod, cls_name, None); is_albumentation = True; uses_albumentations = True
            else: logging.error(f"Aug {i}: Skip unsupported lib {target_str}"); continue
            if target_cls is None: logging.error(f"Aug {i}: Class not found {target_str}"); continue

            try:
                transform_instance = target_cls(**params)
                transforms.append(transform_instance)
                # Use DEBUG for successful addition
                logging.debug(f"Added augmentation {i}: {target_str}")
            except Exception as e: logging.error(f"Aug {i}: Error instantiating {target_str}: {e}") # Keep error
        except Exception as e: logging.error(f"Aug {i}: Fail process {target_str}: {e}", exc_info=True) # Keep error for outer try

    if not transforms: return None
    # Use DEBUG for pipeline type confirmation
    if uses_albumentations: logging.debug(f"Creating Albumentations pipeline ({len(transforms)} steps)."); return A.Compose(transforms)
    else: logging.debug(f"Creating Torchvision pipeline ({len(transforms)} steps)."); return T.Compose(transforms)

# --- Augmented Dataset Class ---
class AugmentedImagePairSlicingDataset(Dataset):
    def __init__(self, src_dir, dst_dir, resolution, overlap_factor=0.0,
                 src_transforms=None, dst_transforms=None, shared_transforms=None,
                 final_transform=None):
        self.src_dir = os.path.abspath(src_dir)
        self.dst_dir = os.path.abspath(dst_dir)
        if not os.path.isdir(self.src_dir): raise FileNotFoundError(f"Src dir not found: {self.src_dir}")
        if not os.path.isdir(self.dst_dir): raise FileNotFoundError(f"Dst dir not found: {self.dst_dir}")
        self.resolution, self.overlap_factor = resolution, overlap_factor
        self.src_transforms, self.dst_transforms = src_transforms, dst_transforms
        self.shared_transforms, self.final_transform = shared_transforms, final_transform
        self.slice_info, self.skipped_count, self.processed_files, self.total_slices_generated = [], 0, 0, 0
        self.skipped_file_reasons = []
        overlap_pixels = int(resolution * overlap_factor); self.stride = max(1, resolution - overlap_pixels)
        src_glob_pattern = os.path.join(self.src_dir, '*.*')
        # Use DEBUG for file search pattern
        if is_main_process(): logging.debug(f"Dataset searching: {src_glob_pattern}")
        src_files = sorted(glob.glob(src_glob_pattern))
        if not src_files: logging.error(f"No source files found in '{self.src_dir}'"); # Keep Error
        # Use DEBUG for number of files found
        if is_main_process(): logging.debug(f"Found {len(src_files)} potential source files.")

        for i, src_path in enumerate(src_files):
            base_name = os.path.basename(src_path); dst_path = os.path.join(self.dst_dir, base_name)
            if not os.path.exists(dst_path): self.skipped_count += 1; self.skipped_file_reasons.append((base_name, "Dst missing")); continue
            try:
                with Image.open(src_path) as img_s, Image.open(dst_path) as img_d: w_s, h_s = img_s.size; w_d, h_d = img_d.size
                if (w_s, h_s) != (w_d, h_d): self.skipped_count += 1; self.skipped_file_reasons.append((base_name, f"Dims mismatch")); continue
                if w_s < resolution or h_s < resolution: self.skipped_count += 1; self.skipped_file_reasons.append((base_name, f"Too small")); continue
                num_slices_for_file = 0
                y_coords = list(range(0, max(0, h_s - resolution) + 1, self.stride)); x_coords = list(range(0, max(0, w_s - resolution) + 1, self.stride))
                if (h_s > resolution) and ((h_s - resolution) % self.stride != 0): y_coords.append(h_s - resolution)
                if (w_s > resolution) and ((w_s - resolution) % self.stride != 0): x_coords.append(w_s - resolution)
                unique_y = sorted(list(set(y_coords))); unique_x = sorted(list(set(x_coords)))
                for y in unique_y:
                    if not (0 <= y <= h_s - resolution): continue
                    for x in unique_x:
                        if not (0 <= x <= w_s - resolution): continue
                        crop_box = (x, y, x + resolution, y + resolution)
                        self.slice_info.append({'src_path': src_path, 'dst_path': dst_path, 'crop_box': crop_box}); num_slices_for_file += 1
                if num_slices_for_file > 0: self.processed_files += 1; self.total_slices_generated += num_slices_for_file
            except (FileNotFoundError, UnidentifiedImageError) as file_err:
                 self.skipped_count += 1; self.skipped_file_reasons.append((base_name, type(file_err).__name__));
                 # Keep Warning for bad files
                 if is_main_process(): logging.warning(f"Skipping {base_name}: {file_err}")
            except Exception as e:
                self.skipped_count += 1; self.skipped_file_reasons.append((base_name, f"Error: {type(e).__name__}"))
                # Keep Warning for other errors
                if is_main_process(): logging.warning(f"Skipping {base_name} due to error: {e}", exc_info=False)

        if is_main_process():
            # Keep INFO for overall summary
            logging.info(f"Dataset Init: Processed {self.processed_files}/{len(src_files)} files -> {self.total_slices_generated} slices. Skipped {self.skipped_count} files.")
            # Use DEBUG for skip reasons unless count is high? Maybe keep top N as WARNING.
            if self.skipped_count > 0:
                 limit = 5
                 logging.warning(f"--- Top {min(limit, self.skipped_count)} File Skip Reasons ---") # Keep Warning
                 for i, (name, reason) in enumerate(self.skipped_file_reasons):
                      if i >= limit: logging.warning(f"  ... ({self.skipped_count - limit} more)"); break # Keep Warning
                      logging.warning(f"  - {name}: {reason}") # Keep Warning
                 logging.warning("-----------------------------") # Keep Warning
        if len(self.slice_info) == 0:
             err_msg = f"CRITICAL: Dataset has 0 usable slices. "
             if self.processed_files > 0: err_msg += f"Processed {self.processed_files} files but generated no slices. Check resolution/overlap."
             elif self.skipped_count > 0: err_msg += f"All {self.skipped_count} potential files skipped."
             else: err_msg += f"No source files found/processed in '{self.src_dir}'."
             logging.error(err_msg) # Keep Error
             raise ValueError(err_msg)
        elif is_main_process():
             # Keep INFO for final confirmation
             logging.info(f"Dataset ready. Total usable slices: {len(self.slice_info)}")

    def __len__(self): return len(self.slice_info)
    def __getitem__(self, idx):
        try: info = self.slice_info[idx]; src_path, dst_path, crop_box = info['src_path'], info['dst_path'], info['crop_box']
        except IndexError: logging.error(f"Index {idx} out of bounds for slice_info (len {len(self.slice_info)})."); return None # Keep Error
        try: src_img = Image.open(src_path).convert('RGB'); dst_img = Image.open(dst_path).convert('RGB')
        except Exception as load_e: logging.error(f"Item {idx}: Img load error ({os.path.basename(src_path)}): {load_e}"); return None # Keep Error
        try:
            src_slice_pil = src_img.crop(crop_box); dst_slice_pil = dst_img.crop(crop_box)
            src_img.close(); dst_img.close()
            use_numpy = isinstance(self.shared_transforms, A.Compose) or isinstance(self.src_transforms, A.Compose) or isinstance(self.dst_transforms, A.Compose)
            if use_numpy: src_slice_current, dst_slice_current = np.array(src_slice_pil), np.array(dst_slice_pil)
            else: src_slice_current, dst_slice_current = src_slice_pil, dst_slice_pil

            if self.shared_transforms:
                if isinstance(self.shared_transforms, A.Compose): aug = self.shared_transforms(image=src_slice_current, mask=dst_slice_current); src_slice_current, dst_slice_current = aug['image'], aug['mask']
                elif isinstance(self.shared_transforms, T.Compose): seed = random.randint(0, 2**32-1); torch.manual_seed(seed); random.seed(seed); src_slice_current = self.shared_transforms(src_slice_current); torch.manual_seed(seed); random.seed(seed); dst_slice_current = self.shared_transforms(dst_slice_current)
            if self.src_transforms:
                if isinstance(self.src_transforms, A.Compose): src_slice_current = self.src_transforms(image=src_slice_current)['image']
                elif isinstance(self.src_transforms, T.Compose): src_slice_current = self.src_transforms(src_slice_current)
            if self.dst_transforms:
                if isinstance(self.dst_transforms, A.Compose): dst_slice_current = self.dst_transforms(image=dst_slice_current)['image']
                elif isinstance(self.dst_transforms, T.Compose): dst_slice_current = self.dst_transforms(dst_slice_current)

            if self.final_transform:
                if use_numpy: src_final_input, dst_final_input = Image.fromarray(src_slice_current), Image.fromarray(dst_slice_current)
                else: src_final_input, dst_final_input = src_slice_current, dst_slice_current
                src_tensor, dst_tensor = self.final_transform(src_final_input), self.final_transform(dst_final_input)
            else: # Manual conversion if no final transform
                if not use_numpy: src_slice_current, dst_slice_current = np.array(src_slice_current), np.array(dst_slice_current)
                src_tensor = torch.from_numpy(src_slice_current.transpose(2, 0, 1)).float() / 255.0; dst_tensor = torch.from_numpy(dst_slice_current.transpose(2, 0, 1)).float() / 255.0
                norm = T.Normalize(mean=NORM_MEAN.tolist(), std=NORM_STD.tolist()); src_tensor, dst_tensor = norm(src_tensor), norm(dst_tensor)
            return src_tensor, dst_tensor
        except Exception as e: logging.error(f"Item {idx}: Transform error ({os.path.basename(src_path)}): {e}", exc_info=False); return None # Keep Error

# --- Helper Functions ---
NORM_MEAN = torch.tensor([0.5, 0.5, 0.5]); NORM_STD = torch.tensor([0.5, 0.5, 0.5])
def denormalize(tensor):
    if tensor is None: return None
    try: mean = NORM_MEAN.to(tensor.device).view(1, 3, 1, 1); std = NORM_STD.to(tensor.device).view(1, 3, 1, 1); return torch.clamp(tensor * std + mean, 0, 1)
    except Exception as e: logging.error(f"Denormalize error: {e}"); return tensor
def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try: yield next(iterator)
        except StopIteration: iterator = iter(iterable)

# --- Preview Capture/Saving ---
def save_previews(model, fixed_src_batch, fixed_dst_batch, output_dir, current_epoch, global_step, device, preview_save_count, preview_refresh_rate):
    if not is_main_process() or fixed_src_batch is None or fixed_dst_batch is None or fixed_src_batch.nelement() == 0: return
    num_grid_cols = 3; num_samples = min(fixed_src_batch.size(0), num_grid_cols)
    if num_samples == 0: return
    src_select, dst_select = fixed_src_batch[:num_samples].cpu(), fixed_dst_batch[:num_samples].cpu()
    model.eval(); device_type = device.type
    # Determine AMP usage for inference based on scaler status (safer)
    use_amp_inf = scaler.is_enabled() if 'scaler' in globals() and scaler is not None else False
    pred_select = None
    try:
        # on CUDA use autocast, otherwise just do a plain no_grad
        if device_type == 'cuda' and use_amp_inf:
            ctx = autocast(device_type=device_type, enabled=True)
        else:
            ctx = torch.no_grad()
        with ctx:
            # make sure we’re feeding float32 into the MPS 
            src_dev = src_select.float().to(device)

            src_dev = src_select.to(device)
            
            model_module = model.module if isinstance(model, DDP) else model
            predicted_batch = model_module(src_dev)
        pred_select = predicted_batch.cpu().float()
    except Exception as e: logging.error(f"Preview inference error (Step {global_step}): {e}"); model.train(); return # Keep Error
    model.train()
    src_denorm, pred_denorm, dst_denorm = denormalize(src_select), denormalize(pred_select), denormalize(dst_select)
    if src_denorm is None or pred_denorm is None or dst_denorm is None: logging.error(f"Preview denorm error (Step {global_step})"); return # Keep Error
    combined = [item for i in range(num_samples) for item in [src_denorm[i], dst_denorm[i], pred_denorm[i]]]
    if not combined: return
    try: grid_tensor = torch.stack(combined); grid = make_grid(grid_tensor, nrow=num_grid_cols, padding=2, normalize=False)
    except Exception as e: logging.error(f"Preview grid error (Step {global_step}): {e}"); return # Keep Error
    try: img_pil = T.functional.to_pil_image(grid)
    except Exception as e: logging.error(f"Preview PIL convert error (Step {global_step}): {e}"); return # Keep Error
    preview_filename = os.path.join(output_dir, "training_preview.jpg")
    try:
        os.makedirs(os.path.dirname(preview_filename), exist_ok=True)
        img_pil.save(preview_filename, "JPEG", quality=90)
        log_msg_base = f"Saved preview ({num_samples} samples)"; log_msg_details = f"(E{current_epoch+1}, S{global_step}, #{preview_save_count})"
        refreshed = preview_refresh_rate > 0 and preview_save_count > 0 and (preview_save_count % preview_refresh_rate == 0)
        # Keep INFO log for preview saving, but make it concise
        log_level = logging.INFO if (refreshed or preview_save_count == 0) else logging.DEBUG
        logging.log(log_level, f"{log_msg_base} {log_msg_details}" + (" - Refreshed" if refreshed else ""))
    except Exception as e: logging.error(f"Failed to save preview image: {e}") # Keep Error

def capture_preview_batch(config, src_transforms, dst_transforms, shared_transforms, standard_transform):
    if not is_main_process(): return None, None
    num_preview_samples = 3
    # Use DEBUG for capture attempt message
    logging.debug(f"Capturing/refreshing fixed batch ({num_preview_samples} samples) for previews...")
    fixed_src_slices, fixed_dst_slices = None, None
    try:
        preview_dataset = AugmentedImagePairSlicingDataset(config.data.src_dir, config.data.dst_dir, config.data.resolution, config.data.overlap_factor, src_transforms, dst_transforms, shared_transforms, standard_transform)
        if len(preview_dataset) == 0: logging.warning("Preview dataset has 0 slices."); return None, None # Keep Warning
        num_samples_to_load = min(num_preview_samples, len(preview_dataset))
        if num_samples_to_load < num_preview_samples: logging.info(f"Preview capturing {num_samples_to_load} samples (dataset smaller).") # Keep INFO
        preview_loader = DataLoader(preview_dataset, batch_size=num_samples_to_load, shuffle=True, num_workers=0, collate_fn=collate_skip_none)
        batch_data = next(iter(preview_loader))
        if batch_data is None: logging.error("Preview DataLoader returned None after collation."); return None, None # Keep Error
        fixed_src_slices, fixed_dst_slices = batch_data
        if fixed_src_slices is not None and fixed_dst_slices is not None and fixed_src_slices.size(0) > 0:
            # Use DEBUG for success message
            logging.debug(f"Captured preview batch size {fixed_src_slices.size(0)}.")
            return fixed_src_slices.cpu(), fixed_dst_slices.cpu()
        else: logging.error("Preview DataLoader returned empty/None tensor."); return None, None # Keep Error
    except StopIteration: logging.error("Preview DataLoader StopIteration."); return None, None # Keep Error
    except ValueError as dataset_ve: logging.error(f"Failed creating preview dataset: {dataset_ve}"); return None, None # Keep Error
    except Exception as e: logging.exception(f"Error capturing preview batch: {e}"); return None, None # Keep Error using exception

# --- Signal Handler ---
shutdown_requested = False
def handle_signal(signum, frame):
    global shutdown_requested
    try: sig_name = signal.Signals(signum).name
    except: sig_name = f"Signal {signum}"
    current_rank = get_rank()
    if not shutdown_requested:
        print(f"\n[Rank {current_rank}] Received {sig_name}. Requesting graceful shutdown...") # Keep print
        logging.warning(f"Received {sig_name}. Requesting graceful shutdown...") # Keep Warning
        shutdown_requested = True
    else:
        print(f"\n[Rank {current_rank}] Shutdown already requested. Received {sig_name} again. Terminating forcefully.") # Keep print
        logging.warning("Shutdown already requested. Terminating forcefully.") # Keep Warning
        os._exit(1)

# --- Collate Function ---
def collate_skip_none(batch):
    original_size = len(batch)
    batch = [item for item in batch if item is not None]
    filtered_size = len(batch)
    if filtered_size == 0: return None
    # Use DEBUG for collation skipping info
    if filtered_size < original_size: logging.debug(f"Collate skipped {original_size - filtered_size}/{original_size} items.")
    try: return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e: logging.error(f"Collate error after filtering: {e}"); return None # Keep Error

# --- Training Function ---
def train(config):
    global shutdown_requested, scaler # Make scaler global for use in save_previews
    training_successful = False; final_global_step = 0

    # --- Signal Registration ---
    signal.signal(signal.SIGINT, handle_signal)
    if CURRENT_OS != 'Windows': signal.signal(signal.SIGTERM, handle_signal)

    # --- DDP & Device Setup ---
    try: setup_ddp()
    except Exception as ddp_e: logging.error(f"FATAL: DDP setup failed: {ddp_e}. Exiting.", exc_info=True); return # Keep Error
    rank = get_rank(); world_size = get_world_size()

    # <<< SUPORTE MACOS >>>
    if CURRENT_OS == 'Darwin' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        # No CUDNN benchmark for MPS
    elif torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device(f"cuda:{local_rank}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    device_type = device.type # 'cuda', 'mps', or 'cpu'
    # <<< Talvez usar sem gloo no futuro? >>>


    # --- Logging Setup ---
    log_level = logging.INFO if is_main_process() else logging.WARNING
    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    file_formatter = logging.Formatter(f'%(asctime)s [R{rank}|{CURRENT_OS}|%(levelname)s] %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
    console_handler = logging.StreamHandler(); console_handler.setLevel(log_level); console_handler.setFormatter(console_formatter); root_logger.addHandler(console_handler)
    if is_main_process():
        try:
            output_dir_abs = os.path.abspath(config.data.output_dir); os.makedirs(output_dir_abs, exist_ok=True)
            log_file = os.path.join(output_dir_abs, 'training.log'); file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setLevel(logging.DEBUG); file_handler.setFormatter(file_formatter); root_logger.addHandler(file_handler)
            logging.info("="*60); logging.info(f" Starting Training "); logging.info(f" Device: {device} | World Size: {world_size}")
            logging.info(f" Output Dir: {output_dir_abs}")
            try: logging.debug("--- Effective Configuration ---\n" + yaml.dump(config_to_dict(config), indent=2, default_flow_style=False, sort_keys=False) + "-----------------------------")
            except Exception: logging.debug(f" Config Object: {config}")
            logging.info("="*60); logging.info(">>> Press Ctrl+C for graceful shutdown <<<")
        except Exception as log_setup_e: logging.error(f"Error setting up file logging: {log_setup_e}")

    # --- Standard Transform ---
    standard_transform = T.Compose([ T.ToTensor(), T.Normalize(mean=NORM_MEAN.tolist(), std=NORM_STD.tolist()) ])

    # --- Create Augmentation Pipelines ---
    src_transforms, dst_transforms, shared_transforms = None, None, None
    try:
        datasets_config = getattr(config, 'dataloader', SimpleNamespace()).datasets
        src_list = getattr(datasets_config, 'src_augs', []); dst_list = getattr(datasets_config, 'dst_augs', []); shared_list = getattr(datasets_config, 'shared_augs', [])
        src_transforms = create_augmentations(src_list if isinstance(src_list, list) else [])
        dst_transforms = create_augmentations(dst_list if isinstance(dst_list, list) else [])
        shared_transforms = create_augmentations(shared_list if isinstance(shared_list, list) else [])
        if is_main_process(): logging.debug("Augmentation pipelines created (if configured).")
    except Exception as e: logging.error(f"FATAL: Augmentation creation failed: {e}. Exiting.", exc_info=True); cleanup_ddp(); return

    # --- Dataset & DataLoader ---
    dataset, dataloader, dataloader_iter = None, None, None
    try:
        dataset = AugmentedImagePairSlicingDataset(config.data.src_dir, config.data.dst_dir, config.data.resolution, config.data.overlap_factor, src_transforms, dst_transforms, shared_transforms, standard_transform)
        if is_main_process():
             logging.info(f"Dataset Size: {len(dataset)} slices.")
             global_bs = config.training.batch_size * world_size; est_bpp = math.ceil(len(dataset)/global_bs) if global_bs>0 else 0
             logging.debug(f"Global Batch: {global_bs} | Est Batches/Pass: {est_bpp} | Iter/Epoch: {config.training.iterations_per_epoch}")
        if len(dataset) < world_size: logging.error(f"FATAL: Dataset size ({len(dataset)}) < World size ({world_size})."); cleanup_ddp(); return
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        pin = True if device_type == 'cuda' else False; persist = True if config.dataloader.num_workers > 0 else False
        prefetch = config.dataloader.prefetch_factor if config.dataloader.num_workers > 0 else None
        if CURRENT_OS == 'Windows' and config.dataloader.num_workers > 0 and is_main_process():
             logging.warning("Using num_workers > 0 on Windows. If issues occur, try num_workers=0.")
        dataloader = DataLoader(dataset, batch_size=config.training.batch_size, sampler=sampler, num_workers=config.dataloader.num_workers, pin_memory=pin, prefetch_factor=prefetch, persistent_workers=persist, collate_fn=collate_skip_none, drop_last=True)
        dataloader_iter = cycle(dataloader)
    except Exception as e: logging.error(f"FATAL: Dataset/DataLoader init failed: {e}. Exiting.", exc_info=True); cleanup_ddp(); return

    # --- Model & LPIPS Setup ---
    model, loss_fn_lpips = None, None
    eff_size = config.model.model_size_dims; use_lpips = False; def_h, bump_h = 64, 96
    if config.training.loss == 'l1+lpips':
        use_lpips = True
        if config.model.model_size_dims == def_h: eff_size = bump_h; msg = f"bumping effective hidden size to {eff_size}"
        else: msg = f"using configured hidden size {eff_size}"
        if is_main_process(): logging.debug(f"LPIPS enabled, {msg}. Lambda={config.training.lambda_lpips}")
        status = {'success': False, 'error': None}
        if is_main_process():
            try:
                # <<< LPIPS por device >>>
                if device_type != 'cuda': logging.debug(f"LPIPS running on {device_type}. May be slow.")
                loss_fn_lpips = lpips.LPIPS(net='vgg').to(device).eval(); [p.requires_grad_(False) for p in loss_fn_lpips.parameters()]; status['success'] = True; logging.info("LPIPS model initialized.")
            except Exception as e: logging.error(f"LPIPS init failed: {e}. Disabling.", exc_info=True); status['error'] = str(e); use_lpips = False
        if world_size > 1: dist.broadcast_object_list([status], src=0); status = status[0]; use_lpips = status['success']
        if rank > 0 and use_lpips:
             try: loss_fn_lpips = lpips.LPIPS(net='vgg').to(device).eval(); [p.requires_grad_(False) for p in loss_fn_lpips.parameters()]
             except Exception as e_rank_n: logging.error(f"LPIPS init Rank {rank} failed: {e_rank_n}. Inconsistency likely!", exc_info=True); logging.warning(f"Rank {rank} proceeding with LPIPS potentially uninitialized!")
        if not use_lpips and config.training.loss == 'l1+lpips':
             if is_main_process(): logging.warning(f"LPIPS disabled due to init failure. Using L1 ONLY.")
    else:
        if is_main_process(): logging.debug(f"Using L1 loss only. Hidden size: {eff_size}.")

    # --- Model Instantiation, SyncBN, Resume, DDP Wrap ---
    model, optimizer, scaler = None, None, None
    start_epoch, start_step = 0, 0
    try:
        model = UNet(n_ch=3, n_cls=3, hidden_size=eff_size).to(device)
        if is_main_process(): logging.debug(f"UNet instantiated: hidden_size={eff_size}, device={device}.")
        if world_size > 1 and device_type == 'cuda': # SyncBN only for CUDA DDP
            if is_main_process(): logging.debug("Converting BatchNorm to SyncBatchNorm.")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            dist.barrier()

        optimizer = optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=1e-5)

        # <<< infelizmente Mac sem fp16 >>>
        use_amp_requested = config.training.use_amp
        use_amp_eff = False # Default to False
        if use_amp_requested:
            if device_type == 'cuda':
                use_amp_eff = True
            elif device_type == 'mps':
                if is_main_process(): logging.warning("AMP requested but device is MPS. Disabling AMP (limited support).")
                use_amp_eff = False
            else: # CPU
                if is_main_process(): logging.warning("AMP requested but device is CPU. AMP disabled.")
                use_amp_eff = False


        scaler = GradScaler(enabled=use_amp_eff) # Scaler is created based on effective usage
        if is_main_process(): logging.info(f"Optimizer: AdamW | AMP Enabled: {use_amp_eff}")
        criterion_l1 = nn.L1Loss()
        latest_ckpt = os.path.join(config.data.output_dir, 'tunet_latest.pth')
        exists_list = [False];
        if is_main_process(): exists_list[0] = os.path.exists(latest_ckpt)
        if world_size > 1: dist.broadcast_object_list(exists_list, src=0)
        if exists_list[0]:
            if is_main_process(): logging.info(f"Attempting resume from checkpoint: {latest_ckpt}")
            try:
                ckpt = torch.load(latest_ckpt, map_location='cpu') # Load to CPU first
                if 'config' not in ckpt: raise ValueError("Checkpoint missing 'config'")
                ckpt_cfg = dict_to_sns(ckpt['config']); ckpt_loss = getattr(getattr(ckpt_cfg, 'training', SimpleNamespace()), 'loss', 'l1'); ckpt_base = getattr(getattr(ckpt_cfg, 'model', SimpleNamespace()), 'model_size_dims', def_h)
                ckpt_eff_size = ckpt_base if not (ckpt_loss == 'l1+lpips' and ckpt_base == def_h) else bump_h
                if ckpt_loss != config.training.loss: raise ValueError(f"Loss mismatch: Ckpt='{ckpt_loss}', Current='{config.training.loss}'")
                if ckpt_eff_size != eff_size: raise ValueError(f"Effective size mismatch: Ckpt={ckpt_eff_size}, Current={eff_size}")
                if is_main_process(): logging.debug("Checkpoint config compatible.")
                state_dict = ckpt['model_state_dict']; is_ddp_ckpt = any(k.startswith('module.') for k in state_dict)
                if world_size > 1 and not is_ddp_ckpt:
                    logging.debug("Adding 'module.' prefix to load non-DDP checkpoint into DDP model.")
                    state_dict = {'module.' + k: v for k, v in state_dict.items()}
                elif world_size == 1 and is_ddp_ckpt:
                    logging.debug("Removing 'module.' prefix to load DDP checkpoint into non-DDP model.")
                    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                if is_main_process(): logging.debug("Loaded model state_dict.")

                if 'optimizer_state_dict' in ckpt:
                     optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                     logging.debug("Loaded optimizer state_dict.")
                else:
                    if is_main_process(): logging.warning("Optimizer state missing in checkpoint.")

                # Load scaler state only if AMP is effectively enabled NOW
                if use_amp_eff and ckpt.get('scaler_state_dict'):
                    try: scaler.load_state_dict(ckpt['scaler_state_dict']); logging.debug("Loaded GradScaler state_dict.")
                    except Exception as se: logging.warning(f"Could not load GradScaler state: {se}.")
                elif use_amp_eff: # If AMP is on now, but state was missing
                     if is_main_process(): logging.warning("AMP enabled, but scaler state missing in ckpt.")

                start_step = ckpt.get('global_step', 0) + 1; start_epoch = start_step // config.training.iterations_per_epoch
                if is_main_process(): logging.info(f"Resuming from Global Step {start_step} (Logical Epoch {start_epoch + 1}).")
                del ckpt; torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception as e:
                logging.error(f"Checkpoint load failed: {e}. Starting fresh.", exc_info=True)
                start_epoch, start_step = 0, 0; optimizer = optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=1e-5); scaler = GradScaler(enabled=use_amp_eff)
        else:
            if is_main_process(): logging.info("No checkpoint found. Starting fresh.")


        if world_size > 1:
            # Pass device_ids only for CUDA backend
            cuda_device_ids = [local_rank] if device_type == 'cuda' else None
            model = DDP(model,
                        device_ids=cuda_device_ids,
                        output_device=(local_rank if device_type == 'cuda' else None),
                        find_unused_parameters=False)


        if world_size > 1 and is_main_process(): logging.debug("Model wrapped with DDP.")
        if world_size > 1: dist.barrier()
    except Exception as model_err: logging.error(f"FATAL: Model setup/resume error: {model_err}. Exiting.", exc_info=True); cleanup_ddp(); return

    # --- Training Loop Variables ---
    global_step = start_step; iter_epoch = config.training.iterations_per_epoch
    fixed_src, fixed_dst = None, None; preview_count = 0
    ep_l1, ep_lpips, ep_steps = 0.0, 0.0, 0
    batch_times = []

    # --- Capture Initial Preview Batch ---
    preview_interval = config.logging.preview_batch_interval
    if is_main_process() and preview_interval > 0:
        fixed_src, fixed_dst = capture_preview_batch(config, src_transforms, dst_transforms, shared_transforms, standard_transform)
        if fixed_src is None: logging.warning("Initial preview batch capture failed.")

    # --- Main Training Loop ---
    model.train()
    try:
        while True:
            if shutdown_requested:
                if is_main_process(): logging.info(f"Shutdown requested @ step {global_step}.")
                break

            loop_iter_start = time.time()
            current_ep_idx = global_step // iter_epoch; is_new_epoch = (global_step % iter_epoch == 0)
            if is_new_epoch:
                if isinstance(dataloader.sampler, DistributedSampler): dataloader.sampler.set_epoch(current_ep_idx)
                if is_main_process() and (global_step > 0 or start_step > 0): logging.info(f"--- Epoch {current_ep_idx + 1} Start (Step {global_step}) ---")
                ep_l1, ep_lpips, ep_steps = 0.0, 0.0, 0

            data_load_start = time.time()
            try: batch = next(dataloader_iter)
            except Exception as e: logging.error(f"S{global_step}: Batch load error: {e}, skipping.", exc_info=True); global_step += 1; continue
            if batch is None: logging.warning(f"S{global_step}: Skipped batch (collation error)."); global_step += 1; continue
            #src, dst = batch; data_load_time = time.time() - data_load_start
            src, dst = batch
            data_load_time = time.time() - data_load_start

            # ── send inputs to the same device as the model ──
            transfer_start = time.time()
            src = src.to(device)
            dst = dst.to(device)
            transfer_time = time.time() - transfer_start

            # --- Forward, Loss, Backward, Optimize ---
            compute_start = time.time()
            optimizer.zero_grad(set_to_none=True)

            # <<< Eu sei... >>>
            # Only use autocast context if AMP is effectively enabled (i.e., on CUDA)
            if use_amp_eff:
                with autocast(device_type=device_type, enabled=True):
                    out = model(src)
                    l1 = criterion_l1(out, dst)
                    lp = torch.tensor(0.0, device=device)
                    if use_lpips and loss_fn_lpips:
                        try: lp = loss_fn_lpips(out, dst).mean()
                        except Exception as e_lp: lp = torch.tensor(0.0, device=device)
                    # Loss calculation inside autocast context
                    loss = l1 + config.training.lambda_lpips * lp
            else:
                # Run standard forward pass if AMP is disabled (CPU or MPS)
                out = model(src)
                l1 = criterion_l1(out, dst)
                lp = torch.tensor(0.0, device=device)
                if use_lpips and loss_fn_lpips:
                    try: lp = loss_fn_lpips(out, dst).mean()
                    except Exception as e_lp: lp = torch.tensor(0.0, device=device)
                # Loss calculation outside autocast context
                loss = l1 + config.training.lambda_lpips * lp
            # <<< MODIFIED AUTOCAST USAGE END >>>

            if not torch.isfinite(loss):
                logging.error(f"S{global_step}: NaN/Inf loss ({loss.item()})! Skip update.")
                global_step += 1
                continue # Keep Error

            # Scaler step only if effectively enabled
            if use_amp_eff:
                 scaler.scale(loss).backward()
                 scaler.step(optimizer)
                 scaler.update()
            else:
                 loss.backward() # Standard backward if AMP is off
                 optimizer.step() # Standard optimizer step

            compute_time = time.time() - compute_start


            compute_time = time.time() - compute_start

            batch_l1 = l1.detach().item(); batch_lp = lp.detach().item() if use_lpips else 0.0
            if world_size > 1:
                l1_t = torch.tensor(batch_l1, device=device); lp_t = torch.tensor(batch_lp, device=device)
                dist.all_reduce(l1_t, op=dist.ReduceOp.AVG); dist.all_reduce(lp_t, op=dist.ReduceOp.AVG)
                batch_l1 = l1_t.item(); batch_lp = lp_t.item()
            ep_l1 += batch_l1; ep_lpips += batch_lp; ep_steps += 1

            iter_time = time.time() - loop_iter_start; batch_times.append(iter_time); batch_times = batch_times[-100:]
            avg_time = sum(batch_times) / len(batch_times) if batch_times else 0.0

            global_step += 1; final_global_step = global_step

            if is_main_process() and global_step % config.logging.log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f'… (D:{data_load_time:.3f} T:{transfer_time:.3f} C:{compute_time:.3f})')
                avg_ep_l1 = ep_l1 / ep_steps if ep_steps > 0 else 0.0
                avg_ep_lp = ep_lpips / ep_steps if use_lpips and ep_steps > 0 else 0.0
                steps_in_ep = global_step % iter_epoch or iter_epoch
                log_msg = (f'Epoch[{current_ep_idx + 1}] Step[{global_step}] ({steps_in_ep}/{iter_epoch}), '
                           f'L1:{batch_l1:.4f}(Avg:{avg_ep_l1:.4f})')
                if use_lpips: log_msg += (f', LPIPS:{batch_lp:.4f}(Avg:{avg_ep_lp:.4f})')
                log_msg += (f', LR:{current_lr:.1e}, T/Step:{avg_time:.3f}s '
                            f'(D:{data_load_time:.3f} T:{transfer_time:.3f} C:{compute_time:.3f})')
                logging.info(log_msg)

            if is_main_process() and preview_interval > 0 and global_step % preview_interval == 0:
                refresh = (fixed_src is None) or (config.logging.preview_refresh_rate > 0 and preview_count > 0 and (preview_count % config.logging.preview_refresh_rate == 0))
                if refresh:
                     new_src, new_dst = capture_preview_batch(config, src_transforms, dst_transforms, shared_transforms, standard_transform)
                     if new_src is not None: fixed_src, fixed_dst = new_src, new_dst
                     elif fixed_src is None: logging.warning(f"S{global_step}: Preview refresh failed (no initial).")
                     else: logging.warning(f"S{global_step}: Preview refresh failed, using old.")
                if fixed_src is not None: save_previews(model, fixed_src, fixed_dst, config.data.output_dir, current_ep_idx, global_step, device, preview_count, config.logging.preview_refresh_rate); preview_count += 1
                elif preview_count == 0: logging.warning(f"S{global_step}: Skipping preview (no batch).")

            save_interval = config.saving.save_iterations_interval; save_now = (save_interval > 0 and global_step % save_interval == 0)
            epoch_end_now = (global_step % iter_epoch == 0) and global_step > 0
            if is_main_process() and (save_now or epoch_end_now):
                 ckpt_ep = global_step // iter_epoch; ep_ckpt_path = os.path.join(config.data.output_dir, f'tunet_epoch_{ckpt_ep:09d}.pth') if epoch_end_now else None
                 latest_path = os.path.join(config.data.output_dir, 'tunet_latest.pth'); cfg_dict = config_to_dict(config)
                 m_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                 # Save scaler state only if AMP was effectively used
                 scaler_state = scaler.state_dict() if use_amp_eff else None
                 ckpt_data = {'epoch': ckpt_ep, 'global_step': global_step, 'model_state_dict': m_state, 'optimizer_state_dict': optimizer.state_dict(),
                              'scaler_state_dict': scaler_state, 'config': cfg_dict, 'effective_model_size': eff_size}
                 try:
                     reason = "interval" if save_now else "epoch end"
                     torch.save(ckpt_data, latest_path)
                     logging.info(f"Saved latest checkpoint ({reason}) @ Step {global_step}")
                     if ep_ckpt_path:
                          torch.save(ckpt_data, ep_ckpt_path)
                          logging.info(f"Saved epoch checkpoint: {os.path.basename(ep_ckpt_path)}")
                          if hasattr(config.saving, 'keep_last_checkpoints') and config.saving.keep_last_checkpoints >= 0:
                              prune_checkpoints(config.data.output_dir, config.saving.keep_last_checkpoints)
                 except Exception as e: logging.error(f"Checkpoint save failed @ Step {global_step}: {e}", exc_info=True)
        # --- End of Training Loop ---
        training_successful = True
    except KeyboardInterrupt:
        if is_main_process(): logging.warning("KeyboardInterrupt. Shutting down...")
        if not shutdown_requested: handle_signal(signal.SIGINT, None)
    except Exception as loop_err:
        logging.error("FATAL Training loop error:", exc_info=True);
        shutdown_requested = True
    finally:
        if shutdown_requested and is_main_process():
            logging.info(f"Performing final checkpoint save (State @ Step {final_global_step})...")
            try:
                 final_ep = final_global_step // iter_epoch; latest_path = os.path.join(config.data.output_dir, 'tunet_latest.pth'); cfg_dict = config_to_dict(config)
                 if model and optimizer and scaler:
                     m_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                     scaler_state = scaler.state_dict() if use_amp_eff else None # Check effective use
                     final_data = {'epoch': final_ep, 'global_step': final_global_step, 'model_state_dict': m_state, 'optimizer_state_dict': optimizer.state_dict(),
                                   'scaler_state_dict': scaler_state, 'config': cfg_dict, 'effective_model_size': eff_size}
                     torch.save(final_data, latest_path); logging.info(f"Final checkpoint saved: {latest_path}")
                 else: logging.error("Cannot save final checkpoint: Model/Optimizer/Scaler missing.")
            except Exception as e: logging.error(f"Final checkpoint save failed: {e}", exc_info=True)

        if is_main_process():
            status = "successfully" if training_successful and not shutdown_requested else "gracefully" if shutdown_requested else "with errors"
            logging.info(f"Training finished {status} at Step {final_global_step}.")
        cleanup_ddp();
        if is_main_process(): logging.info("Script finished.")


# --- Config Helper Functions ---
def config_to_dict(sns):
    if isinstance(sns, SimpleNamespace): return {k: config_to_dict(v) for k, v in sns.__dict__.items()}
    elif isinstance(sns, (list, tuple)): return type(sns)(config_to_dict(item) for item in sns)
    return sns
def dict_to_sns(d):
    if isinstance(d, dict):
        safe_d = {}
        for key, value in d.items():
            safe_key = str(key).replace('-', '_')
            if not safe_key.isidentifier(): logging.warning(f"Config key '{key}'->'{safe_key}' invalid identifier.")
            safe_d[safe_key] = dict_to_sns(value)
        return SimpleNamespace(**safe_d)
    elif isinstance(d, (list, tuple)): return type(d)(dict_to_sns(item) for item in d)
    return d
def merge_configs(base, user):
    merged = copy.deepcopy(base)
    for key, value in user.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict): merged[key] = merge_configs(merged[key], value)
        else: merged[key] = value
    return merged

# --- Checkpoint Pruning Helper ---
# Checar macos deleta cache prune_checkpoints
def prune_checkpoints(output_dir, keep_last):
    if keep_last < 0: logging.debug(f"Pruning disabled (keep={keep_last})."); return
    if keep_last == 0: logging.info("Pruning all epoch checkpoints (keep=0).") # Keep INFO for this case
    try:
        ckpt_files_info = []; pattern = os.path.join(output_dir, 'tunet_epoch_*.pth'); epoch_files = glob.glob(pattern)
        for f_path in epoch_files:
            basename = os.path.basename(f_path); match = re.match(r"tunet_epoch_(\d+)\.pth", basename)
            if match:
                epoch_num = int(match.group(1))
                try: mtime = os.path.getmtime(f_path); ckpt_files_info.append({'path': f_path, 'epoch': epoch_num, 'mtime': mtime})
                except OSError as e: logging.warning(f"Pruning: skip {basename}, cannot get mtime: {e}") # Keep Warning
        ckpt_files_info.sort(key=lambda x: (x['epoch'], x['mtime']), reverse=True)
        if len(ckpt_files_info) <= keep_last: logging.debug(f"Pruning: Found {len(ckpt_files_info)}, keeping {keep_last}. None removed."); return
        files_to_remove = ckpt_files_info[keep_last:]; removed_count = 0
        # Keep INFO for pruning start message
        logging.info(f"Pruning: Keeping last {keep_last} checkpoints. Removing {len(files_to_remove)} of {len(ckpt_files_info)} total.")
        for ckpt_info in files_to_remove:
            try:
                os.remove(ckpt_info['path'])
                # Use DEBUG for individual file removal logs
                logging.debug(f"  Removed: {os.path.basename(ckpt_info['path'])}")
                removed_count += 1
            except Exception as e: logging.warning(f"  Failed remove {os.path.basename(ckpt_info['path'])}: {e}") # Keep Warning
        # Keep INFO for pruning summary
        if removed_count > 0: logging.info(f"Pruning finished. Removed {removed_count} checkpoint(s).")
    except Exception as e: logging.error(f"Checkpoint pruning error in '{output_dir}': {e}", exc_info=True) # Keep Error


# --- Main Execution (`if __name__ == "__main__":`) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train UNet via YAML config using DDP (Cross-Platform)')
    parser.add_argument('--config', type=str, required=True, help='Path to the USER YAML configuration file')
    parser.add_argument('--training.batch_size', type=int, dest='training_batch_size', default=None, help='Override training batch_size')
    parser.add_argument('--training.lr', type=float, dest='training_lr', default=None, help='Override training learning_rate')
    parser.add_argument('--data.output_dir', type=str, dest='data_output_dir', default=None, help='Override data output_dir')
    cli_args = parser.parse_args()

    # --- Initial Config Loading Logging (Simple Console) ---
    initial_log_level = logging.INFO
    initial_log_format = '%(asctime)s [CONFIG] %(message)s' # Simple format for config phase
    logging.basicConfig(level=initial_log_level, format=initial_log_format, handlers=[logging.StreamHandler()], force=True) # Use force=True to override potential previous basicConfigs
    logging.info("Script starting...")
    logging.info(f"User config: {cli_args.config}")

    # --- Load Base Config ---
    base_config_dict = {}
    try:
        user_cfg_abs = os.path.abspath(cli_args.config); user_cfg_dir = os.path.dirname(user_cfg_abs); script_dir = os.path.dirname(os.path.abspath(__file__))
        base_opts = [os.path.join(user_cfg_dir, 'base', 'base.yaml'), os.path.join(script_dir, 'base', 'base.yaml')]
        base_path = next((p for p in base_opts if os.path.exists(p)), None)
        if not base_path: raise FileNotFoundError(f"Base config ('base/base.yaml') not found in {base_opts}")
        with open(base_path, 'r') as f: base_config_dict = yaml.safe_load(f) or {}
        logging.info(f"Loaded base config: {base_path}") # Keep INFO
    except Exception as e: logging.error(f"CRITICAL: Failed loading base config: {e}. Exiting.", exc_info=True); exit(1) # Keep Error

    # --- Load User Config ---
    user_config_dict = {}
    try:
        user_cfg_abs = os.path.abspath(cli_args.config)
        if not os.path.exists(user_cfg_abs): raise FileNotFoundError(f"User config not found: {user_cfg_abs}")
        with open(user_cfg_abs, 'r') as f: user_config_dict = yaml.safe_load(f) or {}
        logging.info(f"Loaded user config: {user_cfg_abs}") # Keep INFO
    except Exception as e: logging.error(f"CRITICAL: Failed loading user config: {e}. Exiting.", exc_info=True); exit(1) # Keep Error

    # --- Merge & Overrides ---
    try: merged_dict = merge_configs(base_config_dict, user_config_dict)
    except Exception as e: logging.error(f"CRITICAL: Config merge failed: {e}. Exiting.", exc_info=True); exit(1) # Keep Error
    overrides = {}
    if cli_args.training_batch_size is not None: overrides.setdefault('training', {})['batch_size'] = cli_args.training_batch_size
    if cli_args.training_lr is not None: overrides.setdefault('training', {})['lr'] = cli_args.training_lr
    if cli_args.data_output_dir is not None: overrides.setdefault('data', {})['output_dir'] = os.path.abspath(cli_args.data_output_dir)
    if overrides: merged_dict = merge_configs(merged_dict, overrides); logging.info(f"Applied CLI overrides: {overrides}") # Keep INFO

    # --- Convert to Namespace & Validate ---
    try: config = dict_to_sns(merged_dict)
    except Exception as e: logging.error(f"CRITICAL: Config conversion failed: {e}. Exiting.", exc_info=True); exit(1) # Keep Error

    # --- Configuration Validation (Keep concise, errors are critical) ---
    missing, error_msgs = [], []
    req_keys = {'data': ['src_dir', 'dst_dir', 'output_dir', 'resolution', 'overlap_factor'], 'model': ['model_size_dims'], 'training': ['iterations_per_epoch', 'batch_size', 'lr', 'loss', 'lambda_lpips', 'use_amp'], 'saving': ['keep_last_checkpoints']}
    # Check required sections and keys
    for sec, keys in req_keys.items():
        sec_obj = getattr(config, sec, None)
        if sec_obj is None:
            error_msgs.append(f"Missing required section: '{sec}'")
            missing.append(sec)
            continue # Skip keys check if section is missing
        # If section exists, check its keys
        for k in keys:
            if getattr(sec_obj, k, None) is None:
                error_msgs.append(f"Missing/null required value: '{sec}.{k}'")
                missing.append(f"{sec}.{k}")


    # Defaulting (use simple getattr with default, avoid extra logs here)
    config.dataloader = getattr(config, 'dataloader', SimpleNamespace())
    def_work = 0 if CURRENT_OS == 'Windows' else 2; config.dataloader.num_workers = getattr(config.dataloader, 'num_workers', def_work)
    config.dataloader.prefetch_factor = getattr(config.dataloader, 'prefetch_factor', 2 if config.dataloader.num_workers > 0 else None)
    config.saving = getattr(config, 'saving', SimpleNamespace()); config.saving.save_iterations_interval = getattr(config.saving, 'save_iterations_interval', 0)
    config.logging = getattr(config, 'logging', SimpleNamespace()); config.logging.log_interval = getattr(config.logging, 'log_interval', 50)
    config.logging.preview_batch_interval = getattr(config.logging, 'preview_batch_interval', 500); config.logging.preview_refresh_rate = getattr(config.logging, 'preview_refresh_rate', 5)
    # Final Value Checks
    if not missing:
        try: # Combine checks for brevity
            if config.training.iterations_per_epoch <= 0: error_msgs.append("train.iter_epoch<=0")
            if config.training.batch_size <= 0: error_msgs.append("train.batch_size<=0")
            if not isinstance(config.training.lr, (int, float)) or config.training.lr <= 0: error_msgs.append("train.lr<=0")
            if config.training.loss not in ['l1', 'l1+lpips']: error_msgs.append("train.loss invalid")
            if config.training.lambda_lpips < 0: error_msgs.append("train.lambda_lpips<0")
            if not (0.0 <= config.data.overlap_factor < 1.0): error_msgs.append("data.overlap invalid")
            ds=16;
            if config.data.resolution<=0 or config.data.resolution % ds !=0: error_msgs.append(f"data.resolution not >0 & div by {ds}")
            if not os.path.isdir(config.data.src_dir): error_msgs.append(f"data.src_dir missing: {config.data.src_dir}") # Added path to msg
            if not os.path.isdir(config.data.dst_dir): error_msgs.append(f"data.dst_dir missing: {config.data.dst_dir}") # Added path to msg
            if config.model.model_size_dims <= 0: error_msgs.append("model.size<=0")
            if config.dataloader.num_workers < 0: error_msgs.append("loader.workers<0")
            if config.dataloader.num_workers>0 and config.dataloader.prefetch_factor is not None and config.dataloader.prefetch_factor<2: error_msgs.append("loader.prefetch<2 invalid")
            if config.saving.save_iterations_interval < 0: error_msgs.append("save.interval<0")
            if config.saving.keep_last_checkpoints < -1: error_msgs.append("save.keep<-1")
            if config.logging.log_interval <= 0: error_msgs.append("log.interval<=0")
            if config.logging.preview_batch_interval < 0: error_msgs.append("log.preview<0")
            if config.logging.preview_refresh_rate < 0: error_msgs.append("log.refresh<0")
        except Exception as e: error_msgs.append(f"Validation error: {e}")
    # Abort on Errors
    if error_msgs or missing:
         print("\n" + "="*30 + " CONFIG ERRORS " + "="*30); all_errs = sorted(list(set(error_msgs + [f"Missing: {m}" for m in missing])))
         for i, msg in enumerate(all_errs): print(f"  {i+1}. {msg}")
         print("="*82 + "\nPlease fix config and restart."); exit(1)
    else: logging.info("[CONFIG] Configuration validated.") # Keep simple INFO confirmation

    # DDP Env Var Check (Keep as Warning)
    if os.environ.get("WORLD_SIZE", "1") != "1" and os.environ.get("LOCAL_RANK", None) is None:
        logging.warning("DDP run detected (WORLD_SIZE>1) but LOCAL_RANK not set. Use 'torchrun' or ensure env vars are set.") # Keep Warning

    # --- Start Training ---
    try: train(config)
    except Exception as main_err: logging.error("Unhandled error outside train loop:", exc_info=True); cleanup_ddp(); exit(1) # Saiu 

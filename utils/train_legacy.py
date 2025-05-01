# train.py
import os
import argparse
import math
from glob import glob
from PIL import Image
import logging
import time
import random
import yaml
from types import SimpleNamespace
import re
import copy # Needed for deep copy during merge
import numpy as np # For potential use with augmentations

import torch
import torch.nn as nn
import torch.optim as optim
# 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
import lpips

# --- Import from dataloader --- #
from dataloader.data import AugmentedImagePairSlicingDataset, create_augmentations

# --- DDP Setup ---
#
def setup_ddp():
    if not dist.is_initialized():
        if "LOCAL_RANK" not in os.environ:
             raise RuntimeError("DDP environment variables (e.g., LOCAL_RANK) not found. Please launch with torchrun.")
        try: dist.init_process_group(backend="nccl")
        except Exception as e: print(f"Error initializing DDP: {e}. Check NCCL compatibility and DDP environment variables."); raise
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

def cleanup_ddp():
    if dist.is_initialized(): dist.destroy_process_group()
def get_rank():
    if not dist.is_available() or not dist.is_initialized(): return 0
    return dist.get_rank()
def get_world_size():
    if not dist.is_available() or not dist.is_initialized(): return 1
    return dist.get_world_size()
def is_main_process(): return get_rank() == 0

# --- UNet Model & Components ---
#
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        mid_channels = max(1, mid_channels) # Ensure mid_channels is at least 1
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

# --- UNet Model ---
class UNet(nn.Module):
    def __init__(self, config, n_ch=3, n_cls=3, bilinear=True):
        super().__init__(); self.n_ch = n_ch; self.n_cls = n_cls;
        # Use model_size_dims from config
        self.hidden_size = config.model.model_size_dims
        self.bilinear = bilinear
        h = self.hidden_size; chs = {'enc1': h, 'enc2': h*2, 'enc3': h*4, 'enc4': h*8, 'bottle': h*16}
        # Ensure channels are at least 1
        for k in chs: chs[k] = max(1, chs[k])
        self.inc = DoubleConv(n_ch, chs['enc1']); self.down1 = Down(chs['enc1'], chs['enc2']); self.down2 = Down(chs['enc2'], chs['enc3']); self.down3 = Down(chs['enc3'], chs['enc4']); self.down4 = Down(chs['enc4'], chs['bottle'])
        self.up1 = Up(chs['bottle'], chs['enc4'], chs['enc4'], bilinear); self.up2 = Up(chs['enc4'], chs['enc3'], chs['enc3'], bilinear); self.up3 = Up(chs['enc3'], chs['enc2'], chs['enc2'], bilinear); self.up4 = Up(chs['enc2'], chs['enc1'], chs['enc1'], bilinear)
        self.outc = OutConv(chs['enc1'], n_cls)
    def forward(self, x):
        x1=self.inc(x); x2=self.down1(x1); x3=self.down2(x2); x4=self.down3(x3); x5=self.down4(x4)
        x=self.up1(x5, x4); x=self.up2(x, x3); x=self.up3(x, x2); x=self.up4(x, x1); return self.outc(x)

# --- Dataset ---
#


# --- Helper Functions ---
#
NORM_MEAN = [0.5, 0.5, 0.5]; NORM_STD = [0.5, 0.5, 0.5]
def denormalize(tensor):
    mean=torch.tensor(NORM_MEAN,device=tensor.device).view(1,3,1,1); std=torch.tensor(NORM_STD,device=tensor.device).view(1,3,1,1)
    return torch.clamp(tensor*std+mean,0,1)

def save_previews(model, fixed_src_batch, fixed_dst_batch, output_dir, current_epoch, global_step, device, preview_save_count, preview_refresh_rate):
    if not is_main_process() or fixed_src_batch is None or fixed_dst_batch is None: return
    num_grid_cols=3;
    if fixed_src_batch.size(0)<num_grid_cols: return
    # Ensure batches are on CPU before manipulation if they aren't already
    src_select=fixed_src_batch[:num_grid_cols].cpu(); dst_select=fixed_dst_batch[:num_grid_cols].cpu()
    model.eval(); device_type=device.type
    with torch.no_grad(), autocast(device_type=device_type, enabled=torch.is_autocast_enabled()):
        src_dev=src_select.to(device); model_module=model.module if isinstance(model,DDP) else model; predicted_batch=model_module(src_dev)
    model.train()
    pred_select=predicted_batch.cpu().float()
    # Denormalize the (potentially augmented) batches for preview
    src_denorm=denormalize(src_select); pred_denorm=denormalize(pred_select); dst_denorm=denormalize(dst_select)
    combined=[item for i in range(num_grid_cols) for item in [src_denorm[i],dst_denorm[i],pred_denorm[i]]]
    if not combined: return
    grid_tensor=torch.stack(combined); grid=make_grid(grid_tensor,nrow=num_grid_cols,padding=2,normalize=False)
    img_pil=T.functional.to_pil_image(grid); preview_filename=os.path.join(output_dir,"training_preview.jpg")
    try:
        img_pil.save(preview_filename,"JPEG",quality=95)
        log_msg = f"Saved preview to {preview_filename} (Epoch {current_epoch+1}, Step {global_step}, Save #{preview_save_count})"
        # Log refresh status correctly
        refreshed_this_time = preview_refresh_rate > 0 and preview_save_count > 0 and preview_save_count % preview_refresh_rate == 0
        if refreshed_this_time:
             logging.info(log_msg + " - Refreshed Batch (Augmented)") # Indicate augmented preview
        else:
             logging.info(log_msg)
    except Exception as e: logging.error(f"Failed to save preview image: {e}")

# --- capture_preview_batch (augmentation pipelines) ---
# Takes augmentation pipelines and standard_transform now
def capture_preview_batch(config, src_transforms, dst_transforms, shared_transforms, standard_transform):
    if not is_main_process(): return None, None
    num_preview_samples = 3
    logging.info(f"Capturing/Refreshing fixed batch ({num_preview_samples} samples) for previews (with augmentations)...") # Updated log message
    try:
        # Use the augmented dataset CLASS, and pass the ACTUAL augmentation pipelines
        preview_dataset = AugmentedImagePairSlicingDataset(
            config.data.src_dir, config.data.dst_dir, config.data.resolution,
            overlap_factor=config.data.overlap_factor,
            src_transforms=src_transforms,         # <--- Pass the actual pipeline
            dst_transforms=dst_transforms,         # <--- Pass the actual pipeline
            shared_transforms=shared_transforms,   # <--- Pass the actual pipeline
            final_transform=standard_transform     # Apply ToTensor/Normalize last
        )

        if len(preview_dataset) >= num_preview_samples:
            # Shuffle=True ensures we get random samples which will then be augmented randomly based on 'p'
            preview_loader = DataLoader(preview_dataset, batch_size=num_preview_samples, shuffle=True, num_workers=0)
            fixed_src_slices, fixed_dst_slices = next(iter(preview_loader))
            if fixed_src_slices.size(0) == num_preview_samples:
                logging.info(f"Captured new batch of size {num_preview_samples} for previews (augmentations will be applied).")
                return fixed_src_slices.cpu(), fixed_dst_slices.cpu()
            else:
                logging.warning(f"Preview DataLoader returned {fixed_src_slices.size(0)} samples instead of {num_preview_samples}.")
                return None, None
        else:
            logging.warning(f"Dataset has only {len(preview_dataset)} slices, need {num_preview_samples}. Cannot refresh preview.")
            return None, None
    except StopIteration:
        logging.error("Preview DataLoader yielded no batches during refresh.")
        return None, None
    except Exception as e:
        logging.exception(f"Error capturing preview batch: {e}")
        return None, None

# --- Helper to convert nested dict to nested SimpleNamespace ---
#
def dict_to_namespace(d):
    if isinstance(d,dict):
        safe_d={};
        for k,v in d.items(): safe_key=k.replace('-','_'); safe_d[safe_key]=dict_to_namespace(v)
        return SimpleNamespace(**safe_d)
    elif isinstance(d,list): return [dict_to_namespace(item) for item in d]
    else: return d

# --- Checkpoint Helper Functions ---
#
CHECKPOINT_FILENAME_PATTERN = "tunet_{epoch:09d}.pth"
LATEST_CHECKPOINT_FILENAME = "tunet_latest.pth"
def save_checkpoint(model, optimizer, scaler, config, global_step, current_epoch, output_dir):
    if not is_main_process(): return
    # Ensure config is serializable (SimpleNamespace might not be directly)
    config_to_save = vars(config) if isinstance(config, SimpleNamespace) else config
    if isinstance(config_to_save, dict): # Recursively convert nested Namespaces in config
        def namespace_to_dict_recursive(ns):
            if isinstance(ns, SimpleNamespace):
                return {k: namespace_to_dict_recursive(v) for k, v in vars(ns).items()}
            elif isinstance(ns, list):
                return [namespace_to_dict_recursive(item) for item in ns]
            else:
                return ns
        config_to_save = namespace_to_dict_recursive(SimpleNamespace(**config_to_save))


    model_state_to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    checkpoint = {
        'global_step': global_step, 'epoch': current_epoch,
        'model_state_dict': model_state_to_save, 'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if config.training.use_amp else None,
        'config': config_to_save } # Save the converted dict
    filename = CHECKPOINT_FILENAME_PATTERN.format(epoch=current_epoch + 1) # Save based on completed epoch
    checkpoint_path = os.path.join(output_dir, filename)
    latest_path = os.path.join(output_dir, LATEST_CHECKPOINT_FILENAME)
    try:
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")
        # Save latest checkpoint link as well
        torch.save(checkpoint, latest_path) # Overwrite latest
    except Exception as e: logging.error(f"Failed to save checkpoint {checkpoint_path}: {e}")

def manage_checkpoints(output_dir, keep_last):
    if not is_main_process() or keep_last <= 0: return
    try:
        checkpoint_files = glob(os.path.join(output_dir, "tunet_*.pth"))
        numbered_checkpoints = []
        for f in checkpoint_files:
            basename = os.path.basename(f)
            if basename == LATEST_CHECKPOINT_FILENAME: continue
            match = re.match(r"tunet_(\d+)\.pth", basename)
            if match:
                epoch_num = int(match.group(1))
                # Use file modification time as a secondary sort key for robustness
                mtime = os.path.getmtime(f)
                numbered_checkpoints.append({'path': f, 'epoch': epoch_num, 'mtime': mtime})
        if len(numbered_checkpoints) <= keep_last: return

        # Sort by epoch, then modification time (newest first)
        numbered_checkpoints.sort(key=lambda x: (x['epoch'], x['mtime']), reverse=True)

        # Checkpoints to delete are all except the 'keep_last' newest ones
        checkpoints_to_delete = numbered_checkpoints[keep_last:]

        logging.info(f"Found {len(numbered_checkpoints)} numbered checkpoints. Keeping last {keep_last}. Deleting {len(checkpoints_to_delete)} older checkpoints.")
        deleted_count = 0
        for ckpt_info in checkpoints_to_delete:
            try:
                os.remove(ckpt_info['path'])
                logging.info(f"  Deleted old checkpoint: {os.path.basename(ckpt_info['path'])}")
                deleted_count += 1
            except OSError as e: logging.error(f"  Failed to delete checkpoint {ckpt_info['path']}: {e}")
        if deleted_count > 0: logging.info(f"Finished deleting {deleted_count} older checkpoints.")
    except Exception as e: logging.error(f"Error during checkpoint management: {e}", exc_info=True)


# --- Training Function ---
def train(config):
    # --- DDP Setup, Logging Setup ---
    #
    setup_ddp()
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{rank}")
    device_type = device.type
    log_level = logging.INFO if is_main_process() else logging.WARNING
    log_format = f'%(asctime)s [RK{rank}][%(levelname)s] %(message)s'
    handlers = [logging.StreamHandler()]
    if is_main_process():
        os.makedirs(config.data.output_dir, exist_ok=True)
        log_file = os.path.join(config.data.output_dir, 'training.log')
        handlers.append(logging.FileHandler(log_file, mode='a')) # Append mode
    # Force=True overrides existing handlers if script is re-run in same process
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers, force=True)

    if is_main_process():
        logging.info("="*50)
        logging.info("Starting training run...")
        logging.info(f"Runtime World Size (Number of GPUs/Processes): {world_size}")
        logging.info(f"Iteration-based training. Steps per epoch: {config.training.iterations_per_epoch}")
        # Log the entire config for reproducibility
        logging.info(f"Using effective configuration:")
        # Pretty print the namespace config
        def print_config(cfg, indent=0):
             prefix = "  " * indent
             if isinstance(cfg, SimpleNamespace):
                  sorted_keys = sorted(vars(cfg).keys())
                  for k in sorted_keys:
                      v = getattr(cfg, k)
                      if isinstance(v, SimpleNamespace):
                           logging.info(f"{prefix}{k}:")
                           print_config(v, indent + 1)
                      else:
                           logging.info(f"{prefix}{k}: {v}")
             elif isinstance(cfg, list):
                  logging.info(f"{prefix}[")
                  for item in cfg: print_config(item, indent+1)
                  logging.info(f"{prefix}]")
             else: # Should not happen at top level if conversion worked
                  logging.info(f"{prefix}{cfg}")
        print_config(config)
        logging.info("-" * 50)


    # --- Create Standard Transform (used after augmentations) --- #
    standard_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    # --- Create Augmentations --- #
    src_augs_list = []
    dst_augs_list = []
    shared_augs_list = []
    src_transforms = None
    dst_transforms = None
    shared_transforms = None

    # Safely access nested config attributes using hasattr
    if hasattr(config, 'dataloader') and hasattr(config.dataloader, 'datasets'):
        datasets_config = config.dataloader.datasets
        src_augs_list = getattr(datasets_config, 'src_augs', [])
        dst_augs_list = getattr(datasets_config, 'dst_augs', [])
        shared_augs_list = getattr(datasets_config, 'shared_augs', [])
        if is_main_process():
             if not any([src_augs_list, dst_augs_list, shared_augs_list]):
                 logging.info("No augmentations specified in config.dataloader.datasets.")
             else:
                 logging.info("Augmentation config found. Creating pipelines...")
                 # Be careful logging potentially large lists, maybe just counts or types
                 logging.info(f"  Source Augs: {len(src_augs_list)} steps")
                 logging.info(f"  Dest Augs: {len(dst_augs_list)} steps")
                 logging.info(f"  Shared Augs: {len(shared_augs_list)} steps")
    else:
         if is_main_process():
             logging.warning("Config section 'dataloader.datasets' not found. No augmentations will be used.")

    # Ensure lists are actually lists before proceeding
    src_augs_list = src_augs_list if isinstance(src_augs_list, list) else []
    dst_augs_list = dst_augs_list if isinstance(dst_augs_list, list) else []
    shared_augs_list = shared_augs_list if isinstance(shared_augs_list, list) else []

    try:
        src_transforms = create_augmentations(src_augs_list)
        dst_transforms = create_augmentations(dst_augs_list)
        shared_transforms = create_augmentations(shared_augs_list)
        if is_main_process() and any([src_augs_list, dst_augs_list, shared_augs_list]):
            logging.info("Augmentation pipelines created successfully.")
    except Exception as e:
        # Use FATAL level if available, otherwise ERROR. Exit cleanly.
        logging.error(f"FATAL: Failed to create augmentations: {e}. Exiting.", exc_info=True)
        if dist.is_initialized(): cleanup_ddp()
        exit(1)


    # --- Dataset & DataLoader --- #
    dataset = None
    try:
        # Use the new Augmented Dataset class
        dataset = AugmentedImagePairSlicingDataset(
            config.data.src_dir, config.data.dst_dir, config.data.resolution,
            overlap_factor=config.data.overlap_factor,
            src_transforms=src_transforms,         # Pass created augmentation pipelines
            dst_transforms=dst_transforms,
            shared_transforms=shared_transforms,
            final_transform=standard_transform     # Pass the standard transform to apply last
        )
        # Logging now uses properties exposed by AugmentedImagePairSlicingDataset
        if is_main_process():
            logging.info(f"Augmented Dataset: Res={dataset.resolution}, Overlap={dataset.overlap_factor:.2f}, Stride={dataset.stride}")
            if dataset.skipped_count > 0:
                logging.warning(f"Skipped {dataset.skipped_count} pairs during dataset init.")
                # Log first few skipped paths for debugging if needed
                limit = 5
                logged = 0
                for path, reason in dataset.skipped_paths:
                   if logged < limit:
                       logging.warning(f"  - Skip reason: {os.path.basename(path)} -> {reason}")
                       logged += 1
                   else:
                       logging.warning(f"  ... (logged {limit} skip reasons)")
                       break

            if dataset.processed_files > 0:
                avg_slices = dataset.total_slices_generated / dataset.processed_files if dataset.processed_files > 0 else 0
                logging.info(f"Found {dataset.total_slices_generated} usable slices from {dataset.processed_files} valid pairs (avg {avg_slices:.1f}).")
            else:
                # If no files processed, log all skip reasons (or up to a limit)
                logging.error(f"Dataset processed 0 valid image pairs. Total skipped: {dataset.skipped_count}.")
                if dataset.skipped_count > 0:
                     limit = 20
                     logged = 0
                     for path, reason in dataset.skipped_paths:
                          if logged < limit:
                               logging.error(f"  - Skipped: {os.path.basename(path)} -> {reason}")
                               logged +=1
                          else:
                               logging.error(f"  ... (logged {limit} skip reasons)")
                               break
                raise ValueError("Dataset initialization failed: Processed 0 valid image pairs. Check logs for skip reasons and paths.")

            if len(dataset) == 0:
                # This case should ideally be caught by processed_files == 0, but double-check
                raise ValueError("Dataset created but contains 0 slices. Check slicing logic or image dimensions.")

    except (FileNotFoundError, ValueError, Exception) as e:
        logging.error(f"Dataset initialization failed: {e}", exc_info=not isinstance(e, (FileNotFoundError, ValueError)))
        if dist.is_initialized(): cleanup_ddp()
        exit(1) # Exit immediately if dataset fails

    # Ensure dataset has enough samples for DDP drop_last=True
    if len(dataset) < world_size * config.training.batch_size:
         logging.error(f"FATAL: Dataset size ({len(dataset)}) is too small for world size ({world_size}) and batch size ({config.training.batch_size}) with drop_last=True. Need at least {world_size * config.training.batch_size} samples.")
         if dist.is_initialized(): cleanup_ddp()
         exit(1)


    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, sampler=sampler, num_workers=config.training.num_workers,
                            pin_memory=True, drop_last=True, prefetch_factor=config.training.prefetch_factor if config.training.num_workers > 0 else None,
                            persistent_workers=True if config.training.num_workers > 0 else False) # Add persistent workers
    batches_per_dataset_pass = len(dataloader) # This is the number of batches per rank per epoch pass
    if is_main_process():
        logging.info(f"DataLoader created. Batches per dataset pass (per GPU): {batches_per_dataset_pass}")
        if batches_per_dataset_pass < config.training.iterations_per_epoch:
             logging.warning(f"iterations_per_epoch ({config.training.iterations_per_epoch}) is greater than batches per pass ({batches_per_dataset_pass}). Dataset will be iterated multiple times per 'epoch'.")
        elif batches_per_dataset_pass > config.training.iterations_per_epoch:
             logging.warning(f"iterations_per_epoch ({config.training.iterations_per_epoch}) is less than batches per pass ({batches_per_dataset_pass}). Not all data will be seen in one 'epoch'.")


    # --- Model Config, Instantiation, Optimizer, Scaler, Loss ---
    # model_size_dims logic for l1+lpips
    model_size = config.model.model_size_dims
    use_lpips = False; loss_fn_lpips = None
    default_hidden_size_for_bump = 64 # The specific value that triggers the bump
    bumped_hidden_size = 96 # The value to bump to

    if config.training.loss == 'l1+lpips':
        use_lpips = True
        # Check if the *original* specified size requires bumping
        if model_size == default_hidden_size_for_bump:
            original_size_before_bump = model_size # Store for config update check
            model_size = bumped_hidden_size # Update the size to be used for instantiation
            # Update the config object IN PLACE only if it hasn't been updated already (e.g. by CLI)
            if config.model.model_size_dims == original_size_before_bump:
                 config.model.model_size_dims = model_size
                 if is_main_process(): logging.info(f"Using l1+lpips loss: Auto-increasing model_size_dims {original_size_before_bump} -> {model_size}.")
            else:
                 # If config already reflects the bumped size (e.g. from CLI override), just log it
                 if is_main_process(): logging.info(f"Using l1+lpips loss: model_size_dims={model_size} (already set/overridden).")
        else:
             if is_main_process(): logging.info(f"Using l1+lpips loss: model_size_dims={model_size} (as specified), LPIPS lambda={config.training.lambda_lpips}.")

        if is_main_process(): # Initialize LPIPS only on main process first for download safety
            try:
                loss_fn_lpips = lpips.LPIPS(net='vgg').to(device).eval() # Move to device
                for param in loss_fn_lpips.parameters(): param.requires_grad = False
                logging.info("LPIPS VGG loss initialized on rank 0.")
            except Exception as e:
                 logging.error(f"Failed to init LPIPS: {e}. Disabling LPIPS.", exc_info=True); use_lpips = False
            # Broadcast LPIPS status (necessary if initialization fails on rank 0)
            lpips_status = [use_lpips]
            if world_size > 1: dist.broadcast_object_list(lpips_status, src=0)
            use_lpips = lpips_status[0]
            if not use_lpips and is_main_process(): logging.warning("LPIPS disabled across all ranks due to init failure on rank 0.")

        # If rank > 0 and use_lpips is True (meaning rank 0 succeeded), initialize LPIPS here too
        if rank > 0 and use_lpips:
             try:
                 # Re-initialize on other ranks (weights should be cached after rank 0)
                 loss_fn_lpips = lpips.LPIPS(net='vgg').to(device).eval()
                 for param in loss_fn_lpips.parameters(): param.requires_grad = False
             except Exception as e:
                  # This ideally shouldn't happen if rank 0 succeeded, but handle it
                  logging.error(f"Rank {rank}: Failed to init LPIPS even though rank 0 succeeded: {e}. Disabling LPIPS locally.", exc_info=True)
                  use_lpips = False # Disable locally if fails here
                  # We might have divergence here if other ranks fail, but training might continue with L1 only

    else: # Loss is 'l1'
        if is_main_process(): logging.info(f"Using {config.training.loss} loss: model_size_dims={model_size}.")


    # Instantiate the model using the (potentially modified) config
    model = UNet(config).to(device)

    try: lr_value = float(config.training.lr)
    except (ValueError, TypeError) as e:
        logging.error(f"FATAL: Could not convert learning rate '{config.training.lr}' (type: {type(config.training.lr)}) to float: {e}")
        raise ValueError(f"Invalid learning rate value in config: {config.training.lr}") from e
    optimizer = optim.AdamW(model.parameters(), lr=lr_value, weight_decay=1e-5) # Consider making weight decay configurable
    scaler = GradScaler(enabled=config.training.use_amp)
    criterion_l1 = nn.L1Loss()

    # --- Resume Logic ---
    start_iteration = 0
    latest_checkpoint_path = os.path.join(config.data.output_dir, LATEST_CHECKPOINT_FILENAME)
    resume_flag = [False] * 1 # Use list for broadcast
    if is_main_process(): resume_flag[0] = os.path.exists(latest_checkpoint_path)
    if world_size > 1: dist.broadcast_object_list(resume_flag, src=0) # Broadcast existence status

    if resume_flag[0]:
        try:
            if is_main_process(): logging.info(f"Attempting resume from: {latest_checkpoint_path}")
            # Load checkpoint to CPU first to avoid GPU memory spike on rank 0
            checkpoint = torch.load(latest_checkpoint_path, map_location='cpu')

            # --- Compatibility Check ---
            if 'config' not in checkpoint: raise ValueError("Checkpoint missing 'config'.")
            ckpt_config_raw = checkpoint['config'] # Saved as dict now
            ckpt_config = dict_to_namespace(ckpt_config_raw) # Convert back to namespace for checks

            # Check critical parameters for compatibility
            # Loss Mode
            ckpt_loss = getattr(getattr(ckpt_config, 'training', None), 'loss', 'l1') # Safe access
            current_loss = config.training.loss
            if ckpt_loss != current_loss:
                raise ValueError(f"Loss mode mismatch: Checkpoint='{ckpt_loss}', Requested='{current_loss}'")

            # Model Size (Effective Size after potential bump)
            ckpt_size_saved = getattr(getattr(ckpt_config, 'model', None), 'model_size_dims', 64) # Safe access
            ckpt_expected_size = ckpt_size_saved
            if ckpt_loss == 'l1+lpips' and ckpt_size_saved == default_hidden_size_for_bump:
                ckpt_expected_size = bumped_hidden_size
            current_expected_size = config.model.model_size_dims # This is already bumped if needed
            if ckpt_expected_size != current_expected_size:
                raise ValueError(f"Model size mismatch: Checkpoint effective={ckpt_expected_size}, Requested={current_expected_size}")

            # Data Resolution
            ckpt_res = getattr(getattr(ckpt_config, 'data', None), 'resolution', 512)
            current_res = config.data.resolution
            if ckpt_res != current_res:
                 raise ValueError(f"Data resolution mismatch: Checkpoint={ckpt_res}, Requested={current_res}")

            if is_main_process(): logging.info("Checkpoint configuration compatible.")
            # --- End Compatibility Check ---

            # Load State Dicts (strict=True by default is good practice)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if config.training.use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            else:
                 if config.training.use_amp and ('scaler_state_dict' not in checkpoint or checkpoint['scaler_state_dict'] is None):
                      logging.warning("Resuming with AMP enabled, but no scaler state found in checkpoint. Initializing new scaler state.")

            # Load global_step
            if 'global_step' in checkpoint:
                start_iteration = checkpoint['global_step']
                if is_main_process(): logging.info(f"Resuming training from iteration {start_iteration + 1}.")
            else: # Fallback for older checkpoints without global_step
                 start_epoch_legacy = checkpoint.get('epoch', -1) # Use -1 to distinguish from epoch 0
                 if start_epoch_legacy >= 0:
                     # Estimate iteration based on *current* iterations_per_epoch
                     # Note: This might be inaccurate if iterations_per_epoch changed!
                     estimated_iteration = (start_epoch_legacy + 1) * config.training.iterations_per_epoch
                     logging.warning(f"Checkpoint missing 'global_step'. Found legacy 'epoch' {start_epoch_legacy}.")
                     logging.warning(f"Estimating resume iteration as {estimated_iteration} based on completed epoch {start_epoch_legacy+1} and current iterations_per_epoch={config.training.iterations_per_epoch}. This might be inaccurate if config changed.")
                     start_iteration = estimated_iteration
                 else:
                     logging.warning("Checkpoint missing 'global_step' and valid 'epoch'. Starting from iteration 0.")
                     start_iteration = 0

            del checkpoint # Free memory
            if world_size > 1: dist.barrier() # Ensure all ranks load checkpoint before proceeding

        except Exception as e:
            logging.error(f"Failed load/resume from {latest_checkpoint_path}: {e}", exc_info=True)
            logging.warning("Starting training from scratch (Iteration 0).")
            start_iteration = 0
            # Ensure barrier even on failure if DDP initialized, prevent ranks from proceeding differently
            if world_size > 1 and dist.is_initialized():
                 logging.warning("Resume failed on rank 0, forcing start from scratch on all ranks.")
                 # Broadcast a failure signal? Simpler to just let others timeout or proceed from 0. Barrier helps sync.
                 dist.barrier()

    else: # No checkpoint found
        if is_main_process(): logging.info(f"No checkpoint found at {latest_checkpoint_path}. Starting from iteration 0.")
        start_iteration = 0
        # Barrier necessary? Not strictly, as all ranks will start from 0.
        # if world_size > 1: dist.barrier()


    # --- Wrap model with DDP ---
    # It's generally recommended to wrap *after* loading state_dict for non-DDP checkpoints
    # find_unused_parameters=False can speed up training if you are sure there are none.
    # If unsure, or if using model variants, set to True. Let's default to False for UNet.
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)


    # --- Preview State ---
    #
    fixed_src_slices = None; fixed_dst_slices = None; preview_save_count = 0
    # Capture initial batch only if starting from scratch and previews are enabled
    if start_iteration == 0 and is_main_process() and config.logging.preview_batch_interval > 0:
         logging.info("Capturing initial fixed batch for previews (with augmentations)...") # UPDATED Log
         # Pass the created augmentation pipelines and the standard_transform
         # --- UPDATED CALL ---
         fixed_src_slices, fixed_dst_slices = capture_preview_batch(
             config, src_transforms, dst_transforms, shared_transforms, standard_transform
         )
         # --- END UPDATED CALL ---
         if fixed_src_slices is None:
             logging.warning("Could not capture initial preview batch.")


    # --- Training Loop ---
    #
    global_step = start_iteration
    # Calculate current epoch based on starting iteration and steps per epoch
    current_epoch = global_step // config.training.iterations_per_epoch

    if is_main_process():
         logging.info(f"Starting training loop from Global Step: {global_step + 1} (Epoch: {current_epoch + 1})")

    epoch_start_time = time.time() # For epoch timing

    try: # Wrap main loop for KeyboardInterrupt
        while True: # Loop indefinitely until manually stopped or scheduler finishes (if any)
            model.train()
            sampler.set_epoch(current_epoch) # Ensure proper shuffling each epoch

            # Log epoch start on main process
            # Check if this iteration starts a new epoch
            is_new_epoch_start = (global_step % config.training.iterations_per_epoch == 0)

            if is_main_process() and is_new_epoch_start:
                epoch_duration = time.time() - epoch_start_time if current_epoch > 0 else 0
                log_epoch_msg = f"Starting Epoch {current_epoch + 1}"
                if epoch_duration > 0:
                    log_epoch_msg += f" (Previous epoch duration: {epoch_duration:.2f}s)"
                # Approx dataset pass assumes full pass aligns with iterations_per_epoch
                dataset_pass_approx = (global_step // config.training.iterations_per_epoch) + 1
                log_epoch_msg += f" (Dataset Pass approx {dataset_pass_approx})"
                logging.info(log_epoch_msg)
                epoch_start_time = time.time() # Reset timer for new epoch


            batch_iter_start_time = time.time()
            data_load_start_time = time.time()

            for i, batch_data in enumerate(dataloader):
                data_load_end_time = time.time() # Time to fetch batch

                # --- Batch Processing ---
                if batch_data is None:
                     if is_main_process(): logging.warning(f"Step[{global_step+1}]: Received None data from DataLoader, skipping batch.")
                     # This shouldn't happen with standard DataLoader unless dataset __getitem__ returns None
                     continue # Skip to next iteration

                try: src_slices, dst_slices = batch_data
                except Exception as e:
                    if is_main_process(): logging.error(f"Step[{global_step+1}]: Error unpacking batch data: {e}. Skipping batch.", exc_info=True)
                    # Potentially problematic, might indicate issues in dataset or collation
                    global_step += 1 # Increment step even if batch fails? Debatable. Let's increment to avoid potential infinite loops on bad data.
                    continue

                iter_prep_end_time = time.time() # Time to unpack + move to device (if non_blocking)

                src_slices=src_slices.to(device, non_blocking=True)
                dst_slices=dst_slices.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True) # More memory efficient

                compute_start_time = time.time()
                with autocast(device_type=device_type, enabled=config.training.use_amp):
                    outputs = model(src_slices)
                    l1_loss = criterion_l1(outputs, dst_slices)
                    lpips_loss = torch.tensor(0.0, device=device) # Default zero loss

                    # Calculate LPIPS loss only if enabled and function available
                    if use_lpips and loss_fn_lpips is not None:
                        try:
                            # Ensure inputs to LPIPS are in expected range [-1, 1] or [0, 1]
                            # Our denormalize goes [0,1]. LPIPS typically expects [-1, 1].
                            # Let's assume LPIPS handles [0,1] via its internal normalization,
                            # but if results seem odd, transform outputs/dst_slices to [-1,1] here.
                            # Example: outputs_lpips = outputs * 2.0 - 1.0
                            # Example: dst_slices_lpips = dst_slices * 2.0 - 1.0
                            # lpips_loss = loss_fn_lpips(outputs_lpips, dst_slices_lpips).mean()
                            lpips_loss = loss_fn_lpips(outputs, dst_slices).mean() # Using [0,1] range output from model/target

                        except Exception as e:
                             # Log LPIPS errors less frequently to avoid spamming
                             if is_main_process() and global_step % (config.logging.log_interval * 10) == 0:
                                  logging.warning(f"Step[{global_step+1}]: LPIPS calculation failed: {e}. Setting LPIPS loss to 0 for this batch.", exc_info=False) # No traceback usually
                             lpips_loss=torch.tensor(0.0,device=device) # Ensure it's zero if calculation fails

                    # Combine losses
                    loss = l1_loss + config.training.lambda_lpips * lpips_loss

                # --- Backprop and Optimize ---
                # Check for NaN/Inf loss before backward pass
                if not torch.isfinite(loss):
                    if is_main_process(): logging.error(f"Step[{global_step+1}]: NaN or Inf loss detected! Loss={loss.item():.4f}, L1={l1_loss.item():.4f}, LPIPS={lpips_loss.item():.4f}. Skipping update.")
                    # Potentially dump tensors or grads for debugging here
                    # Consider adding gradient clipping if NaN/Inf occurs often
                    global_step += 1 # Increment step
                    # Reset timers? Continue to next iteration
                    batch_iter_start_time = time.time()
                    data_load_start_time = time.time()
                    continue # Skip optimizer step and logging for this batch

                # Scale loss, backward pass, optimizer step
                scaler.scale(loss).backward()
                # Optional: Gradient clipping (before scaler.step)
                # scaler.unscale_(optimizer) # Unscale gradients
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip norm
                scaler.step(optimizer)
                scaler.update() # Update scaler for next iteration

                compute_end_time = time.time() # Time for forward, loss, backward, step
                global_step += 1

                # --- Logging, Preview, Checkpoint (on main process) ---
                if is_main_process():
                    # Loss Logging (gather losses from all ranks for accurate average)
                    if global_step % config.logging.log_interval == 0:
                        # Reduce losses across all ranks
                        l1_loss_reduced = l1_loss.detach().clone()
                        lpips_loss_reduced = lpips_loss.detach().clone()
                        if world_size > 1:
                             dist.all_reduce(l1_loss_reduced, op=dist.ReduceOp.AVG)
                             if use_lpips: dist.all_reduce(lpips_loss_reduced, op=dist.ReduceOp.AVG)

                        batch_l1_loss_avg = l1_loss_reduced.item()
                        batch_lpips_loss_avg = lpips_loss_reduced.item() if use_lpips else 0.0

                        current_lr=optimizer.param_groups[0]['lr']
                        data_time = data_load_end_time - data_load_start_time
                        prep_time = iter_prep_end_time - data_load_end_time
                        compute_time = compute_end_time - compute_start_time
                        total_step_time = time.time() - batch_iter_start_time # Includes sync time? Approx.

                        log_msg = (f'Epoch[{current_epoch+1}] Step[{global_step}/{config.training.iterations_per_epoch*(current_epoch+1)}] ' # Show progress within epoch
                                   f'L1:{batch_l1_loss_avg:.4f}')
                        if use_lpips: log_msg += (f' LPIPS:{batch_lpips_loss_avg:.4f}')
                        log_msg += (f' LR:{current_lr:.2e} ' # Shorter LR format
                                    f'T/Step:{total_step_time:.3f}s '
                                    f'(D:{data_time:.3f} P:{prep_time:.3f} C:{compute_time:.3f})')
                        logging.info(log_msg)

                        # Reset batch timer for next interval measurement
                        batch_iter_start_time = time.time()
                        data_load_start_time = time.time() # Reset data load timer too


                    # --- Preview Generation ---
                    if config.logging.preview_batch_interval > 0 and global_step % config.logging.preview_batch_interval == 0:
                        # Determine if refresh is needed based on count and rate
                        needs_refresh = (preview_save_count == 0) or \
                                        (config.logging.preview_refresh_rate > 0 and \
                                         preview_save_count > 0 and \
                                         (preview_save_count % config.logging.preview_refresh_rate == 0))

                        if needs_refresh and (fixed_src_slices is None or config.logging.preview_refresh_rate > 0):
                             # Only capture *new* batch if needed (first time or refresh interval)
                             logging.info(f"Step {global_step}: Refreshing preview batch (with augmentations)...") # UPDATED Log
                             # Pass the pipelines when refreshing
                             # --- UPDATED CALL ---
                             new_src, new_dst = capture_preview_batch(
                                 config, src_transforms, dst_transforms, shared_transforms, standard_transform
                             )
                             # --- END UPDATED CALL ---
                             if new_src is not None and new_dst is not None:
                                 fixed_src_slices, fixed_dst_slices = new_src, new_dst
                                 logging.info(f"Step {global_step}: Successfully refreshed preview batch.")
                             elif fixed_src_slices is None: # If refresh failed AND we never had a batch
                                  logging.warning(f"Step {global_step}: Failed to capture preview batch. Skipping preview generation.")
                             else:
                                  # If refresh fails but we had a previous batch, keep using it. It will be augmented again by save_previews' model call if needed.
                                  logging.warning(f"Step {global_step}: Failed to refresh preview batch. Reusing previous batch (will be re-augmented on save).")

                        # Save preview if we have a valid batch (either old or refreshed)
                        # The fixed_src/dst slices loaded here WILL be augmented based on the 'p' values
                        if fixed_src_slices is not None and fixed_dst_slices is not None:
                             # save_previews itself doesn't re-augment, it just displays what capture_preview_batch returned
                             save_previews(model, fixed_src_slices, fixed_dst_slices, config.data.output_dir,
                                           current_epoch, global_step, device,
                                           preview_save_count, config.logging.preview_refresh_rate)
                             preview_save_count += 1 # Increment count AFTER saving
                        elif is_main_process() and preview_save_count == 0 : # Log only if we never managed to save one
                             logging.warning(f"Step {global_step}: Skipping preview generation (no valid batch available).")


                    # --- Checkpoint Saving & Management ---
                    # Check if an epoch has just been completed (based on global_step and iterations_per_epoch)
                    epoch_just_finished = (global_step % config.training.iterations_per_epoch == 0)

                    if epoch_just_finished and global_step > 0:
                         completed_epoch_number = (global_step // config.training.iterations_per_epoch) - 1 # Epoch that just finished
                         logging.info(f"Epoch {completed_epoch_number + 1} finished at step {global_step}.")
                         save_checkpoint(model, optimizer, scaler, config, global_step, completed_epoch_number, config.data.output_dir)
                         manage_checkpoints(config.data.output_dir, config.saving.keep_last_checkpoints)


                # --- Check if epoch completed to advance epoch counter and potentially break inner loop ---
                # This logic ensures current_epoch increments correctly for the *next* iteration/epoch start log
                if global_step > 0 and global_step % config.training.iterations_per_epoch == 0:
                    current_epoch += 1 # Increment epoch counter for the next epoch
                    if world_size > 1: dist.barrier() # Sync ranks at epoch boundary
                    # No need to break the inner loop here, the outer while True handles continuation
                    # Breaking here would require managing the dataloader iterator externally, which is complex.
                    # Let the dataloader naturally finish or repeat based on the sampler.

            # --- End of inner dataloader loop (one pass through the data subset for this rank) ---
            # The loop continues to the next epoch via the outer `while True`


    except KeyboardInterrupt:
        if is_main_process(): logging.warning("Training interrupted by user (KeyboardInterrupt).")
        # Optional: Save a final checkpoint on interrupt?
        # if is_main_process():
        #     logging.info("Attempting to save final checkpoint on interrupt...")
        #     save_checkpoint(model, optimizer, scaler, config, global_step, current_epoch, config.data.output_dir)

    except Exception as e:
         # Catch unexpected errors during training loop
         logging.error("An unexpected error occurred during training:", exc_info=True)
         # Optional: Save checkpoint on other errors? Risky if state is corrupted.

    finally:
        # --- Cleanup ---
        if is_main_process(): logging.info("Cleaning up DDP...")
        cleanup_ddp()
        if is_main_process(): logging.info("Training script finished.")


# --- Config Merging Helper ---
# (Deep merge function)
def merge_dicts(base, user):
    merged = copy.deepcopy(base) # Start with a deep copy of the base
    for key, value in user.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            # If both values are dicts, recurse
            merged[key] = merge_dicts(merged[key], value)
        # elif isinstance(value, list) and key in merged and isinstance(merged[key], list):
            # Optionally handle list merging strategy (e.g., append, replace)
            # Default: Replace list entirely
            # merged[key] = merged[key] + value # Example: Append
            # merged[key] = value # Default: Replace
        else:
            # Otherwise, user value replaces base value
            merged[key] = value
    return merged


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TuNet using YAML config with DDP (Iteration-Based)')
    parser.add_argument('--config', type=str, required=True, help='Path to the USER configuration YAML file')
    # --- Define CLI overrides matching the nested structure ---
    # Format: --section.key value
    # Data Args
    parser.add_argument('--data.src_dir', type=str, dest='data_src_dir')
    parser.add_argument('--data.dst_dir', type=str, dest='data_dst_dir')
    parser.add_argument('--data.output_dir', type=str, dest='data_output_dir')
    parser.add_argument('--data.resolution', type=int, dest='data_resolution')
    parser.add_argument('--data.overlap_factor', type=float, dest='data_overlap_factor')
    # Model Args
    parser.add_argument('--model.model_size_dims', type=int, dest='model_model_size_dims')
    # Training Args
    parser.add_argument('--training.iterations_per_epoch', type=int, dest='training_iterations_per_epoch')
    parser.add_argument('--training.batch_size', type=int, dest='training_batch_size')
    parser.add_argument('--training.lr', type=float, dest='training_lr')
    parser.add_argument('--training.loss', type=str, choices=['l1', 'l1+lpips'], dest='training_loss')
    parser.add_argument('--training.lambda_lpips', type=float, dest='training_lambda_lpips')
    parser.add_argument('--training.use_amp', type=lambda x: (str(x).lower() == 'true'), dest='training_use_amp')
    parser.add_argument('--training.num_workers', type=int, dest='training_num_workers')
    parser.add_argument('--training.prefetch_factor', type=int, dest='training_prefetch_factor')
    # Logging Args
    parser.add_argument('--logging.log_interval', type=int, dest='logging_log_interval')
    parser.add_argument('--logging.preview_batch_interval', type=int, dest='logging_preview_batch_interval')
    parser.add_argument('--logging.preview_refresh_rate', type=int, dest='logging_preview_refresh_rate')
    # Saving Args
    parser.add_argument('--saving.keep_last_checkpoints', type=int, dest='saving_keep_last_checkpoints')
    # Add dataloader overrides if needed (e.g., disable specific augs via CLI - complex)

    cli_args = parser.parse_args()

    # --- Load Base Config ---
    base_config_dict = {}
    try:
        # Look for base.yaml relative to user config FIRST, then relative to script
        user_config_dir = os.path.dirname(os.path.abspath(cli_args.config))
        base_path1 = os.path.join(user_config_dir, 'base', 'base.yaml')

        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_path2 = os.path.join(script_dir, 'base', 'base.yaml')

        base_config_path = None
        if os.path.exists(base_path1): base_config_path = base_path1
        elif os.path.exists(base_path2): base_config_path = base_path2
        else:
            print(f"ERROR: Base configuration file ('base/base.yaml') not found relative to user config ('{user_config_dir}/base/') or script ('{script_dir}/base/').")
            exit(1)

        print(f"Loading base config from: {base_config_path}")
        with open(base_config_path, 'r') as f:
            base_config_dict = yaml.safe_load(f)
        if base_config_dict is None: base_config_dict = {} # Handle empty base config
        print("Base config loaded successfully.")
    except yaml.YAMLError as e: print(f"ERROR: Parsing base YAML file {base_config_path}: {e}"); exit(1)
    except Exception as e: print(f"ERROR: Reading base config file {base_config_path}: {e}"); exit(1)

    # --- Load User Config ---
    user_config_dict = {}
    try:
        print(f"Loading user config from: {cli_args.config}")
        with open(cli_args.config, 'r') as f:
            user_config_dict = yaml.safe_load(f)
        if user_config_dict is None: user_config_dict = {} # Handle empty user config
        print("User config loaded successfully.")
    except FileNotFoundError: print(f"ERROR: User configuration file not found: {cli_args.config}"); exit(1)
    except yaml.YAMLError as e: print(f"ERROR: Parsing user YAML file {cli_args.config}: {e}"); exit(1)
    except Exception as e: print(f"ERROR: Reading user config file {cli_args.config}: {e}"); exit(1)

    # --- Merge Configs (User overrides Base) ---
    merged_config_dict = merge_dicts(base_config_dict, user_config_dict)
    print("Configs merged (User overrides Base).")

    # --- Apply CLI Overrides (CLI overrides Merged) ---
    # Map flat CLI arg names ('data_src_dir') to nested dict keys (['data', 'src_dir'])
    override_map = {
        'data_src_dir': ['data', 'src_dir'],
        'data_dst_dir': ['data', 'dst_dir'],
        'data_output_dir': ['data', 'output_dir'],
        'data_resolution': ['data', 'resolution'],
        'data_overlap_factor': ['data', 'overlap_factor'],
        'model_model_size_dims': ['model', 'model_size_dims'],
        'training_iterations_per_epoch': ['training', 'iterations_per_epoch'],
        'training_batch_size': ['training', 'batch_size'],
        'training_lr': ['training', 'lr'],
        'training_loss': ['training', 'loss'],
        'training_lambda_lpips': ['training', 'lambda_lpips'],
        'training_use_amp': ['training', 'use_amp'],
        'training_num_workers': ['training', 'num_workers'],
        'training_prefetch_factor': ['training', 'prefetch_factor'],
        'logging_log_interval': ['logging', 'log_interval'],
        'logging_preview_batch_interval': ['logging', 'preview_batch_interval'],
        'logging_preview_refresh_rate': ['logging', 'preview_refresh_rate'],
        'saving_keep_last_checkpoints': ['saving', 'keep_last_checkpoints'],
    }
    def set_nested(dic, keys, value):
        for key in keys[:-1]: dic = dic.setdefault(key, {}) # Create intermediate dicts if needed
        dic[keys[-1]] = value

    applied_overrides = False
    for arg_key, nested_keys in override_map.items():
        value = getattr(cli_args, arg_key, None)
        if value is not None:
            set_nested(merged_config_dict, nested_keys, value)
            print(f"Applied CLI override: {' -> '.join(nested_keys)} = {value}")
            applied_overrides = True
    if applied_overrides: print("CLI overrides applied.")
    else: print("No CLI overrides provided.")

    # --- Convert final merged dict to Namespace for easier access ---
    config = dict_to_namespace(merged_config_dict)

    # --- Basic Configuration Validation ---
    print("Validating configuration...")
    required_keys = {
        # Section: [key1, key2, ...]
        'data': ['src_dir', 'dst_dir', 'output_dir', 'resolution', 'overlap_factor'],
        'model': ['model_size_dims'],
        'training': ['iterations_per_epoch', 'batch_size', 'lr', 'loss', 'lambda_lpips', 'use_amp', 'num_workers', 'prefetch_factor'],
        'logging': ['log_interval', 'preview_batch_interval', 'preview_refresh_rate'],
        'saving': ['keep_last_checkpoints']
        # dataloader section is optional, handled internally
    }
    missing = False # Flag to indicate if any validation error occurred
    error_msgs = [] # List to store specific error messages

    # Check top-level sections and their keys
    for section, keys in required_keys.items():
         if hasattr(config, section):
             section_obj = getattr(config, section)
             # --- Inlined logic from check_keys ---
             if section_obj is None: # Check if the section attribute exists but is None
                  error_msgs.append(f"Config section '{section}' exists but is None")
                  missing = True
                  continue # Skip checking keys for this section if the section object itself is None

             # Check each required key within the section
             for key in keys:
                 if not hasattr(section_obj, key):
                     error_msgs.append(f"Missing configuration key '{section}.{key}'")
                     missing = True
             # --- End Inlined logic ---
         else:
             # Section itself is missing
             error_msgs.append(f"Missing config section '{section}'")
             missing = True

    # Validate specific values only if the basic structure (required keys) is present
    if not missing:
        try: # Wrap value checks in try-except for robustness against unexpected types
            if not (0.0 <= config.data.overlap_factor < 1.0): error_msgs.append(f"data.overlap_factor ({config.data.overlap_factor}) must be in the range [0.0, 1.0)"); missing = True
            if config.logging.preview_batch_interval < 0: config.logging.preview_batch_interval = 0 # Allow disabling with 0
            if config.logging.preview_refresh_rate < 0: config.logging.preview_refresh_rate = 0 # Allow disabling with 0
            if config.saving.keep_last_checkpoints < 0: error_msgs.append(f"saving.keep_last_checkpoints ({config.saving.keep_last_checkpoints}) cannot be negative."); missing = True

            lr_check_value = getattr(config.training, 'lr', None)
            if not isinstance(lr_check_value, (int, float)) or lr_check_value <= 0: error_msgs.append(f"training.lr must be a positive number. Got: {lr_check_value!r} (type: {type(lr_check_value)})"); missing = True

            if config.training.lambda_lpips < 0: error_msgs.append(f"training.lambda_lpips ({config.training.lambda_lpips}) must be non-negative."); missing = True
            if config.training.iterations_per_epoch <= 0: error_msgs.append(f"training.iterations_per_epoch ({config.training.iterations_per_epoch}) must be positive."); missing = True
            if config.training.batch_size <= 0: error_msgs.append(f"training.batch_size ({config.training.batch_size}) must be positive."); missing = True
            if config.training.num_workers < 0: error_msgs.append(f"training.num_workers ({config.training.num_workers}) cannot be negative."); missing = True
            if config.training.num_workers > 0 and config.training.prefetch_factor < 1: error_msgs.append(f"training.prefetch_factor ({config.training.prefetch_factor}) must be at least 1 when num_workers > 0."); missing = True

            downsample_factor = 16 # For 4 downsampling layers (2^4)
            if config.data.resolution <= 0 or config.data.resolution % downsample_factor != 0:
                 error_msgs.append(f"data.resolution ({config.data.resolution}) should be positive and divisible by {downsample_factor} for the UNet architecture.")
                 missing = True # Treat as error

            if config.training.loss not in ['l1', 'l1+lpips']: error_msgs.append(f"training.loss must be 'l1' or 'l1+lpips', got '{config.training.loss}'"); missing = True

            # Check dataloader augmentation format if section exists
            if hasattr(config, 'dataloader') and hasattr(config.dataloader, 'datasets'):
                datasets_cfg = config.dataloader.datasets
                for key in ['src_augs', 'dst_augs', 'shared_augs']:
                     if hasattr(datasets_cfg, key):
                          aug_list = getattr(datasets_cfg, key)
                          if aug_list is None: # Check if key exists but is None
                               # This is acceptable, means empty list
                               pass
                          elif not isinstance(aug_list, list):
                               error_msgs.append(f"'dataloader.datasets.{key}' must be a list or None in the config YAML. Found type: {type(aug_list)}"); missing = True
                          else: # If it's a list, check items
                              for idx, item in enumerate(aug_list):
                                  # Augmentation items are converted to SimpleNamespace by dict_to_namespace
                                  if not isinstance(item, SimpleNamespace) or not hasattr(item, '_target_'):
                                       error_msgs.append(f"Invalid augmentation item at index {idx} in 'dataloader.datasets.{key}'. Must be a mapping with '_target_'. Found: {item!r}"); missing = True
                                       break # Stop checking this list after first error
        except AttributeError as e:
            error_msgs.append(f"Error accessing configuration value during validation: {e}. Check config structure.")
            missing = True
        except Exception as e:
             error_msgs.append(f"Unexpected error during configuration value validation: {e}")
             missing = True


    if missing:
        print("="*40 + "\nERROR: Invalid or missing configuration detected:")
        # Print only unique error messages
        unique_errors = sorted(list(set(error_msgs)))
        for msg in unique_errors: print(f"  - {msg}")
        print("="*40)
        exit(1)
    else:
         print("Configuration validated successfully.")

    # --- DDP Check & Launch ---
    # (Keep the rest of the __main__ block as it was)
    if "LOCAL_RANK" not in os.environ:
         print("="*40 + "\nERROR: DDP environment variables (e.g., LOCAL_RANK) not found! \nPlease launch using 'torchrun' or 'torch.distributed.launch'. \nExample: torchrun --standalone --nproc_per_node=NUM_GPUS train.py --config path/to/user_config.yaml\n" + "="*40)
         exit(1)

    # --- Start Training ---
    train(config) # Pass the final validated config object

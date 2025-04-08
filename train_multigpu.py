import os
import argparse
import math
from glob import glob
from PIL import Image
import logging
import time
import random
import yaml
import copy
from types import SimpleNamespace
import itertools # For infinite dataloader cycle
import signal # For graceful shutdown signal handling

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

# --- DDP Setup ---
def setup_ddp():
    # ... (no changes) ...
    if not dist.is_initialized():
        if "NODE_RANK" not in os.environ: os.environ["MASTER_ADDR"] = "localhost"; os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        if "RANK" not in os.environ: os.environ["RANK"] = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
        if "WORLD_SIZE" not in os.environ: os.environ["WORLD_SIZE"] = os.environ.get("OMPI_COMM_WORLD_SIZE", "1")
        if "LOCAL_RANK" not in os.environ: os.environ["LOCAL_RANK"] = "0"
        try: dist.init_process_group(backend="nccl")
        except Exception as e: print(f"Error initializing DDP: {e}. Check DDP environment variables."); raise
    local_rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_rank)
    if get_rank() == 0: logging.info(f"DDP Initialized: Rank {get_rank()}/{get_world_size()} on device cuda:{local_rank}")

def cleanup_ddp():
    if dist.is_initialized(): dist.destroy_process_group()
def get_rank():
    if not dist.is_available() or not dist.is_initialized(): return 0
    return dist.get_rank()
def get_world_size():
    if not dist.is_available() or not dist.is_initialized(): return 1
    return dist.get_world_size()
def is_main_process():
    return get_rank() == 0

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        mid_channels = max(1, mid_channels)
        self.d = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self, x): return self.d(x)
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.m = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.m(x)
class Up(nn.Module):
    # ... (no changes) ...
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
        diffY = x2.size()[2] - x1.size()[2]; diffX = x2.size()[3] - x1.size()[3]
        if diffY != 0 or diffX != 0:
            x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class OutConv(nn.Module):
    # ... (no changes) ...
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x): return self.c(x)
class UNet(nn.Module):
    # ... (no changes) ...
    def __init__(self, n_ch=3, n_cls=3, hidden_size=64, bilinear=True):
        super().__init__()
        self.n_ch = n_ch; self.n_cls = n_cls; self.hidden_size = hidden_size; self.bilinear = bilinear
        h = hidden_size
        chs = {'enc1': h, 'enc2': h*2, 'enc3': h*4, 'enc4': h*8, 'bottle': h*16}
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
class ImagePairSlicingDataset(Dataset):
    def __init__(self, src_dir, dst_dir, resolution, overlap_factor=0.0, transform=None):
        self.src_dir=src_dir; self.dst_dir=dst_dir; self.resolution=resolution; self.transform=transform; self.slice_info=[]; self.overlap_factor=overlap_factor
        if src_dir is None or dst_dir is None: raise ValueError("src_dir and dst_dir must be provided.")
        if not os.path.isdir(src_dir): raise FileNotFoundError(f"Source directory not found: {src_dir}")
        if not os.path.isdir(dst_dir): raise FileNotFoundError(f"Destination directory not found: {dst_dir}")
        if not(0.0<=overlap_factor<1.0): raise ValueError("overlap_factor must be [0.0, 1.0)")
        ovp=int(resolution*overlap_factor); self.stride=max(1,resolution-ovp)
        src_files=sorted(glob(os.path.join(src_dir,'*.*')))
        if not src_files: raise FileNotFoundError(f"No source images found in {src_dir}")
        self.skipped_count=0; self.processed_files=0; self.total_slices_generated=0; self.skipped_paths=[]
        for src_path in src_files:
            bname=os.path.basename(src_path); dst_path=os.path.join(dst_dir,bname)
            if not os.path.exists(dst_path): self.skipped_count+=1; self.skipped_paths.append((src_path,"Dst Missing")); continue
            try:
                with Image.open(src_path) as img: w,h=img.size
                if w<resolution or h<resolution: self.skipped_count+=1; self.skipped_paths.append((src_path,"Too Small")); continue
                n_s=0; py=list(range(0,h-resolution,self.stride))+[h-resolution]; px=list(range(0,w-resolution,self.stride))+[w-resolution]
                uy=sorted(list(set(py))); ux=sorted(list(set(px)))
                for y in uy:
                    for x in ux: c=(x,y,x+resolution,y+resolution); self.slice_info.append((src_path,dst_path,c)); n_s+=1
                if n_s>0: self.processed_files+=1; self.total_slices_generated+=n_s
            except Exception as e: self.skipped_count+=1; self.skipped_paths.append((src_path,f"Error:{e}")); continue
    def __len__(self): return len(self.slice_info)
    def __getitem__(self,idx):
        src_p,dst_p,c=self.slice_info[idx]
        try:
            src_i=Image.open(src_p).convert('RGB'); dst_i=Image.open(dst_p).convert('RGB')
            src_s=src_i.crop(c); dst_s=dst_i.crop(c)
            if src_s.size!=(self.resolution,self.resolution): src_s=src_s.resize((self.resolution,self.resolution),Image.Resampling.LANCZOS)
            if dst_s.size!=(self.resolution,self.resolution): dst_s=dst_s.resize((self.resolution,self.resolution),Image.Resampling.LANCZOS)
            if self.transform: src_s=self.transform(src_s); dst_s=self.transform(dst_s)
            return src_s,dst_s
        except Exception as e: raise RuntimeError(f"Error loading/processing slice idx {idx} ({src_p},{c})") from e

NORM_MEAN = [0.5, 0.5, 0.5]; NORM_STD = [0.5, 0.5, 0.5]
def denormalize(tensor):
    mean = torch.tensor(NORM_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)

def cycle(iterable):
    while True:
        for x in iterable: yield x

def save_previews(model, fixed_src_batch, fixed_dst_batch, config, logical_epoch, global_step, device, preview_save_count):
    # ... (no changes) ...
    if not is_main_process() or fixed_src_batch is None or fixed_dst_batch is None: return
    num_grid_cols = 3;
    if fixed_src_batch.size(0) < num_grid_cols: return
    src_select = fixed_src_batch[:num_grid_cols].cpu(); dst_select = fixed_dst_batch[:num_grid_cols].cpu()
    model.eval(); device_type = device.type
    with torch.no_grad(), autocast(device_type=device_type, enabled=config.training.use_amp):
        src_dev = src_select.to(device);
        predicted_batch = model.module(src_dev)
    model.train()
    pred_select = predicted_batch.cpu().float()
    src_denorm = denormalize(src_select); pred_denorm = denormalize(pred_select); dst_denorm = denormalize(dst_select)
    combined_interleaved = [item for i in range(num_grid_cols) for item in [src_denorm[i], dst_denorm[i], pred_denorm[i]]]
    if not combined_interleaved: return
    grid_tensor = torch.stack(combined_interleaved)
    grid = make_grid(grid_tensor, nrow=num_grid_cols, padding=2, normalize=False)
    img_pil = T.functional.to_pil_image(grid)
    preview_filename = os.path.join(config.data.output_dir, "training_preview.jpg")
    preview_refresh_rate = config.logging.preview_refresh_rate
    try:
        img_pil.save(preview_filename, "JPEG", quality=95)
        if preview_save_count == 0 or (preview_refresh_rate > 0 and preview_save_count % preview_refresh_rate == 0):
             logging.info(f"Saved preview (Ep {logical_epoch}, Step {global_step}, Save #{preview_save_count}) to {preview_filename}")
    except Exception as e: logging.error(f"Failed to save preview image: {e}")

def capture_preview_batch(config, transform):
    if not is_main_process(): return None, None
    num_preview_samples = 3; logging.info(f"Refreshing fixed batch ({num_preview_samples} samples) for previews...")
    try:
        preview_dataset = ImagePairSlicingDataset(
            config.data.src_dir, config.data.dst_dir, config.data.resolution,
            overlap_factor=config.data.overlap_factor, transform=transform
        )
        num_samples_to_load = min(num_preview_samples, len(preview_dataset))
        if num_samples_to_load < num_preview_samples and is_main_process():
             logging.warning(f"Dataset has only {len(preview_dataset)} slices, loading {num_samples_to_load} for preview.")
        if num_samples_to_load > 0:
            preview_loader = DataLoader(preview_dataset, batch_size=num_samples_to_load, shuffle=True, num_workers=0)
            fixed_src_slices, fixed_dst_slices = next(iter(preview_loader))
            logging.info(f"Captured new batch of size {fixed_src_slices.size(0)} for previews.");
            return fixed_src_slices.cpu(), fixed_dst_slices.cpu()
        else:
            logging.warning(f"Dataset has 0 valid slices. Cannot refresh preview.")
            return None, None
    except StopIteration: logging.error("Preview DataLoader yielded no batches."); return None, None
    except Exception as e: logging.exception(f"Error capturing preview batch: {e}"); return None, None

# --- Signal Handler for Graceful Shutdown ---
shutdown_requested = False
def handle_signal(signum, frame):
    global shutdown_requested
    if not shutdown_requested:
        print(f"\nReceived signal {signum}. Requesting graceful shutdown...")
        logging.warning(f"Received signal {signum}. Requesting graceful shutdown...")
        shutdown_requested = True
    else:
        print("Shutdown already requested. Terminating forcefully.")
        logging.warning("Shutdown already requested. Terminating forcefully.")
        exit(1) # Force exit on second signal


# --- Training Function (Modified Loop and Checkpointing) ---
def train(config):
    global shutdown_requested # Allow modification of the global flag

    # Register signal handlers for graceful shutdown (SIGINT: Ctrl+C, SIGTERM)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # --- DDP Setup ---
    setup_ddp(); rank = get_rank(); world_size = get_world_size()
    device = torch.device(f"cuda:{rank}"); device_type = device.type

    # --- Logging Setup ---
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
        except Exception: logging.info(f"Config: {config}") # Fallback
        logging.info("="*50)
        logging.info(">>> Press Ctrl+C to request graceful shutdown (saves checkpoint). <<<")

    # --- Dataset & DataLoader ---
    transform = T.Compose([ T.ToTensor(), T.Normalize(mean=NORM_MEAN, std=NORM_STD) ])
    dataset = None
    try:
        dataset = ImagePairSlicingDataset(
            config.data.src_dir, config.data.dst_dir, config.data.resolution,
            overlap_factor=config.data.overlap_factor, transform=transform
        )
        if is_main_process():
             logging.info(f"Dataset: Res={dataset.resolution}, Overlap={dataset.overlap_factor:.2f}, Stride={dataset.stride}")
             if dataset.skipped_count > 0: logging.warning(f"Skipped {dataset.skipped_count} pairs during dataset init.")
             num_slices = dataset.total_slices_generated; num_files = dataset.processed_files
             avg_slices = num_slices / num_files if num_files > 0 else 0
             logging.info(f"Found {num_slices} slices from {num_files} pairs (avg {avg_slices:.1f}).")
             if len(dataset) == 0: raise ValueError("Dataset created but contains 0 slices.")
             logging.info(f"Total dataset slices: {len(dataset)}")
             est_batches_per_pass = math.ceil(len(dataset) / (config.training.batch_size * world_size))
             logging.info(f"Est. batches per dataset pass: {est_batches_per_pass}")
             logging.info(f"Logical Epoch = {config.training.iterations_per_epoch} iterations (for sampler reset).")
             if hasattr(config, 'dataloader') and hasattr(config.dataloader, 'datasets'):
                 logging.info("Ignoring 'dataloader.datasets' augmentation settings in config for now.")

    except (FileNotFoundError, ValueError, Exception) as e:
        logging.error(f"Dataset initialization failed: {e}", exc_info=True)
        if dist.is_initialized(): cleanup_ddp(); return

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, sampler=sampler,
                            num_workers=config.dataloader.num_workers, pin_memory=True, drop_last=False,
                            prefetch_factor=config.dataloader.prefetch_factor if config.dataloader.num_workers > 0 else None)
    dataloader_iter = cycle(dataloader)

    # --- Model Size Calculation ---
    effective_model_size = config.model.model_size_dims
    use_lpips = False; loss_fn_lpips = None; default_hidden_size = 64
    # ... (same logic as before to determine effective_model_size and init LPIPS) ...
    if config.training.loss == 'l1+lpips':
        use_lpips = True
        if config.model.model_size_dims == default_hidden_size: effective_model_size = 96
        if is_main_process(): logging.info(f"Effective UNet hidden size: {effective_model_size}, LPIPS lambda={config.training.lambda_lpips}.")
        if is_main_process():
            try:
                loss_fn_lpips = lpips.LPIPS(net='vgg').to(device).eval()
                for param in loss_fn_lpips.parameters(): param.requires_grad = False
                logging.info("LPIPS VGG loss initialized.")
            except Exception as e: logging.error(f"Failed LPIPS init: {e}. Disabling LPIPS.", exc_info=True); use_lpips = False
    else:
        if is_main_process(): logging.info(f"L1 loss only: Effective UNet hidden size = {effective_model_size}.")


    # --- Instantiate Model, Optimizer, Scaler ---
    model = UNet(n_ch=3, n_cls=3, hidden_size=effective_model_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=1e-5)
    scaler = GradScaler(enabled=config.training.use_amp)
    criterion_l1 = nn.L1Loss()

    # --- Resume Logic (Load step and epoch) ---
    start_epoch = 0 # Logical start epoch
    start_step = 0  # Global start step (iteration)
    latest_checkpoint_path = os.path.join(config.data.output_dir, 'tunet_latest.pth')
    resume_flag = [os.path.exists(latest_checkpoint_path)] if is_main_process() else [False]
    if world_size > 1: dist.broadcast_object_list(resume_flag, src=0)

    if resume_flag[0]:
        try:
            if is_main_process(): logging.info(f"Attempting resume from checkpoint: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            # --- Compatibility Check (as before) ---
            # ... (same compatibility check logic comparing checkpoint config/args to current config) ...
            saved_config_data = checkpoint.get('config'); saved_args_data = checkpoint.get('args')
            ckpt_loss = None; ckpt_base_model_dims = None; ckpt_effective_hidden_size = None
            if saved_config_data:
                ckpt_config_sns = dict_to_sns(saved_config_data); ckpt_loss = ckpt_config_sns.training.loss; ckpt_base_model_dims = ckpt_config_sns.model.model_size_dims
            elif saved_args_data:
                ckpt_loss = getattr(saved_args_data, 'loss', 'l1'); ckpt_base_model_dims = getattr(saved_args_data, 'model_size_dims', getattr(saved_args_data, 'unet_hidden_size', default_hidden_size))
            else: ckpt_loss = config.training.loss; ckpt_base_model_dims = config.model.model_size_dims # Assume current
            ckpt_effective_hidden_size = ckpt_base_model_dims
            if ckpt_loss == 'l1+lpips' and ckpt_base_model_dims == default_hidden_size: ckpt_effective_hidden_size = 96
            current_loss = config.training.loss
            if ckpt_loss != current_loss: raise ValueError(f"Loss mismatch: Ckpt '{ckpt_loss}', Current '{current_loss}'")
            if ckpt_effective_hidden_size != effective_model_size: raise ValueError(f"Effective size mismatch: Ckpt {ckpt_effective_hidden_size}, Current {effective_model_size}")
            if is_main_process(): logging.info("Checkpoint configuration compatible.")
            # --- Load State Dicts ---
            state_dict = checkpoint['model_state_dict']
            if all(key.startswith('module.') for key in state_dict): state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if config.training.use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None: scaler.load_state_dict(checkpoint['scaler_state_dict'])
            # --- Load Epoch and Step ---
            start_epoch = checkpoint.get('epoch', 0); start_step = checkpoint.get('global_step', 0)
            if is_main_process(): logging.info(f"Resumed from logical epoch {start_epoch}, global step {start_step}.")
            del checkpoint; torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Failed load/resume checkpoint: {e}", exc_info=True)
            logging.warning("Starting training from scratch."); start_epoch = 0; start_step = 0
    else:
        if is_main_process(): logging.info(f"No checkpoint found. Starting training from scratch.")

    # --- Wrap model with DDP ---
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # --- Training Loop Variables ---
    global_step = start_step
    current_epoch = start_epoch # Tracks the current logical epoch
    iterations_per_epoch = config.training.iterations_per_epoch

    fixed_src_slices = fixed_dst_slices = None; preview_save_count = 0
    epoch_l1_loss_accum = 0.0; epoch_lpips_loss_accum = 0.0; epoch_step_count = 0
    batch_iter_start_time = time.time()

    if is_main_process():
         logging.info(f"Starting training loop (runs indefinitely).")
         logging.info(f"Iterations per Logical Epoch (Sampler Reset): {iterations_per_epoch}")
         logging.info(f"Starting from Global Step: {global_step}, Logical Epoch: {current_epoch}")


    # --- Main Training Loop (Infinite) ---
    model.train()
    try:
        while True:
            if shutdown_requested:
                if is_main_process():
                    logging.info(f"Shutdown requested at step {global_step}. Exiting training loop.")
                break

            is_new_epoch_start = (global_step == start_step) or (global_step % iterations_per_epoch == 0)
            if is_new_epoch_start:
                current_epoch = global_step // iterations_per_epoch
                sampler.set_epoch(current_epoch)
                if is_main_process():
                    logging.info(f"--- Starting Logical Epoch {current_epoch + 1} (Step {global_step}) ---")
                epoch_l1_loss_accum = 0.0
                epoch_lpips_loss_accum = 0.0
                epoch_step_count = 0
                batch_iter_start_time = time.time()

            try:
                batch_data = next(dataloader_iter)
                src_slices, dst_slices = batch_data
            except StopIteration:
                logging.error("Infinite dataloader stopped unexpectedly. Re-creating.")
                dataloader_iter = cycle(dataloader)
                continue
            except Exception as e:
                if is_main_process():
                    logging.error(f"Step {global_step}: Batch load error: {e}, skipping.", exc_info=True)
                global_step += 1
                continue

            iter_data_end_time = time.time()
            src_slices = src_slices.to(device, non_blocking=True)
            dst_slices = dst_slices.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device_type, enabled=config.training.use_amp):
                outputs = model(src_slices)
                l1_loss = criterion_l1(outputs, dst_slices)
                lpips_loss = torch.tensor(0.0, device=device)
                if use_lpips and loss_fn_lpips is not None:
                    try:
                        lpips_loss = loss_fn_lpips(outputs, dst_slices).mean()
                    except Exception as e:
                        if is_main_process() and global_step % (config.logging.log_interval * 10) == 1:
                            logging.warning(f"LPIPS calc failed step {global_step}: {e}. Setting loss to 0.")
                        lpips_loss = torch.tensor(0.0, device=device)
                loss = l1_loss + config.training.lambda_lpips * lpips_loss

            if torch.isnan(loss) or torch.isinf(loss):
                if is_main_process():
                    logging.error(f"Step {global_step}: NaN/Inf loss! Skipping step.")
                global_step += 1
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            iter_compute_end_time = time.time()

            dist.all_reduce(l1_loss, op=dist.ReduceOp.AVG)
            if use_lpips:
                dist.all_reduce(lpips_loss, op=dist.ReduceOp.AVG)
            batch_l1_loss = l1_loss.item()
            batch_lpips_loss = lpips_loss.item()
            epoch_l1_loss_accum += batch_l1_loss
            epoch_lpips_loss_accum += batch_lpips_loss
            epoch_step_count += 1

            global_step += 1
            current_logical_epoch_display = global_step // iterations_per_epoch + 1  # 1-based

            if global_step % config.logging.log_interval == 0 and is_main_process():
                current_lr = optimizer.param_groups[0]['lr']
                avg_epoch_l1 = epoch_l1_loss_accum / epoch_step_count if epoch_step_count > 0 else 0.0
                avg_epoch_lpips = epoch_lpips_loss_accum / epoch_step_count if use_lpips and epoch_step_count > 0 else 0.0
                data_time = iter_data_end_time - batch_iter_start_time
                compute_time = iter_compute_end_time - iter_data_end_time
                total_batch_time = time.time() - batch_iter_start_time
                steps_in_epoch = global_step % iterations_per_epoch if global_step % iterations_per_epoch != 0 else iterations_per_epoch

                log_msg = (f'Epoch[{current_logical_epoch_display}], Step[{global_step}] ({steps_in_epoch}/{iterations_per_epoch}), '
                           f'L1:{batch_l1_loss:.4f}(Avg:{avg_epoch_l1:.4f})')
                if use_lpips:
                    log_msg += (f', LPIPS:{batch_lpips_loss:.4f}(Avg:{avg_epoch_lpips:.4f})')
                log_msg += (f', LR:{current_lr:.1e}, T/B:{total_batch_time:.3f}s (D:{data_time:.3f},C:{compute_time:.3f})')
                logging.info(log_msg)
                batch_iter_start_time = time.time()

            preview_interval = config.logging.preview_batch_interval
            if preview_interval > 0 and global_step % preview_interval == 0 and is_main_process():
                refresh_preview = (fixed_src_slices is None) or (config.logging.preview_refresh_rate > 0 and preview_save_count % config.logging.preview_refresh_rate == 0)
                if refresh_preview:
                    new_src, new_dst = capture_preview_batch(config, transform)
                    if new_src is not None and new_dst is not None:
                        fixed_src_slices, fixed_dst_slices = new_src, new_dst
                if fixed_src_slices is not None and fixed_dst_slices is not None:
                    save_previews(model, fixed_src_slices, fixed_dst_slices, config, current_logical_epoch_display, global_step, device, preview_save_count)
                    preview_save_count += 1
                else:
                    logging.warning(f"Step {global_step}: Skipping preview (fixed batch None).")

            # --- Periodic Checkpointing ---
            save_interval = config.saving.save_iterations_interval
            save_interval_now = (save_interval > 0 and global_step % save_interval == 0)
            if is_main_process():
                ckpt_epoch_num = global_step // iterations_per_epoch
                checkpoint_path = os.path.join(config.data.output_dir, f'tunet_epoch_{ckpt_epoch_num:09d}.pth')
                latest_path = os.path.join(config.data.output_dir, 'tunet_latest.pth')
                config_dict = config_to_dict(config)
                checkpoint_data = {
                    'epoch': ckpt_epoch_num,
                    'global_step': global_step,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if config.training.use_amp else None,
                    'config': config_dict
                }
                try:
                    save_epoch_now = is_new_epoch_start and global_step > 0
                    if save_epoch_now:
                        torch.save(checkpoint_data, checkpoint_path)
                        logging.info(f"Epoch checkpoint saved: {checkpoint_path} (Step {global_step})")
                        # Always update the latest checkpoint at the beginning of a new epoch.
                        torch.save(checkpoint_data, latest_path)
                        logging.info(f"Updated latest checkpoint: {latest_path} (Step {global_step}, Epoch {ckpt_epoch_num})")
                    elif save_interval_now or shutdown_requested:
                        torch.save(checkpoint_data, latest_path)
                        if save_interval_now:
                            logging.info(f"Updated latest checkpoint: {latest_path} (Step {global_step})")
                        elif shutdown_requested:
                            logging.info(f"Shutdown checkpoint saved: {latest_path} (Step {global_step})")

    # After saving the checkpoint, prune old checkpoints.
                    if hasattr(config.saving, 'keep_last_checkpoints') and config.saving.keep_last_checkpoints > 0:
                        prune_checkpoints(config.data.output_dir, config.saving.keep_last_checkpoints)

                except Exception as e:
                    logging.error(f"Failed save checkpoint step {global_step}: {e}", exc_info=True)

            if world_size > 1 and (global_step % iterations_per_epoch == 0):
                dist.barrier()

    except KeyboardInterrupt:
        if is_main_process():
            logging.warning("KeyboardInterrupt received directly. Attempting graceful exit.")
        shutdown_requested = True
    except Exception as train_loop_error:
        logging.error("Error occurred during training loop:", exc_info=True)
        if is_main_process():
            logging.info("Attempting to save final state due to error...")
        shutdown_requested = True
    finally:
        if shutdown_requested and is_main_process():
            logging.info("Performing final checkpoint save...")
            try:
                ckpt_epoch_num = global_step // iterations_per_epoch
                latest_path = os.path.join(config.data.output_dir, 'tunet_latest.pth')
                config_dict = config_to_dict(config)
                checkpoint_data = {
                    'epoch': ckpt_epoch_num,
                    'global_step': global_step,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict() if config.training.use_amp else None,
                    'config': config_dict
                }
                torch.save(checkpoint_data, latest_path)
                logging.info(f"Final latest checkpoint saved: {latest_path} (Step {global_step})")
            except Exception as e:
                logging.error(f"Failed to save final checkpoint: {e}", exc_info=True)

        if is_main_process():
            logging.info(f"Training loop finished/terminated at step {global_step}.")
        cleanup_ddp()


# --- Config Helper Functions (config_to_dict, dict_to_sns, merge_configs) ---
def config_to_dict(sns):
    if isinstance(sns, SimpleNamespace): return {k: config_to_dict(v) for k, v in sns.__dict__.items()}
    elif isinstance(sns, (list, tuple)): return [config_to_dict(item) for item in sns]
    else: return sns
def dict_to_sns(d):
    if isinstance(d, dict):
        for key, value in d.items(): d[key] = dict_to_sns(value)
        return SimpleNamespace(**d)
    elif isinstance(d, (list, tuple)): return type(d)(dict_to_sns(item) for item in d)
    else: return d
def merge_configs(base, user):
    merged = copy.deepcopy(base)
    for key, value in user.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else: merged[key] = value
    return merged

# --- Checkpoint Pruning Helper ---

import os

def prune_checkpoints(output_dir, keep_last):
    import glob as glob_module  # Import the module under a different name
    ckpt_files = sorted(
        glob_module.glob(os.path.join(output_dir, 'tunet_epoch_*.pth')),
        key=os.path.getmtime
    )
    if len(ckpt_files) <= keep_last:
        return
    files_to_remove = ckpt_files[:-keep_last]
    for file in files_to_remove:
        try:
            os.remove(file)
            logging.info(f"Pruned old checkpoint: {file}")
        except Exception as e:
            logging.warning(f"Failed to prune checkpoint {file}: {e}")



# --- Main Execution (`if __name__ == "__main__":`) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train UNet indefinitely via YAML config using DDP')
    parser.add_argument('--config', type=str, required=True, help='Path to the USER YAML configuration file')
    cli_args = parser.parse_args()

    # Basic logging for config loading
    if os.environ.get("RANK", "0") == "0":
         logging.basicConfig(level=logging.INFO, format='%(asctime)s [CONFIG] %(message)s')

    # --- Load Base Config ---
    script_dir = os.path.dirname(os.path.abspath(__file__)) # Use abspath for reliability
    base_config_path = os.path.join(script_dir, 'base', 'base.yaml')
    user_config_path = cli_args.config
    try:
        with open(base_config_path, 'r') as f: base_config_dict = yaml.safe_load(f)
        if base_config_dict is None: base_config_dict = {}
        if is_main_process(): logging.info(f"Loaded base config: {base_config_path}")
    except Exception as e: print(f"ERROR loading base config {base_config_path}: {e}"); exit(1)

    # --- Load User Config ---
    try:
        with open(user_config_path, 'r') as f: user_config_dict = yaml.safe_load(f)
        if user_config_dict is None: user_config_dict = {}
        if is_main_process(): logging.info(f"Loaded user config: {user_config_path}")
    except Exception as e: print(f"ERROR loading user config {user_config_path}: {e}"); exit(1)

    # --- Merge & Convert ---
    merged_config_dict = merge_configs(base_config_dict, user_config_dict)
    if is_main_process():
         print("-" * 20 + " Merged Config Dictionary " + "-" * 20)
         try: print(yaml.dump(merged_config_dict, indent=2, default_flow_style=False))
         except Exception as dump_error: print(f"Could not dump: {dump_error}\n{merged_config_dict}")
         print("-" * 60)
    config = dict_to_sns(merged_config_dict)

    # --- Final Configuration Validation (Removed max_epochs check) ---
    required_paths = {'data': ['src_dir', 'dst_dir', 'output_dir']}
    missing = []
    # ... (path validation as before) ...
    for section, keys in required_paths.items():
        if not hasattr(config, section): missing.extend([f"{section}.{key}" for key in keys]); continue
        sec_obj = getattr(config, section)
        for key in keys:
            if not hasattr(sec_obj, key) or getattr(sec_obj, key) is None: missing.append(f"{section}.{key}")
    if missing: print("\nERROR: Required paths missing/null:", ", ".join(missing)); exit(1)

    # --- Specific Value Validations ---
    try:
        training_cfg = getattr(config, 'training')
        data_cfg = getattr(config, 'data')
        model_cfg = getattr(config, 'model')
        logging_cfg = getattr(config, 'logging', SimpleNamespace())
        saving_cfg = getattr(config, 'saving', SimpleNamespace())
        dataloader_cfg = getattr(config, 'dataloader', SimpleNamespace())

        # --- !! VALIDATION UPDATES !! ---
        # REMOVED max_epochs check
        if getattr(training_cfg, 'iterations_per_epoch', 0) <= 0:
             raise ValueError(f"config.training.iterations_per_epoch ({getattr(training_cfg, 'iterations_per_epoch', 'MISSING')}) must be > 0")
        # Check save interval (must be non-negative)
        save_interval_val = getattr(saving_cfg, 'save_iterations_interval', 0)
        if save_interval_val < 0:
             raise ValueError(f"config.saving.save_iterations_interval ({save_interval_val}) must be >= 0")
        # --- End Validation Updates ---

        if not (0.0 <= getattr(data_cfg, 'overlap_factor', -1.0) < 1.0):
             raise ValueError(f"config.data.overlap_factor ({getattr(data_cfg, 'overlap_factor', 'MISSING')}) must be [0.0, 1.0)")
        if getattr(model_cfg, 'model_size_dims', 0) <= 0:
             raise ValueError(f"config.model.model_size_dims ({getattr(model_cfg, 'model_size_dims', 'MISSING')}) must be > 0")
        if getattr(training_cfg, 'batch_size', 0) <= 0:
             raise ValueError(f"config.training.batch_size ({getattr(training_cfg, 'batch_size', 'MISSING')}) must be > 0")
        if getattr(training_cfg, 'loss', '') == 'l1+lpips' and getattr(training_cfg, 'lambda_lpips', 0) < 0:
            print(f"WARNING: lambda_lpips negative. Using abs().")
            config.training.lambda_lpips = abs(training_cfg.lambda_lpips)

        # Sanitize intervals
        config.logging.log_interval = max(1, getattr(logging_cfg, 'log_interval', 50))
        config.logging.preview_batch_interval = max(0, getattr(logging_cfg, 'preview_batch_interval', 500))
        config.logging.preview_refresh_rate = max(0, getattr(logging_cfg, 'preview_refresh_rate', 5))
        # Save interval already checked >= 0
        config.saving.save_iterations_interval = max(0, save_interval_val) # Ensure it's stored
        config.dataloader.num_workers = max(0, getattr(dataloader_cfg, 'num_workers', 4))
        config.dataloader.prefetch_factor = max(2, getattr(dataloader_cfg, 'prefetch_factor', 2))

    except (AttributeError, ValueError, TypeError) as e:
         print(f"\nERROR in configuration validation: {e}")
         print("Check YAML structure/values.")
         if is_main_process():
             print("\n" + "-"*10 + " Final Config (before error) " + "-"*10)
             try: print(yaml.dump(config_to_dict(config), indent=2))
             except Exception: print(config)
             print("-" * 50)
         exit(1)

    # Check DDP Env Vars
    if "LOCAL_RANK" not in os.environ:
         print("="*40 + "\n WARNING: DDP env vars not found! Use 'torchrun'. Forcing single process." + "\n" + "="*40)
         os.environ["RANK"] = "0"; os.environ["WORLD_SIZE"] = "1"; os.environ["LOCAL_RANK"] = "0"
         if os.environ.get("RANK") == "0":
              logging.basicConfig(level=logging.INFO, format='%(asctime)s [CONFIG] %(message)s', force=True)

    # Start training
    train(config)

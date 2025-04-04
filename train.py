import os
import argparse
import math
from glob import glob
from PIL import Image
import logging
import time
import random

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
import lpips # <-- Import lpips

# --- DDP Setup ---
def setup_ddp():
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
def is_main_process(): return get_rank() == 0

# --- Corrected UNet Model Components ---
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
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
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.m = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.m(x)

class Up(nn.Module):
    """Upscaling then double conv"""
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x): return self.c(x)

# --- Corrected UNet Model ---
class UNet(nn.Module):
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

# --- Dataset ---
class ImagePairSlicingDataset(Dataset):
    def __init__(self, src_dir, dst_dir, resolution, overlap_factor=0.0, transform=None):
        self.src_dir=src_dir; self.dst_dir=dst_dir; self.resolution=resolution; self.transform=transform; self.slice_info=[]; self.overlap_factor=overlap_factor
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

# --- Helper Functions ---
NORM_MEAN = [0.5, 0.5, 0.5]; NORM_STD = [0.5, 0.5, 0.5]
def denormalize(tensor):
    mean = torch.tensor(NORM_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)
def save_previews(model, fixed_src_batch, fixed_dst_batch, output_dir, epoch, device, preview_save_count, preview_refresh_rate):
    if not is_main_process() or fixed_src_batch is None or fixed_dst_batch is None: return
    num_grid_cols = 3;
    if fixed_src_batch.size(0) < num_grid_cols: return
    src_select = fixed_src_batch[:num_grid_cols].cpu(); dst_select = fixed_dst_batch[:num_grid_cols].cpu()
    model.eval(); device_type = device.type
    with torch.no_grad(), autocast(device_type=device_type, enabled=torch.is_autocast_enabled()):
        src_dev = src_select.to(device); predicted_batch = model.module(src_dev)
    model.train()
    pred_select = predicted_batch.cpu().float()
    src_denorm = denormalize(src_select); pred_denorm = denormalize(pred_select); dst_denorm = denormalize(dst_select)
    combined_interleaved = [item for i in range(num_grid_cols) for item in [src_denorm[i], dst_denorm[i], pred_denorm[i]]]
    if not combined_interleaved: return
    grid_tensor = torch.stack(combined_interleaved)
    grid = make_grid(grid_tensor, nrow=num_grid_cols, padding=2, normalize=False)
    img_pil = T.functional.to_pil_image(grid)
    preview_filename = os.path.join(output_dir, "training_preview.jpg")
    try:
        img_pil.save(preview_filename, "JPEG", quality=95)
        if preview_save_count == 0 or (preview_refresh_rate > 0 and preview_save_count % preview_refresh_rate == 0):
             logging.info(f"Saved 3x3 training preview to {preview_filename} (Epoch {epoch+1}, Save #{preview_save_count})")
    except Exception as e: logging.error(f"Failed to save preview image: {e}")

def capture_preview_batch(args, transform):
    if not is_main_process(): return None, None
    num_preview_samples = 3; logging.info(f"Refreshing fixed batch ({num_preview_samples} samples) for previews...")
    try:
        preview_dataset = ImagePairSlicingDataset(args.src_dir, args.dst_dir, args.resolution, overlap_factor=args.overlap_factor, transform=transform)
        if len(preview_dataset) >= num_preview_samples:
            preview_loader = DataLoader(preview_dataset, batch_size=num_preview_samples, shuffle=True, num_workers=0)
            fixed_src_slices, fixed_dst_slices = next(iter(preview_loader))
            if fixed_src_slices.size(0) == num_preview_samples:
                logging.info(f"Captured new batch of size {num_preview_samples} for previews."); 
                return fixed_src_slices.cpu(), fixed_dst_slices.cpu()
            else:
                logging.warning(f"Preview DataLoader returned {fixed_src_slices.size(0)} samples instead of {num_preview_samples}.")
                return None, None
        else:
            logging.warning(f"Dataset has only {len(preview_dataset)} slices, need {num_preview_samples}. Cannot refresh preview.")
            return None, None
    except StopIteration:
        logging.error("Preview DataLoader yielded no batches during refresh."); 
        return None, None
    except Exception as e:
        logging.exception(f"Error capturing preview batch: {e}");
        return None, None

# --- Training Function ---
def train(args):
    # --- DDP Setup ---
    setup_ddp(); rank = get_rank(); world_size = get_world_size()
    device = torch.device(f"cuda:{rank}"); device_type = device.type

    # --- Logging Setup ---
    log_level = logging.INFO if is_main_process() else logging.WARNING
    log_format = f'%(asctime)s [RK{rank}][%(levelname)s] %(message)s'
    handlers = [logging.StreamHandler()]
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = os.path.join(args.output_dir, 'training.log')
        handlers.append(logging.FileHandler(log_file, mode='a'))
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers, force=True)

    if is_main_process():
        logging.info("="*50)
        logging.info("Starting training run...")
        logging.info(f"Args: {vars(args)}")

    # --- Dataset & DataLoader ---
    transform = T.Compose([ T.ToTensor(), T.Normalize(mean=NORM_MEAN, std=NORM_STD) ])
    dataset = None
    try:
        dataset = ImagePairSlicingDataset(args.src_dir, args.dst_dir, args.resolution, overlap_factor=args.overlap_factor, transform=transform)
        if is_main_process():
             logging.info(f"Dataset: Res={dataset.resolution}, Overlap={dataset.overlap_factor}, Stride={dataset.stride}")
             if dataset.skipped_count > 0: logging.warning(f"Skipped {dataset.skipped_count} pairs during dataset init.")
             if dataset.processed_files > 0:
                 logging.info(f"Found {dataset.total_slices_generated} slices from {dataset.processed_files} pairs (avg {dataset.total_slices_generated/dataset.processed_files:.1f}).")
             else:
                 raise ValueError("Dataset processed 0 valid image pairs.")
             if len(dataset) == 0: raise ValueError("Dataset created but contains 0 slices.")
    except (FileNotFoundError, ValueError, Exception) as e:
        logging.error(f"Dataset initialization failed: {e}", exc_info=not isinstance(e, (FileNotFoundError, ValueError)))
        if dist.is_initialized(): cleanup_ddp(); return

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True, prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None)
    if is_main_process(): logging.info(f"DataLoader created. Steps per epoch: {len(dataloader)}")

    # --- Model Configuration (Loss Mode) ---
    unet_hidden_size = args.unet_hidden_size
    use_lpips = False
    loss_fn_lpips = None

    if args.loss == 'l1+lpips':
        use_lpips = True
        if args.unet_hidden_size == 64:
            unet_hidden_size = 96
            if is_main_process():
                logging.info(f"Using l1+lpips loss: Increasing UNet hidden size -> {unet_hidden_size}.")
        if is_main_process():
            logging.info(f"Using l1+lpips loss: UNet hidden={unet_hidden_size}, LPIPS lambda={args.lambda_lpips}.")
            try:
                loss_fn_lpips = lpips.LPIPS(net='vgg').to(device).eval()
                for param in loss_fn_lpips.parameters():
                    param.requires_grad = False
                logging.info("LPIPS VGG loss initialized.")
            except Exception as e:
                logging.error(f"Failed to init LPIPS: {e}. Disabling LPIPS.", exc_info=True)
                use_lpips = False
    else:
        if is_main_process():
            logging.info(f"Using l1 loss only: UNet hidden={unet_hidden_size}.")

    # --- Instantiate Model, Optimizer, Scaler ---
    model = UNet(n_ch=3, n_cls=3, hidden_size=unet_hidden_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = GradScaler(enabled=args.use_amp)
    criterion_l1 = nn.L1Loss()

    # --- Resume Logic ---
    start_epoch = 0
    latest_checkpoint_path = os.path.join(args.output_dir, 'unet_latest.pth')
    resume_flag = [os.path.exists(latest_checkpoint_path)] if is_main_process() else [False]
    if world_size > 1: dist.broadcast_object_list(resume_flag, src=0)

    if resume_flag[0]:
        try:
            if is_main_process():
                logging.info(f"Attempting resume from: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, map_location=device)
            ckpt_args = checkpoint.get('args')
            if ckpt_args is not None:
                 ckpt_loss = getattr(ckpt_args, 'loss', None)
                 if ckpt_loss is None:
                     ckpt_quality = getattr(ckpt_args, 'quality', 'LQ')
                     ckpt_loss = 'l1+lpips' if ckpt_quality == 'HQ' else 'l1'
                 ckpt_default_hidden = 64
                 ckpt_effective_hidden_size = getattr(ckpt_args, 'unet_hidden_size', ckpt_default_hidden)
                 if ckpt_loss == 'l1+lpips' and ckpt_effective_hidden_size == ckpt_default_hidden:
                     ckpt_effective_hidden_size = 96
                 if ckpt_loss != args.loss:
                     raise ValueError(f"Loss mode mismatch: ckpt='{ckpt_loss}', requested='{args.loss}'")
                 if ckpt_effective_hidden_size != unet_hidden_size:
                     raise ValueError(f"Hidden size mismatch: ckpt effective={ckpt_effective_hidden_size}, requested={unet_hidden_size}")
                 if is_main_process(): logging.info("Checkpoint args compatible.")
            else:
                logging.warning("Ckpt missing 'args', cannot verify compatibility.")

            state_dict = checkpoint['model_state_dict']
            if all(key.startswith('module.') for key in state_dict):
                state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if args.use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch']
            if is_main_process():
                logging.info(f"Resumed from epoch {start_epoch}. Training starts from epoch {start_epoch + 1}.")
            del checkpoint; torch.cuda.empty_cache()
        except Exception as e:
            logging.error(f"Failed load/resume: {e}", exc_info=True)
            logging.warning("Starting from scratch."); start_epoch = 0
    else:
        if is_main_process():
            logging.info(f"No checkpoint found at {latest_checkpoint_path}. Starting from scratch.")

    # --- Wrap model with DDP ---
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # --- Preview State ---
    fixed_src_slices = fixed_dst_slices = None; preview_save_count = 0

    # --- Training Loop ---
    if is_main_process():
        logging.info(f"Starting loop: epoch {start_epoch + 1} -> {args.epochs}...")
    if start_epoch >= args.epochs and is_main_process():
        logging.warning("Start epoch >= total epochs. No training needed.")

    for epoch in range(start_epoch, args.epochs):
        model.train(); sampler.set_epoch(epoch)
        epoch_l1_loss = 0.0; epoch_lpips_loss = 0.0; batch_iter_start_time = time.time()
        num_batches = len(dataloader)

        for i, batch_data in enumerate(dataloader):
            if batch_data is None:
                 if is_main_process():
                     logging.warning(f"Epc[{epoch+1}], Bch[{i+1}]: None data, skipping.")
                 continue
            try:
                src_slices, dst_slices = batch_data
            except Exception as e:
                if is_main_process():
                    logging.error(f"Epc[{epoch+1}], Bch[{i+1}]: Batch error: {e}, skipping.")
                continue

            iter_data_end_time = time.time()
            src_slices = src_slices.to(device, non_blocking=True)
            dst_slices = dst_slices.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device_type, enabled=args.use_amp):
                outputs = model(src_slices)
                l1_loss = criterion_l1(outputs, dst_slices)
                lpips_loss = torch.tensor(0.0, device=device)
                if use_lpips and loss_fn_lpips is not None:
                    try:
                        lpips_loss = loss_fn_lpips(outputs, dst_slices).mean()
                    except Exception as e:
                         if is_main_process() and i % (args.log_interval * 10) == 0:
                              logging.warning(f"LPIPS calc failed: {e}. Skip LPIPS this batch.")
                         lpips_loss = torch.tensor(0.0, device=device)

                loss = l1_loss + args.lambda_lpips * lpips_loss

            if torch.isnan(loss):
                if is_main_process():
                    logging.error(f"Epc[{epoch+1}], Bch[{i+1}]: NaN loss! L1={l1_loss.item():.4f}, LPIPS={lpips_loss.item():.4f}. Skip update.")
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
            epoch_l1_loss += batch_l1_loss
            epoch_lpips_loss += batch_lpips_loss

            if (i + 1) % args.log_interval == 0 and is_main_process():
                batches_processed = i+1
                current_lr = optimizer.param_groups[0]['lr']
                avg_epoch_l1 = epoch_l1_loss / batches_processed
                avg_epoch_lpips = epoch_lpips_loss / batches_processed if use_lpips else 0.0
                data_time = iter_data_end_time - batch_iter_start_time
                compute_time = iter_compute_end_time - iter_data_end_time
                total_batch_time = time.time() - batch_iter_start_time

                log_msg = (f'Epc[{epoch+1}/{args.epochs}], Bch[{batches_processed}/{num_batches}], L1:{batch_l1_loss:.4f}(Avg:{avg_epoch_l1:.4f})')
                if use_lpips:
                    log_msg += (f', LPIPS:{batch_lpips_loss:.4f}(Avg:{avg_epoch_lpips:.4f})')
                log_msg += (f', LR:{current_lr:.1e}, T/Bch:{total_batch_time:.3f}s (D:{data_time:.3f},C:{compute_time:.3f})')
                logging.info(log_msg)
                batch_iter_start_time = time.time()

        avg_epoch_l1 = epoch_l1_loss / num_batches if num_batches > 0 else 0.0
        avg_epoch_lpips = epoch_lpips_loss / num_batches if num_batches > 0 and use_lpips else 0.0
        if is_main_process():
             log_msg = f"Epoch {epoch+1} finished. Avg Loss(L1): {avg_epoch_l1:.4f}"
             if use_lpips:
                 log_msg += f", Avg Loss(LPIPS): {avg_epoch_lpips:.4f}"
             logging.info(log_msg)

             if args.preview_interval > 0 and (epoch + 1) % args.preview_interval == 0:
                 refresh_preview = (fixed_src_slices is None) or (args.preview_refresh_rate > 0 and preview_save_count % args.preview_refresh_rate == 0)
                 if refresh_preview:
                     new_src, new_dst = capture_preview_batch(args, transform)
                     if new_src is not None and new_dst is not None:
                         fixed_src_slices, fixed_dst_slices = new_src, new_dst
                 if fixed_src_slices is not None:
                     save_previews(model, fixed_src_slices, fixed_dst_slices, args.output_dir, epoch, device, preview_save_count, args.preview_refresh_rate)
                     preview_save_count += 1
                 else:
                     logging.warning(f"Epc {epoch+1}: Skipping preview (no batch).")

             if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
                 checkpoint_path = os.path.join(args.output_dir, f'unet_epoch_{epoch+1}.pth')
                 checkpoint = {
                     'epoch': epoch+1,
                     'model_state_dict': model.module.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scaler_state_dict': scaler.state_dict() if args.use_amp else None,
                     'args': args
                 }
                 torch.save(checkpoint, checkpoint_path)
                 logging.info(f"Checkpoint saved: {checkpoint_path}")
                 latest_path = os.path.join(args.output_dir, 'unet_latest.pth')
                 torch.save(checkpoint, latest_path)

        if world_size > 1: dist.barrier()

    if is_main_process():
        logging.info("Training loop finished.")
        cleanup_ddp()

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train UNet with DDP, Loss Options, Overlap, Previews, and Resume')
    # Data paths
    parser.add_argument('--src_dir', type=str, required=True, help='Directory containing source images')
    parser.add_argument('--dst_dir', type=str, required=True, help='Directory containing target/modified images')
    parser.add_argument('--output_dir', type=str, default='./unet_output', help='Directory for checkpoints, logs, previews')
    # Model/Training params
    # Removed --quality argument; now using --loss
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'l1+lpips'],
                        help='Loss type to use: "l1" for L1 loss only, "l1+lpips" for combined L1 and LPIPS loss')
    parser.add_argument('--unet_hidden_size', type=int, default=64, help='Base UNet channels (default=64 for l1, auto-bumped for l1+lpips if default)')
    parser.add_argument('--resolution', type=int, default=512, choices=[512, 1024], help='Target resolution for image slices')
    parser.add_argument('--overlap_factor', type=float, default=0.25, help='Slice overlap factor (0.0 to <1.0). Default 0.25')
    parser.add_argument('--epochs', type=int, default=100, help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size *per GPU*')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda_lpips', type=float, default=1.0, help='Weight for LPIPS loss in l1+lpips mode')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers per GPU')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Dataloader prefetch factor (per worker)')
    parser.add_argument('--use_amp', action='store_true', help='Enable Automatic Mixed Precision (AMP)')
    # Logging/Saving/Preview
    parser.add_argument('--log_interval', type=int, default=50, help='Log training status every N batches')
    parser.add_argument('--save_interval', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--preview_interval', type=int, default=1, help='Save 3x3 preview image every N epochs (0 to disable)')
    parser.add_argument('--preview_refresh_rate', type=int, default=5, help='Refresh preview images every N preview saves (0 to disable refresh)')

    args = parser.parse_args()
    # Validation
    if not (0.0 <= args.overlap_factor < 1.0):
        print(f"ERROR: overlap_factor must be [0.0, 1.0)")
        exit(1)
    if args.preview_interval < 0: args.preview_interval = 0
    if args.preview_refresh_rate < 0: args.preview_refresh_rate = 0
    if args.lambda_lpips < 0:
        print(f"WARNING: lambda_lpips should be non-negative.")
        args.lambda_lpips = max(0, args.lambda_lpips)

    if "LOCAL_RANK" not in os.environ:
         print("="*40 + "\n ERROR: DDP environment variables not found! \nPlease launch this script using 'torchrun'. \n" + "="*40)
    else:
         train(args)

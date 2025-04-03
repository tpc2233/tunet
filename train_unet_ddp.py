import os
import argparse
import math
from glob import glob
from PIL import Image # No longer need ImageDraw, ImageFont
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
# Use updated AMP imports
from torch.amp import GradScaler, autocast

# --- DDP Setup ---
# (Same as before)
# ... (setup_ddp, cleanup_ddp, get_rank, get_world_size, is_main_process) ...
def setup_ddp():
    if not dist.is_initialized():
        if "NODE_RANK" not in os.environ: os.environ["MASTER_ADDR"] = "localhost"; os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
        if "RANK" not in os.environ: os.environ["RANK"] = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
        if "WORLD_SIZE" not in os.environ: os.environ["WORLD_SIZE"] = os.environ.get("OMPI_COMM_WORLD_SIZE", "1")
        if "LOCAL_RANK" not in os.environ: os.environ["LOCAL_RANK"] = "0"
        try: dist.init_process_group(backend="nccl")
        except Exception as e: print(f"Error initializing DDP: {e}"); raise
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

# --- UNet Model Components ---
# (Same as before: DoubleConv, Down, Up, OutConv)
# ... (DoubleConv, Down, Up, OutConv definitions) ...
class DoubleConv(nn.Module):
    def __init__(self, i, o, m=None): super().__init__(); m=m or o; self.d=nn.Sequential(nn.Conv2d(i, m, 3, 1, 1, bias=False), nn.BatchNorm2d(m), nn.ReLU(True), nn.Conv2d(m, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
    def forward(self, x): return self.d(x)
class Down(nn.Module):
    def __init__(self, i, o): super().__init__(); self.m=nn.Sequential(nn.MaxPool2d(2), DoubleConv(i, o))
    def forward(self, x): return self.m(x)
class Up(nn.Module):
    def __init__(self, i, o, b=True): super().__init__(); self.b=b; self.up=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if b else nn.ConvTranspose2d(i, i//2, kernel_size=2, stride=2); self.conv=DoubleConv(i,o,i//2 if b else o)
    def forward(self, x1, x2): x1=self.up(x1); dY=x2.size(2)-x1.size(2); dX=x2.size(3)-x1.size(3); x1=nn.functional.pad(x1,[dX//2,dX-dX//2,dY//2,dY-dY//2]); x=torch.cat([x2,x1],dim=1); return self.conv(x)
class OutConv(nn.Module):
    def __init__(self, i, o): super().__init__(); self.c=nn.Conv2d(i, o, 1)
    def forward(self, x): return self.c(x)

# --- UNet Model ---
# (Same as before)
# ... (UNet class definition) ...
class UNet(nn.Module):
    def __init__(self, n_ch=3, n_cls=3, bil=True): super().__init__(); self.nch=n_ch; self.ncls=n_cls; self.bil=bil; self.inc=DoubleConv(n_ch,64); self.d1=Down(64,128); self.d2=Down(128,256); self.d3=Down(256,512); f=2 if bil else 1; self.d4=Down(512,1024//f); self.u1=Up(1024,512//f,bil); self.u2=Up(512,256//f,bil); self.u3=Up(256,128//f,bil); self.u4=Up(128,64,bil); self.outc=OutConv(64,n_cls)
    def forward(self, x): x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3); x5=self.d4(x4); x=self.u1(x5,x4); x=self.u2(x,x3); x=self.u3(x,x2); x=self.u4(x,x1); return self.outc(x)

# --- Dataset ---
# (Same as before)
# ... (ImagePairSlicingDataset definition) ...
class ImagePairSlicingDataset(Dataset):
    def __init__(self, src_dir, dst_dir, resolution, overlap_factor=0.0, transform=None):
        self.src_dir = src_dir; self.dst_dir = dst_dir; self.resolution = resolution; self.transform = transform
        self.slice_info = []
        if not (0.0 <= overlap_factor < 1.0): raise ValueError("overlap_factor must be between 0.0 and 1.0")
        self.overlap_factor = overlap_factor
        overlap_pixels = int(resolution * overlap_factor); self.stride = max(1, resolution - overlap_pixels)
        if is_main_process(): logging.info(f"Dataset: Res={resolution}, Overlap={overlap_factor}, Stride={self.stride}"); start_time = time.time()
        src_files = sorted(glob(os.path.join(src_dir, '*.*')))
        if not src_files: raise FileNotFoundError(f"No source images found in {src_dir}")
        skipped_count = processed_files = total_slices_generated = 0
        for src_path in src_files:
            basename = os.path.basename(src_path); dst_path = os.path.join(dst_dir, basename)
            if not os.path.exists(dst_path):
                if is_main_process(): logging.warning(f"Dst not found for {src_path}, skipping."); skipped_count += 1; continue
            try:
                with Image.open(src_path) as img: width, height = img.size
                if width < resolution or height < resolution:
                    if is_main_process(): logging.warning(f"Image {src_path} ({width}x{height}) smaller than resolution ({resolution}x{resolution}), skipping."); skipped_count += 1; continue
                n_slices_img = 0
                possible_y = list(range(0, height - resolution, self.stride)) + [height - resolution]
                possible_x = list(range(0, width - resolution, self.stride)) + [width - resolution]
                unique_y = sorted(list(set(possible_y))); unique_x = sorted(list(set(possible_x)))
                for y in unique_y:
                    for x in unique_x:
                        coords = (x, y, x + resolution, y + resolution)
                        self.slice_info.append((src_path, dst_path, coords)); n_slices_img += 1
                if n_slices_img > 0: processed_files += 1; total_slices_generated += n_slices_img
            except Exception as e:
                 if is_main_process(): logging.error(f"Error processing {src_path} or {dst_path}: {e}. Skipping."); skipped_count += 1
        if is_main_process():
            end_time = time.time(); logging.info(f"Dataset creation took {end_time - start_time:.2f} seconds.")
            if skipped_count > 0: logging.warning(f"Skipped {skipped_count} image pairs/files.")
        if not self.slice_info: raise ValueError(f"No valid image slices found. Check dirs, integrity, resolution ({resolution}), overlap.")
        if is_main_process():
            avg_slices = total_slices_generated / processed_files if processed_files > 0 else 0
            logging.info(f"Found {total_slices_generated} slices from {processed_files} pairs (avg {avg_slices:.1f} slices/image).")
    def __len__(self): return len(self.slice_info)
    def __getitem__(self, idx):
        src_path, dst_path, coords = self.slice_info[idx]
        try:
            src_img = Image.open(src_path).convert('RGB'); dst_img = Image.open(dst_path).convert('RGB')
            src_slice = src_img.crop(coords); dst_slice = dst_img.crop(coords)
            if src_slice.size != (self.resolution, self.resolution): src_slice = src_slice.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
            if dst_slice.size != (self.resolution, self.resolution): dst_slice = dst_slice.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
            if self.transform: src_slice = self.transform(src_slice); dst_slice = self.transform(dst_slice)
            return src_slice, dst_slice
        except Exception as e:
            logging.error(f"Error in __getitem__ idx {idx} ({src_path}, {dst_path}, {coords}): {e}"); raise e


# --- Helper Functions ---
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

def denormalize(tensor):
    mean = torch.tensor(NORM_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)

# --- MODIFIED save_previews function ---
def save_previews(model, fixed_src_batch, fixed_dst_batch, output_dir, epoch, device):
    """Generates and saves a 3x3 preview grid (Src | Dst | Pred)."""
    if not is_main_process(): return
    if fixed_src_batch is None or fixed_dst_batch is None: return

    num_grid_cols = 3 # Hardcoded for 3x3 grid

    # Check if we have enough samples in the captured batch
    if fixed_src_batch.size(0) < num_grid_cols:
        logging.warning(f"Need at least {num_grid_cols} samples for preview grid, but only got {fixed_src_batch.size(0)}. Skipping preview.")
        return

    # Select exactly num_grid_cols samples
    src_select = fixed_src_batch[:num_grid_cols].cpu()
    dst_select = fixed_dst_batch[:num_grid_cols].cpu()

    model.eval()
    device_type = device.type
    with torch.no_grad(), autocast(device_type=device_type, enabled=torch.is_autocast_enabled()):
        # Run inference only on the selected samples
        src_dev = src_select.to(device)
        predicted_batch = model.module(src_dev)
    model.train()

    pred_select = predicted_batch.cpu().float()

    # Denormalize selected batches
    src_denorm = denormalize(src_select)
    pred_denorm = denormalize(pred_select)
    dst_denorm = denormalize(dst_select)

    # --- Interleave for grid layout [S1, D1, P1, S2, D2, P2, S3, D3, P3] ---
    combined_interleaved = []
    for i in range(num_grid_cols):
        combined_interleaved.append(src_denorm[i]) # Row 1: Source
        combined_interleaved.append(dst_denorm[i]) # Row 2: Target/Destination
        combined_interleaved.append(pred_denorm[i]) # Row 3: Predicted/Inference

    # Stack the interleaved tensors
    if not combined_interleaved: # Should not happen if check above passed
        return
    grid_tensor = torch.stack(combined_interleaved)

    # Create the grid, nrow determines the number of columns
    grid = make_grid(grid_tensor, nrow=num_grid_cols, padding=2, normalize=False)
    img_pil = T.functional.to_pil_image(grid)

    # --- Save the image ---
    preview_filename = os.path.join(output_dir, "training_preview.jpg")
    try:
        img_pil.save(preview_filename, "JPEG", quality=95)
        if (epoch + 1) % (args.preview_interval * 5) == 0 or epoch == 0:
             logging.info(f"Saved 3x3 training preview image to {preview_filename} (Epoch {epoch+1})")
    except Exception as e:
        logging.error(f"Failed to save preview image: {e}")


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
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers, force=True)

    if is_main_process(): logging.info("Starting training..."); logging.info(f"Args: {vars(args)}")

    # --- Dataset and DataLoader ---
    transform = T.Compose([ T.ToTensor(), T.Normalize(mean=NORM_MEAN, std=NORM_STD) ])
    try:
        dataset = ImagePairSlicingDataset(args.src_dir, args.dst_dir, args.resolution,
                                          overlap_factor=args.overlap_factor, transform=transform)
    except (FileNotFoundError, ValueError, Exception) as e:
        logging.exception(f"Dataset initialization failed: {e}"); cleanup_ddp(); return
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True,
                            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None)

    # --- Model ---
    model = UNet(n_ch=3, n_cls=3).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # --- Loss, Optimizer, Scaler ---
    criterion = nn.L1Loss(); optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = GradScaler(enabled=args.use_amp)

    # --- Fixed Batch for Previews (Try to get exactly 3) ---
    fixed_src_slices = fixed_dst_slices = None
    num_preview_samples = 3 # Hardcode for 3x3 grid
    if is_main_process() and args.preview_interval > 0:
        logging.info(f"Capturing fixed batch ({num_preview_samples} samples) for previews...")
        try:
            preview_dataset = ImagePairSlicingDataset(args.src_dir, args.dst_dir, args.resolution,
                                                     overlap_factor=args.overlap_factor, transform=transform)
            if len(preview_dataset) >= num_preview_samples:
                 # Request exactly num_preview_samples
                preview_loader = DataLoader(preview_dataset, batch_size=num_preview_samples, shuffle=False, num_workers=0)
                fixed_src_slices, fixed_dst_slices = next(iter(preview_loader))
                # Check if we actually got the requested number
                if fixed_src_slices.size(0) == num_preview_samples:
                    fixed_src_slices = fixed_src_slices.cpu(); fixed_dst_slices = fixed_dst_slices.cpu()
                    logging.info(f"Captured batch of size {num_preview_samples} for previews.")
                else:
                    logging.warning(f"DataLoader returned {fixed_src_slices.size(0)} samples instead of {num_preview_samples}. Preview might be skipped.")
                    fixed_src_slices = None # Reset if batch size is wrong
            else:
                 logging.warning(f"Preview dataset has only {len(preview_dataset)} slices, need {num_preview_samples}. Skipping preview setup.")
        except StopIteration: logging.error("Preview DataLoader yielded no batches.")
        except Exception as e: logging.exception(f"Error capturing fixed batch: {e}")

    # --- Training Loop ---
    if is_main_process(): logging.info(f"Training for {args.epochs} epochs... Steps/epoch: {len(dataloader)}")
    for epoch in range(args.epochs):
        model.train(); sampler.set_epoch(epoch)
        epoch_loss = 0.0; batch_iter_start_time = time.time()
        for i, (src_slices, dst_slices) in enumerate(dataloader):
            iter_data_end_time = time.time()
            src_slices=src_slices.to(device, non_blocking=True); dst_slices=dst_slices.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device_type, enabled=args.use_amp):
                outputs = model(src_slices); loss = criterion(outputs, dst_slices)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            iter_compute_end_time = time.time()
            dist.all_reduce(loss, op=dist.ReduceOp.AVG); batch_loss = loss.item(); epoch_loss += batch_loss
            # Logging
            if (i + 1) % args.log_interval == 0 and is_main_process():
                batches_processed = i+1; current_lr = optimizer.param_groups[0]['lr']; avg_epoch_loss_so_far = epoch_loss/batches_processed
                data_time=iter_data_end_time-batch_iter_start_time; compute_time=iter_compute_end_time-iter_data_end_time; total_batch_time=time.time()-batch_iter_start_time
                logging.info(f'Epc[{epoch+1}/{args.epochs}], Bch[{batches_processed}/{len(dataloader)}], Loss:{batch_loss:.4f} (AvgEpc:{avg_epoch_loss_so_far:.4f}), LR:{current_lr:.1e}, Time/Bch:{total_batch_time:.3f}s (D:{data_time:.3f},C:{compute_time:.3f})')
                batch_iter_start_time = time.time()
        # End of Epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        if is_main_process():
             logging.info(f"Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}")
             # Previews
             if args.preview_interval > 0 and (epoch + 1) % args.preview_interval == 0:
                 if fixed_src_slices is not None: save_previews(model, fixed_src_slices, fixed_dst_slices, args.output_dir, epoch, device)
                 else: logging.warning(f"Epoch {epoch+1}: Skipping preview (no fixed batch).")
             # Checkpointing
             if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
                 checkpoint_path = os.path.join(args.output_dir, f'unet_epoch_{epoch+1}.pth')
                 checkpoint = {'epoch': epoch+1, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict() if args.use_amp else None, 'args': args}
                 torch.save(checkpoint, checkpoint_path); logging.info(f"Checkpoint saved: {checkpoint_path}")
                 latest_path = os.path.join(args.output_dir, 'unet_latest.pth'); torch.save(checkpoint, latest_path)
        if world_size > 1: dist.barrier()
    if is_main_process(): logging.info("Training finished."); cleanup_ddp()

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train UNet with DDP, Overlapping Slicing, and Previews')
    # --- REMOVED --num_previews ---
    # Data paths
    parser.add_argument('--src_dir', type=str, required=True, help='Directory containing source images')
    parser.add_argument('--dst_dir', type=str, required=True, help='Directory containing target/modified images')
    parser.add_argument('--output_dir', type=str, default='./unet_output', help='Directory for checkpoints, logs, previews')
    # Model/Training params
    parser.add_argument('--resolution', type=int, default=512, choices=[512, 1024], help='Target resolution for image slices')
    parser.add_argument('--overlap_factor', type=float, default=0.25, help='Slice overlap factor (0.0 to <1.0). Default 0.25')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size *per GPU*')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers per GPU')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Dataloader prefetch factor (per worker)')
    parser.add_argument('--use_amp', action='store_true', help='Enable Automatic Mixed Precision (AMP)')
    # Logging/Saving/Preview
    parser.add_argument('--log_interval', type=int, default=50, help='Log training status every N batches')
    parser.add_argument('--save_interval', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--preview_interval', type=int, default=1, help='Save 3x3 preview image every N epochs (0 to disable)')

    args = parser.parse_args()
    # Validation
    if not (0.0 <= args.overlap_factor < 1.0): print(f"ERROR: overlap_factor ({args.overlap_factor}) must be [0.0, 1.0)"); exit(1)
    if args.preview_interval < 0: args.preview_interval = 0 # Allow disabling

    # Launch Training
    if "LOCAL_RANK" not in os.environ:
         print("="*40 + "\n ERROR: DDP environment variables not found! \nPlease launch this script using 'torchrun'. \n" + "="*40)
    else:
        train(args)

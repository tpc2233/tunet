import os
import argparse
import math
from glob import glob
from PIL import Image
import logging
import time
import random # Added for potential future shuffling/sampling if needed

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
# (Same as before)
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
        # Logging is moved to the train function for clarity on rank 0
        # if is_main_process(): logging.info(f"Dataset: Res={resolution}, Overlap={overlap_factor}, Stride={self.stride}"); start_time = time.time()
        src_files = sorted(glob(os.path.join(src_dir, '*.*')))
        if not src_files: raise FileNotFoundError(f"No source images found in {src_dir}")
        self.skipped_count = 0
        self.processed_files = 0
        self.total_slices_generated = 0
        for src_path in src_files:
            basename = os.path.basename(src_path); dst_path = os.path.join(dst_dir, basename)
            if not os.path.exists(dst_path):
                # Logged later if is_main_process
                self.skipped_count += 1; continue
            try:
                with Image.open(src_path) as img: width, height = img.size
                if width < resolution or height < resolution:
                    # Logged later if is_main_process
                    self.skipped_count += 1; continue
                n_slices_img = 0
                possible_y = list(range(0, height - resolution, self.stride)) + [height - resolution]
                possible_x = list(range(0, width - resolution, self.stride)) + [width - resolution]
                unique_y = sorted(list(set(possible_y))); unique_x = sorted(list(set(possible_x)))
                for y in unique_y:
                    for x in unique_x:
                        coords = (x, y, x + resolution, y + resolution)
                        self.slice_info.append((src_path, dst_path, coords)); n_slices_img += 1
                if n_slices_img > 0: self.processed_files += 1; self.total_slices_generated += n_slices_img
            except Exception as e:
                 # Logged later if is_main_process, store path for logging
                 self.skipped_count += 1
                 # Optionally store the error message/path if needed for detailed logging
        # Final count logging moved to train function
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
            # Error logging happens in main loop or during initial checks now
            # logging.error(f"Error in __getitem__ idx {idx} ({src_path}, {dst_path}, {coords}): {e}");
            raise e # Re-raise to potentially stop training if too many errors


# --- Helper Functions ---
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

def denormalize(tensor):
    mean = torch.tensor(NORM_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)

# --- MODIFIED save_previews function ---
# (No changes needed here, it just uses the passed batches)
def save_previews(model, fixed_src_batch, fixed_dst_batch, output_dir, epoch, device, preview_save_count):
    """Generates and saves a 3x3 preview grid (Src | Dst | Pred)."""
    # preview_save_count added just for optional logging improvement
    if not is_main_process(): return
    if fixed_src_batch is None or fixed_dst_batch is None: return

    num_grid_cols = 3
    if fixed_src_batch.size(0) < num_grid_cols: return # Already checked during capture

    src_select = fixed_src_batch[:num_grid_cols].cpu()
    dst_select = fixed_dst_batch[:num_grid_cols].cpu()

    model.eval()
    device_type = device.type
    with torch.no_grad(), autocast(device_type=device_type, enabled=torch.is_autocast_enabled()):
        src_dev = src_select.to(device)
        predicted_batch = model.module(src_dev)
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
        # Log less frequently? Only on refresh or first few?
        if preview_save_count % args.preview_refresh_rate == 0 or preview_save_count <= 1:
             logging.info(f"Saved 3x3 training preview image to {preview_filename} (Epoch {epoch+1}, Save #{preview_save_count})")
    except Exception as e:
        logging.error(f"Failed to save preview image: {e}")

# --- NEW Helper Function to Capture Preview Batch ---
def capture_preview_batch(args, transform):
    """Attempts to load and return a batch of 3 samples for preview."""
    if not is_main_process(): return None, None

    num_preview_samples = 3
    logging.info(f"Refreshing fixed batch ({num_preview_samples} samples) for previews...")
    try:
        # Recreate dataset object - ensures it reflects current files if they change?
        # Or pass the main dataset object if that's preferred (might be cleaner)
        preview_dataset = ImagePairSlicingDataset(args.src_dir, args.dst_dir, args.resolution,
                                                 overlap_factor=args.overlap_factor, transform=transform)

        if len(preview_dataset) >= num_preview_samples:
            # Use shuffle=True to get random samples each time!
            preview_loader = DataLoader(preview_dataset, batch_size=num_preview_samples, shuffle=True, num_workers=0)
            fixed_src_slices, fixed_dst_slices = next(iter(preview_loader))

            if fixed_src_slices.size(0) == num_preview_samples:
                # Keep on CPU
                logging.info(f"Captured new batch of size {num_preview_samples} for previews.")
                return fixed_src_slices.cpu(), fixed_dst_slices.cpu()
            else:
                logging.warning(f"DataLoader returned {fixed_src_slices.size(0)} samples instead of {num_preview_samples}. Preview might be skipped.")
                return None, None
        else:
             logging.warning(f"Preview dataset has only {len(preview_dataset)} slices, need {num_preview_samples}. Cannot refresh preview batch.")
             return None, None
    except StopIteration:
        logging.error("Preview DataLoader yielded no batches during refresh.")
        return None, None
    except Exception as e:
        logging.exception(f"Error capturing preview batch during refresh: {e}")
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
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers, force=True)

    if is_main_process(): logging.info("Starting training..."); logging.info(f"Args: {vars(args)}")

    # --- Dataset and DataLoader ---
    transform = T.Compose([ T.ToTensor(), T.Normalize(mean=NORM_MEAN, std=NORM_STD) ])
    try:
        # Create the main dataset instance
        dataset = ImagePairSlicingDataset(args.src_dir, args.dst_dir, args.resolution,
                                          overlap_factor=args.overlap_factor, transform=transform)
        # Log dataset stats from rank 0 after creation
        if is_main_process():
             logging.info(f"Dataset: Res={dataset.resolution}, Overlap={dataset.overlap_factor}, Stride={dataset.stride}")
             if dataset.skipped_count > 0:
                 logging.warning(f"Skipped {dataset.skipped_count} image pairs/files during dataset init (check names, sizes, integrity).")
             if dataset.processed_files > 0:
                 avg_slices = dataset.total_slices_generated / dataset.processed_files
                 logging.info(f"Found {dataset.total_slices_generated} slices from {dataset.processed_files} pairs (avg {avg_slices:.1f} slices/image).")
             else:
                  logging.error("No image pairs processed successfully!")
                  # No point continuing if no data
                  raise ValueError("Dataset processed 0 valid image pairs.")

    except (FileNotFoundError, ValueError, Exception) as e:
        # Log full traceback if it's an unexpected error
        if not isinstance(e, (FileNotFoundError, ValueError)):
            logging.exception(f"Unexpected dataset initialization error: {e}")
        else:
             logging.error(f"Dataset initialization failed: {e}") # Log expected errors simply
        if dist.is_initialized(): cleanup_ddp()
        return

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True,
                            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None)

    # --- Model ---
    model = UNet(n_ch=3, n_cls=3).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # --- Loss, Optimizer, Scaler ---
    criterion = nn.L1Loss(); optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # Reverted GradScaler init
    scaler = GradScaler(enabled=args.use_amp)

    # --- Preview State (Rank 0 Only) ---
    fixed_src_slices = fixed_dst_slices = None
    preview_save_count = 0

    # --- Training Loop ---
    if is_main_process(): logging.info(f"Training for {args.epochs} epochs... Steps/epoch: {len(dataloader)}")
    for epoch in range(args.epochs):
        model.train(); sampler.set_epoch(epoch)
        epoch_loss = 0.0; batch_iter_start_time = time.time()
        for i, batch_data in enumerate(dataloader):
            # Basic check for data loading errors within the loop
            if batch_data is None:
                 if is_main_process(): logging.warning(f"Epoch {epoch+1}, Batch {i+1}: Received None from DataLoader, skipping batch.")
                 continue
            try:
                src_slices, dst_slices = batch_data
            except Exception as e:
                if is_main_process(): logging.error(f"Epoch {epoch+1}, Batch {i+1}: Error unpacking batch: {e}, skipping batch.")
                continue

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
        avg_epoch_loss = epoch_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        if is_main_process():
             logging.info(f"Epoch {epoch+1} finished. Avg Loss: {avg_epoch_loss:.4f}")

             # --- Preview Logic ---
             if args.preview_interval > 0 and (epoch + 1) % args.preview_interval == 0:
                 refresh_preview = False
                 # Check if refresh is needed (only if refresh rate > 0)
                 if args.preview_refresh_rate > 0:
                     if preview_save_count % args.preview_refresh_rate == 0:
                         refresh_preview = True
                 # Also refresh if batch is currently None (initial capture)
                 if fixed_src_slices is None:
                     refresh_preview = True

                 # Attempt to capture/recapture if needed
                 if refresh_preview:
                     new_src, new_dst = capture_preview_batch(args, transform)
                     # Only update if capture was successful
                     if new_src is not None and new_dst is not None:
                         fixed_src_slices = new_src
                         fixed_dst_slices = new_dst
                     else:
                         logging.warning(f"Epoch {epoch+1}: Failed to refresh preview batch.")
                         # Keep the old batch if refresh failed, unless it was None initially

                 # Save preview if we have a valid batch
                 if fixed_src_slices is not None:
                     save_previews(model, fixed_src_slices, fixed_dst_slices, args.output_dir, epoch, device, preview_save_count)
                     preview_save_count += 1 # Increment only after successful save attempt
                 else:
                     logging.warning(f"Epoch {epoch+1}: Skipping preview save (no valid batch available).")

             # --- Checkpointing ---
             if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
                 checkpoint_path = os.path.join(args.output_dir, f'unet_epoch_{epoch+1}.pth')
                 checkpoint = {'epoch': epoch+1, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict() if args.use_amp else None, 'args': args}
                 torch.save(checkpoint, checkpoint_path); logging.info(f"Checkpoint saved: {checkpoint_path}")
                 latest_path = os.path.join(args.output_dir, 'unet_latest.pth'); torch.save(checkpoint, latest_path)

        if world_size > 1: dist.barrier() # Sync all processes before next epoch

    if is_main_process(): logging.info("Training finished."); cleanup_ddp()

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train UNet with DDP, Overlapping Slicing, and Previews')
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
    # --- ADDED preview_refresh_rate ---
    parser.add_argument('--preview_refresh_rate', type=int, default=5, help='Refresh preview images every N preview saves (0 to disable refresh, keeps initial batch)')

    args = parser.parse_args()
    # Validation
    if not (0.0 <= args.overlap_factor < 1.0): print(f"ERROR: overlap_factor must be [0.0, 1.0)"); exit(1)
    if args.preview_interval < 0: args.preview_interval = 0
    if args.preview_refresh_rate < 0: args.preview_refresh_rate = 0 # Ensure non-negative

    # Launch Training
    if "LOCAL_RANK" not in os.environ:
         print("="*40 + "\n ERROR: DDP environment variables not found! \nPlease launch this script using 'torchrun'. \n" + "="*40)
    else:
        train(args)

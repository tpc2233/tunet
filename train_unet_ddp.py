import os
import argparse
import math
from glob import glob
from PIL import Image
import logging
import time # Added for timing dataset creation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

# --- DDP Setup ---
# (Same as before)
def setup_ddp():
    """Initializes the distributed process group."""
    if not dist.is_initialized():
        # Automatically set MASTER_ADDR and MASTER_PORT if using torchrun standalone
        if "NODE_RANK" not in os.environ: # Heuristic for standalone mode
            os.environ["MASTER_ADDR"] = "localhost"
            if "MASTER_PORT" not in os.environ:
                 os.environ["MASTER_PORT"] = "29500" # Default port

        # Ensure necessary env vars are set for init_process_group
        # These are usually set by the launcher (torchrun, slurm, etc.)
        if "RANK" not in os.environ: os.environ["RANK"] = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
        if "WORLD_SIZE" not in os.environ: os.environ["WORLD_SIZE"] = os.environ.get("OMPI_COMM_WORLD_SIZE", "1")
        # LOCAL_RANK is critical and set by torchrun/launch
        if "LOCAL_RANK" not in os.environ:
             # Fallback for single process/debug scenario? Risky for real DDP.
             os.environ["LOCAL_RANK"] = "0"


        try:
             dist.init_process_group(backend="nccl") # Use 'nccl' for NVIDIA GPUs
        except Exception as e:
            print(f"Error initializing DDP: {e}")
            print("Check environment variables (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, LOCAL_RANK)")
            raise

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if get_rank() == 0:
        logging.info(f"DDP Initialized: Rank {get_rank()}/{get_world_size()} on device cuda:{local_rank}")

def cleanup_ddp():
    """Destroys the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_rank():
    """Gets the rank of the current process."""
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    """Gets the total number of processes."""
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    """Checks if the current process is the main process (rank 0)."""
    return get_rank() == 0

# --- UNet Model Components ---
# (Same as before: DoubleConv, Down, Up, OutConv)
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
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
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

# --- UNet Model ---
# (Same as before)
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


# --- Dataset ---

class ImagePairSlicingDataset(Dataset):
    def __init__(self, src_dir, dst_dir, resolution, overlap_factor=0.0, transform=None):
        """
        Initializes the dataset with optional overlapping slices.

        Args:
            src_dir (str): Directory containing source images.
            dst_dir (str): Directory containing target images.
            resolution (int): The width and height of the slices.
            overlap_factor (float): Factor of overlap (0.0 to < 1.0).
                                    0.0 means no overlap (stride=resolution).
                                    0.5 means 50% overlap (stride=resolution/2).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.resolution = resolution
        self.transform = transform
        self.slice_info = [] # Stores tuples: (src_path, dst_path, slice_coords)

        # --- Calculate stride based on overlap ---
        if not (0.0 <= overlap_factor < 1.0):
             raise ValueError("overlap_factor must be between 0.0 (inclusive) and 1.0 (exclusive)")
        self.overlap_factor = overlap_factor
        overlap_pixels = int(resolution * overlap_factor)
        self.stride = max(1, resolution - overlap_pixels) # Ensure stride is at least 1

        if is_main_process():
            logging.info(f"Dataset: Resolution={resolution}, Overlap Factor={overlap_factor}, Stride={self.stride}")
            start_time = time.time()

        src_files = sorted(glob(os.path.join(src_dir, '*.*'))) # Basic matching, adjust if needed

        if not src_files:
             raise FileNotFoundError(f"No source images found in {src_dir}")

        skipped_count = 0
        processed_files = 0
        total_slices_generated = 0

        for src_path in src_files:
            basename = os.path.basename(src_path)
            dst_path = os.path.join(dst_dir, basename)

            if not os.path.exists(dst_path):
                if is_main_process():
                    logging.warning(f"Destination image not found for {src_path}, skipping.")
                skipped_count += 1
                continue

            try:
                # Get image dimensions
                with Image.open(src_path) as img:
                    width, height = img.size

                # Skip images smaller than the slice resolution
                if width < resolution or height < resolution:
                    if is_main_process():
                        logging.warning(f"Image {src_path} ({width}x{height}) is smaller than resolution "
                                        f"({resolution}x{resolution}), skipping.")
                    skipped_count += 1
                    continue

                # --- Generate Slice Coordinates with Overlap ---
                n_slices_img = 0
                possible_y = list(range(0, height - resolution, self.stride)) + [height - resolution]
                possible_x = list(range(0, width - resolution, self.stride)) + [width - resolution]
                # Use set to ensure unique coordinates if stride perfectly divides (width/height - resolution)
                unique_y = sorted(list(set(possible_y)))
                unique_x = sorted(list(set(possible_x)))

                for y in unique_y:
                    for x in unique_x:
                        coords = (x, y, x + resolution, y + resolution)
                        self.slice_info.append((src_path, dst_path, coords))
                        n_slices_img += 1

                if n_slices_img > 0:
                    processed_files += 1
                    total_slices_generated += n_slices_img
                # No else needed here because we already checked image size

            except Exception as e:
                 if is_main_process():
                    logging.error(f"Error processing {src_path} or {dst_path}: {e}. Skipping.")
                 skipped_count += 1

        if is_main_process():
            end_time = time.time()
            logging.info(f"Dataset creation took {end_time - start_time:.2f} seconds.")
            if skipped_count > 0:
                logging.warning(f"Skipped {skipped_count} image pairs/files due to missing files, size issues, or errors.")

        if not self.slice_info:
            raise ValueError(f"No valid image slices found. Check directories, image integrity, resolution ({resolution}), and overlap settings.")

        if is_main_process():
            avg_slices = total_slices_generated / processed_files if processed_files > 0 else 0
            logging.info(f"Found {total_slices_generated} total slices from {processed_files} image pairs (avg {avg_slices:.1f} slices/image).")


    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, idx):
        src_path, dst_path, coords = self.slice_info[idx]

        try:
            # Load images only when needed
            src_img = Image.open(src_path).convert('RGB')
            dst_img = Image.open(dst_path).convert('RGB')

            # Crop to the calculated slice coordinates
            src_slice = src_img.crop(coords)
            dst_slice = dst_img.crop(coords)

            # Ensure slices have the exact target resolution (should be guaranteed by logic above, but good safety check)
            if src_slice.size != (self.resolution, self.resolution):
                 # This case should ideally not happen with the current logic
                 logging.warning(f"Slice size mismatch for {src_path} at {coords}. Expected "
                                 f"{self.resolution}x{self.resolution}, got {src_slice.size}. Resizing.")
                 src_slice = src_slice.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
            if dst_slice.size != (self.resolution, self.resolution):
                 logging.warning(f"Slice size mismatch for {dst_path} at {coords}. Expected "
                                 f"{self.resolution}x{self.resolution}, got {dst_slice.size}. Resizing.")
                 dst_slice = dst_slice.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)


            if self.transform:
                src_slice = self.transform(src_slice)
                dst_slice = self.transform(dst_slice)

            return src_slice, dst_slice

        except Exception as e:
            # Provide more context in error message
            logging.error(f"Error in __getitem__ for index {idx} (src: {src_path}, dst: {dst_path}, coords: {coords}): {e}")
            # Depending on robustness needs, you might return None and handle it in collate_fn,
            # or return dummy data, or re-raise the exception. Re-raising is simplest for now.
            raise e

# --- Helper Functions ---
# (Same as before: NORM_MEAN, NORM_STD, denormalize, save_previews)
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

def denormalize(tensor):
    """Reverses the normalization applied by T.Normalize."""
    mean = torch.tensor(NORM_MEAN).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(NORM_STD).view(1, 3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

def save_previews(model, fixed_src_batch, fixed_dst_batch, output_dir, epoch, device, num_previews):
    """Generates and saves preview images."""
    if not is_main_process(): return
    if fixed_src_batch is None or fixed_dst_batch is None:
        # Log only once maybe? Or reduce frequency.
        # logging.warning("Fixed batch for previews not yet captured. Skipping preview generation.")
        return

    model.eval()
    with torch.no_grad(), autocast(enabled=torch.is_autocast_enabled()): # Use autocast if enabled during training
        src_dev = fixed_src_batch.to(device)
        predicted_batch = model.module(src_dev)
    model.train()

    src_cpu = fixed_src_batch.cpu()
    pred_cpu = predicted_batch.cpu().float() # Ensure float for denorm
    dst_cpu = fixed_dst_batch.cpu()

    src_denorm = denormalize(src_cpu)
    pred_denorm = denormalize(pred_cpu)
    dst_denorm = denormalize(dst_cpu)

    num_actual = min(num_previews, src_denorm.size(0))
    if num_actual == 0: return

    combined = torch.cat([src_denorm[:num_actual], pred_denorm[:num_actual], dst_denorm[:num_actual]], dim=0)
    grid = make_grid(combined, nrow=num_actual, padding=2, normalize=False)
    img_pil = T.functional.to_pil_image(grid)
    preview_filename = os.path.join(output_dir, "training_preview.jpg")
    try:
        img_pil.save(preview_filename, "JPEG")
        # Reduce logging frequency for previews? Maybe only log every N saves or on error.
        # logging.info(f"Saved training preview image to {preview_filename} (Epoch {epoch+1})")
    except Exception as e:
        logging.error(f"Failed to save preview image: {e}")


# --- Training Function ---

def train(args):
    # --- DDP Setup ---
    setup_ddp()
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{rank}")

    # --- Logging Setup (only main process creates file handler) ---
    log_level = logging.INFO if is_main_process() else logging.WARNING # Less verbose on non-main ranks
    log_format = f'%(asctime)s [RK{rank}][%(levelname)s] %(message)s'
    handlers = [logging.StreamHandler()]
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        log_file = os.path.join(args.output_dir, 'training.log')
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)

    if is_main_process():
        logging.info("Starting training script.")
        logging.info(f"Arguments: {vars(args)}")
        logging.info(f"Using {world_size} GPUs.")

    # --- Dataset and DataLoader ---
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    try:
        dataset = ImagePairSlicingDataset(args.src_dir, args.dst_dir, args.resolution,
                                          overlap_factor=args.overlap_factor, transform=transform)
    except (FileNotFoundError, ValueError, Exception) as e: # Catch broader exceptions during init
        logging.exception(f"Dataset initialization failed: {e}") # Use logging.exception to include traceback
        if dist.is_initialized(): cleanup_ddp()
        return # Exit if dataset fails

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    # Consider increasing num_workers if dataset I/O becomes a bottleneck, especially with more slices
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            sampler=sampler,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None) # Added prefetch

    # --- Model ---
    model = UNet(n_channels=3, n_classes=3).to(device)
    # If using SyncBatchNorm, convert before wrapping with DDP
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # --- Loss and Optimizer ---
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = GradScaler(enabled=args.use_amp)

    # --- Fixed Batch for Previews (Rank 0 Only) ---
    # (Same logic as before to capture first batch)
    fixed_src_slices = None
    fixed_dst_slices = None
    if is_main_process() and args.preview_interval > 0 and args.num_previews > 0:
        logging.info("Capturing fixed batch for previews...")
        try:
            # Create a separate dataloader instance *without shuffle* for consistency
            preview_dataset = ImagePairSlicingDataset(args.src_dir, args.dst_dir, args.resolution,
                                                     overlap_factor=args.overlap_factor, transform=transform)
            if len(preview_dataset) > 0:
                preview_loader = DataLoader(preview_dataset, batch_size=args.num_previews, shuffle=False, num_workers=0)
                data_iter = iter(preview_loader)
                fixed_src_slices, fixed_dst_slices = next(data_iter)
                fixed_src_slices = fixed_src_slices.cpu()
                fixed_dst_slices = fixed_dst_slices.cpu()
                logging.info(f"Captured batch of size {fixed_src_slices.size(0)} for previews.")
                # Ensure num_previews doesn't exceed the captured batch size
                args.num_previews = fixed_src_slices.size(0)
            else:
                 logging.warning("Preview dataset is empty, cannot capture fixed batch.")

        except StopIteration:
            logging.error("Preview DataLoader yielded no batches.")
        except Exception as e:
            logging.exception(f"Error capturing fixed batch: {e}")

    # --- Training Loop ---
    if is_main_process():
        logging.info(f"Starting training for {args.epochs} epochs...")
        logging.info(f"Total steps per epoch: {len(dataloader)}")

    for epoch in range(args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        batch_iter_start_time = time.time() # For batch timing

        for i, (src_slices, dst_slices) in enumerate(dataloader):
            iter_data_end_time = time.time() # Time spent loading data

            src_slices = src_slices.to(device, non_blocking=True)
            dst_slices = dst_slices.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=args.use_amp):
                outputs = model(src_slices)
                loss = criterion(outputs, dst_slices)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            iter_compute_end_time = time.time() # Time spent on forward/backward/step

            # --- Reduce loss across GPUs ---
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            batch_loss = loss.item()
            epoch_loss += batch_loss

            # --- Logging (Main Process Only) ---
            if (i + 1) % args.log_interval == 0 and is_main_process():
                batches_processed = i + 1
                current_lr = optimizer.param_groups[0]['lr'] # Get current LR
                avg_batch_loss_interval = epoch_loss / batches_processed # Avg loss so far in epoch
                data_time = iter_data_end_time - batch_iter_start_time
                compute_time = iter_compute_end_time - iter_data_end_time
                total_batch_time = time.time() - batch_iter_start_time

                logging.info(f'Epoch [{epoch+1}/{args.epochs}], Batch [{batches_processed}/{len(dataloader)}], '
                             f'Loss: {batch_loss:.4f} (Avg Epoch: {avg_batch_loss_interval:.4f}), '
                             f'LR: {current_lr:.1e}, '
                             f'Time/Batch: {total_batch_time:.3f}s (Data: {data_time:.3f}s, Compute: {compute_time:.3f}s)')
                # Reset timer for next interval
                batch_iter_start_time = time.time()

        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss / len(dataloader)
        if is_main_process():
             logging.info(f"Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")

             # --- Generate and Save Previews ---
             if args.preview_interval > 0 and (epoch + 1) % args.preview_interval == 0:
                 if fixed_src_slices is not None and fixed_dst_slices is not None:
                      save_previews(model, fixed_src_slices, fixed_dst_slices, args.output_dir, epoch, device, args.num_previews)
                 else:
                      logging.warning(f"Epoch {epoch+1}: Skipping preview generation as fixed batch is not available.")

             # --- Checkpointing ---
             if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs: # Save on last epoch too
                 checkpoint_path = os.path.join(args.output_dir, f'unet_epoch_{epoch+1}.pth')
                 checkpoint = {
                     'epoch': epoch + 1,
                     'model_state_dict': model.module.state_dict(),
                     'optimizer_state_dict': optimizer.state_dict(),
                     'scaler_state_dict': scaler.state_dict() if args.use_amp else None,
                     'args': args
                 }
                 torch.save(checkpoint, checkpoint_path)
                 logging.info(f"Checkpoint saved to {checkpoint_path}")

                 latest_path = os.path.join(args.output_dir, 'unet_latest.pth')
                 torch.save(checkpoint, latest_path)

        # Wait for all processes to finish epoch before starting next (esp. for checkpointing)
        if world_size > 1:
            dist.barrier()


    if is_main_process():
        logging.info("Training finished.")

    # --- Cleanup ---
    cleanup_ddp()


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train UNet with DDP, Overlapping Slicing, and Previews')

    # Data paths
    parser.add_argument('--src_dir', type=str, required=True, help='Directory containing source images')
    parser.add_argument('--dst_dir', type=str, required=True, help='Directory containing target/modified images')
    parser.add_argument('--output_dir', type=str, default='./unet_output', help='Directory for checkpoints, logs, previews')

    # Model/Training params
    parser.add_argument('--resolution', type=int, default=512, choices=[512, 1024], help='Target resolution for image slices')
    parser.add_argument('--overlap_factor', type=float, default=0.25, help='Slice overlap factor (0.0 to <1.0). Default 0.25 (25%% overlap)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size *per GPU*')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers per GPU')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='Dataloader prefetch factor (per worker)')
    parser.add_argument('--use_amp', action='store_true', help='Enable Automatic Mixed Precision (AMP)')

    # Logging/Saving/Preview
    parser.add_argument('--log_interval', type=int, default=50, help='Log training status every N batches')
    parser.add_argument('--save_interval', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--preview_interval', type=int, default=1, help='Save preview image every N epochs (0 to disable)')
    parser.add_argument('--num_previews', type=int, default=4, help='Number of image slices in the preview grid')

    args = parser.parse_args()

    # Validation
    if not (0.0 <= args.overlap_factor < 1.0):
        print(f"ERROR: overlap_factor ({args.overlap_factor}) must be between 0.0 and < 1.0")
        exit(1)
    if args.preview_interval > 0 and args.num_previews <= 0:
         print("WARNING: num_previews must be > 0 to generate previews. Disabling previews.")
         args.preview_interval = 0

    # --- Launch Training ---
    if "LOCAL_RANK" not in os.environ:
         print("="*40)
         print(" ERROR: DDP environment variables not found! ")
         print(" Please launch this script using 'torchrun'. ")
         print(" Example for 2 GPUs with 50% overlap:")
         print(" torchrun --standalone --nnodes=1 --nproc_per_node=2 train_unet_ddp.py \\")
         print("   --src_dir /path/to/src --dst_dir /path/to/dst --resolution 512 \\")
         print("   --overlap_factor 0.5 --batch_size 2 --epochs 50 --use_amp \\")
         print("   --preview_interval 1 --num_previews 4")
         print("="*40)
    else:
        # No need for barrier here, logging/dir creation handled by main process check inside train()
        train(args)

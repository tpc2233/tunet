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

# --- UNet Model ---
class UNet(nn.Module):
    def __init__(self, config, n_ch=3, n_cls=3, bilinear=True):
        super().__init__(); self.n_ch = n_ch; self.n_cls = n_cls;
        # <-- CHANGED: Use model_size_dims from config -->
        self.hidden_size = config.model.model_size_dims
        self.bilinear = bilinear
        h = self.hidden_size; chs = {'enc1': h, 'enc2': h*2, 'enc3': h*4, 'enc4': h*8, 'bottle': h*16}
        self.inc = DoubleConv(n_ch, chs['enc1']); self.down1 = Down(chs['enc1'], chs['enc2']); self.down2 = Down(chs['enc2'], chs['enc3']); self.down3 = Down(chs['enc3'], chs['enc4']); self.down4 = Down(chs['enc4'], chs['bottle'])
        self.up1 = Up(chs['bottle'], chs['enc4'], chs['enc4'], bilinear); self.up2 = Up(chs['enc4'], chs['enc3'], chs['enc3'], bilinear); self.up3 = Up(chs['enc3'], chs['enc2'], chs['enc2'], bilinear); self.up4 = Up(chs['enc2'], chs['enc1'], chs['enc1'], bilinear)
        self.outc = OutConv(chs['enc1'], n_cls)
    def forward(self, x):
        x1=self.inc(x); x2=self.down1(x1); x3=self.down2(x2); x4=self.down3(x3); x5=self.down4(x4)
        x=self.up1(x5, x4); x=self.up2(x, x3); x=self.up3(x, x2); x=self.up4(x, x1); return self.outc(x)

# --- Dataset ---
# 
class ImagePairSlicingDataset(Dataset):
    def __init__(self, src_dir, dst_dir, resolution, overlap_factor=0.0, transform=None):
        self.src_dir=src_dir; self.dst_dir=dst_dir; self.resolution=resolution; self.transform=transform; self.slice_info=[]; self.overlap_factor=overlap_factor
        if not(0.0<=overlap_factor<1.0): raise ValueError("overlap_factor must be [0.0, 1.0)")
        ovp=int(resolution*overlap_factor); self.stride=max(1,resolution-resolution+ovp)
        src_files=sorted(glob(os.path.join(src_dir,'*.*')))
        if not src_files: raise FileNotFoundError(f"No source images found in {src_dir}")
        self.skipped_count=0; self.processed_files=0; self.total_slices_generated=0; self.skipped_paths=[]
        for src_path in src_files:
            bname=os.path.basename(src_path); dst_path=os.path.join(dst_dir,bname)
            if not os.path.exists(dst_path): self.skipped_count+=1; self.skipped_paths.append((src_path,"Dst Missing")); continue
            try:
                with Image.open(src_path) as img: w,h=img.size
                if w<resolution or h<resolution: self.skipped_count+=1; self.skipped_paths.append((src_path,"Too Small")); continue
                n_s=0;
                py=list(range(0,h-resolution,self.stride))+([h-resolution] if h>resolution else [0]); px=list(range(0,w-resolution,self.stride))+([w-resolution] if w>resolution else [0])
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
# 
NORM_MEAN = [0.5, 0.5, 0.5]; NORM_STD = [0.5, 0.5, 0.5]
def denormalize(tensor):
    mean=torch.tensor(NORM_MEAN,device=tensor.device).view(1,3,1,1); std=torch.tensor(NORM_STD,device=tensor.device).view(1,3,1,1)
    return torch.clamp(tensor*std+mean,0,1)

def save_previews(model, fixed_src_batch, fixed_dst_batch, output_dir, current_epoch, global_step, device, preview_save_count, preview_refresh_rate):
    if not is_main_process() or fixed_src_batch is None or fixed_dst_batch is None: return
    num_grid_cols=3;
    if fixed_src_batch.size(0)<num_grid_cols: return
    src_select=fixed_src_batch[:num_grid_cols].cpu(); dst_select=fixed_dst_batch[:num_grid_cols].cpu()
    model.eval(); device_type=device.type
    with torch.no_grad(), autocast(device_type=device_type, enabled=torch.is_autocast_enabled()):
        src_dev=src_select.to(device); model_module=model.module if isinstance(model,DDP) else model; predicted_batch=model_module(src_dev)
    model.train()
    pred_select=predicted_batch.cpu().float()
    src_denorm=denormalize(src_select); pred_denorm=denormalize(pred_select); dst_denorm=denormalize(dst_select)
    combined=[item for i in range(num_grid_cols) for item in [src_denorm[i],dst_denorm[i],pred_denorm[i]]]
    if not combined: return
    grid_tensor=torch.stack(combined); grid=make_grid(grid_tensor,nrow=num_grid_cols,padding=2,normalize=False)
    img_pil=T.functional.to_pil_image(grid); preview_filename=os.path.join(output_dir,"training_preview.jpg")
    try:
        img_pil.save(preview_filename,"JPEG",quality=95)
        log_msg = f"Saved preview to {preview_filename} (Epoch {current_epoch+1}, Step {global_step}, Save #{preview_save_count})"
        if preview_save_count == 0 or (preview_refresh_rate > 0 and preview_save_count % preview_refresh_rate == 0):
             logging.info(log_msg + " - Refreshed Batch")
        else:
             logging.info(log_msg)
    except Exception as e: logging.error(f"Failed to save preview image: {e}")

def capture_preview_batch(config, transform):
    if not is_main_process(): return None,None
    num_preview_samples=3; logging.info(f"Refreshing fixed batch ({num_preview_samples} samples) for previews...")
    try:
        preview_dataset=ImagePairSlicingDataset(config.data.src_dir, config.data.dst_dir, config.data.resolution, overlap_factor=config.data.overlap_factor, transform=transform)
        if len(preview_dataset)>=num_preview_samples:
            preview_loader=DataLoader(preview_dataset,batch_size=num_preview_samples,shuffle=True,num_workers=0)
            fixed_src_slices,fixed_dst_slices=next(iter(preview_loader))
            if fixed_src_slices.size(0)==num_preview_samples: logging.info(f"Captured new batch of size {num_preview_samples} for previews."); return fixed_src_slices.cpu(),fixed_dst_slices.cpu()
            else: logging.warning(f"Preview DataLoader returned {fixed_src_slices.size(0)} samples instead of {num_preview_samples}."); return None,None
        else: logging.warning(f"Dataset has only {len(preview_dataset)} slices, need {num_preview_samples}. Cannot refresh preview."); return None,None
    except StopIteration: logging.error("Preview DataLoader yielded no batches during refresh."); return None,None
    except Exception as e: logging.exception(f"Error capturing preview batch: {e}"); return None,None

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
    model_state_to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    checkpoint = {
        'global_step': global_step, 'epoch': current_epoch,
        'model_state_dict': model_state_to_save, 'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if config.training.use_amp else None,
        'config': config }
    filename = CHECKPOINT_FILENAME_PATTERN.format(epoch=current_epoch + 1)
    checkpoint_path = os.path.join(output_dir, filename)
    latest_path = os.path.join(output_dir, LATEST_CHECKPOINT_FILENAME)
    try:
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")
        torch.save(checkpoint, latest_path)
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
                numbered_checkpoints.append({'path': f, 'epoch': epoch_num})
        if len(numbered_checkpoints) <= keep_last: return
        numbered_checkpoints.sort(key=lambda x: x['epoch'])
        checkpoints_to_delete = numbered_checkpoints[:-keep_last]
        logging.info(f"Found {len(numbered_checkpoints)} checkpoints. Keeping last {keep_last}. Deleting {len(checkpoints_to_delete)} older checkpoints.")
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
    setup_ddp(); rank=get_rank(); world_size=get_world_size()
    device=torch.device(f"cuda:{rank}"); device_type=device.type
    log_level=logging.INFO if is_main_process() else logging.WARNING
    log_format=f'%(asctime)s [RK{rank}][%(levelname)s] %(message)s'
    handlers=[logging.StreamHandler()]
    if is_main_process():
        os.makedirs(config.data.output_dir,exist_ok=True)
        log_file=os.path.join(config.data.output_dir,'training.log')
        handlers.append(logging.FileHandler(log_file,mode='a'))
    logging.basicConfig(level=log_level,format=log_format,handlers=handlers,force=True)

    if is_main_process():
        logging.info("="*50); logging.info("Starting training run...")
        logging.info(f"Runtime World Size (Number of GPUs/Processes): {world_size}")
        logging.info(f"Iteration-based training. Steps per epoch: {config.training.iterations_per_epoch}")
        logging.info(f"Using effective configuration: {config}")

    # --- Dataset & DataLoader ---
    #
    transform = T.Compose([ T.ToTensor(), T.Normalize(mean=NORM_MEAN, std=NORM_STD) ])
    dataset = None
    try:
        dataset = ImagePairSlicingDataset(
            config.data.src_dir, config.data.dst_dir, config.data.resolution,
            overlap_factor=config.data.overlap_factor, transform=transform )
        if is_main_process():
             logging.info(f"Dataset: Res={dataset.resolution}, Overlap={dataset.overlap_factor:.2f}, Stride={dataset.stride}")
             if dataset.skipped_count > 0: logging.warning(f"Skipped {dataset.skipped_count} pairs during dataset init.")
             if dataset.processed_files > 0:
                 avg_slices = dataset.total_slices_generated / dataset.processed_files if dataset.processed_files > 0 else 0
                 logging.info(f"Found {dataset.total_slices_generated} slices from {dataset.processed_files} pairs (avg {avg_slices:.1f}).")
             else: raise ValueError("Dataset processed 0 valid image pairs.")
             if len(dataset) == 0: raise ValueError("Dataset created but contains 0 slices.")
    except (FileNotFoundError, ValueError, Exception) as e:
        logging.error(f"Dataset initialization failed: {e}", exc_info=not isinstance(e, (FileNotFoundError, ValueError)))
        if dist.is_initialized(): cleanup_ddp(); return
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, sampler=sampler, num_workers=config.training.num_workers,
                            pin_memory=True, drop_last=True, prefetch_factor=config.training.prefetch_factor if config.training.num_workers > 0 else None)
    batches_per_dataset_pass = len(dataloader)
    if is_main_process(): logging.info(f"DataLoader created. Batches per dataset pass: {batches_per_dataset_pass}")

    # --- Model Config, Instantiation, Optimizer, Scaler, Loss ---
    # <-- model_size_dims -->
    model_size = config.model.model_size_dims
    use_lpips = False; loss_fn_lpips = None
    default_hidden_size_for_bump = 64 # The specific value that triggers the bump
    bumped_hidden_size = 96 # The value to bump to

    if config.training.loss == 'l1+lpips':
        use_lpips = True
        # Check if the *original* specified size requires bumping
        if model_size == default_hidden_size_for_bump:
            model_size = bumped_hidden_size # Update the size to be used
            # Update the config object IN PLACE so it's saved correctly
            config.model.model_size_dims = model_size
            if is_main_process(): logging.info(f"Using l1+lpips loss: Auto-increasing model_size_dims {default_hidden_size_for_bump} -> {model_size}.")
        else:
             if is_main_process(): logging.info(f"Using l1+lpips loss: model_size_dims={model_size} (as specified), LPIPS lambda={config.training.lambda_lpips}.")
        if is_main_process(): # Initialize LPIPS
            try:
                loss_fn_lpips = lpips.LPIPS(net='vgg').to(device).eval()
                for param in loss_fn_lpips.parameters(): param.requires_grad = False
                logging.info("LPIPS VGG loss initialized.")
            except Exception as e: logging.error(f"Failed to init LPIPS: {e}. Disabling LPIPS.", exc_info=True); use_lpips = False
    else:
        if is_main_process(): logging.info(f"Using {config.training.loss} loss: model_size_dims={model_size}.")

    # Instantiate the model using the (potentially modified) config
    model = UNet(config).to(device)

    try: lr_value = float(config.training.lr)
    except (ValueError, TypeError) as e:
        logging.error(f"FATAL: Could not convert learning rate '{config.training.lr}' (type: {type(config.training.lr)}) to float: {e}")
        raise ValueError(f"Invalid learning rate value in config: {config.training.lr}") from e
    optimizer = optim.AdamW(model.parameters(), lr=lr_value, weight_decay=1e-5)
    scaler = GradScaler(enabled=config.training.use_amp)
    criterion_l1 = nn.L1Loss()

    # --- Resume Logic ---
    start_iteration = 0
    latest_checkpoint_path = os.path.join(config.data.output_dir, LATEST_CHECKPOINT_FILENAME)
    resume_flag = [os.path.exists(latest_checkpoint_path)] if is_main_process() else [False]
    if world_size > 1: dist.broadcast_object_list(resume_flag, src=0)
    if resume_flag[0]:
        try:
            if is_main_process(): logging.info(f"Attempting resume from: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, map_location='cpu')
            # Compatibility Check
            if 'config' not in checkpoint and 'args' not in checkpoint: raise ValueError("Checkpoint missing config/args.")
            ckpt_config_raw = checkpoint.get('config', checkpoint.get('args'))
            ckpt_config = dict_to_namespace(ckpt_config_raw) if isinstance(ckpt_config_raw, dict) else ckpt_config_raw
            if isinstance(ckpt_config, argparse.Namespace): ckpt_config = SimpleNamespace(**vars(ckpt_config))

            # <-- CHANGED: Checkpoint loading logic for backward compatibility -->
            ckpt_loss = 'l1'; ckpt_size_saved = 64 # Defaults
            # Get Loss
            if hasattr(ckpt_config,'training') and hasattr(ckpt_config.training,'loss'): ckpt_loss=ckpt_config.training.loss
            elif hasattr(ckpt_config,'loss'): ckpt_loss=ckpt_config.loss
            # Get Model Size (try new name first, fallback to old)
            if hasattr(ckpt_config,'model'):
                ckpt_size_saved = getattr(ckpt_config.model, 'model_size_dims', getattr(ckpt_config.model, 'unet_hidden_size', 64))
            elif hasattr(ckpt_config, 'model_size_dims'): ckpt_size_saved = ckpt_config.model_size_dims
            elif hasattr(ckpt_config, 'unet_hidden_size'): ckpt_size_saved = ckpt_config.unet_hidden_size

            # Calculate effective sizes for comparison
            ckpt_expected_size = ckpt_size_saved
            if ckpt_loss=='l1+lpips' and ckpt_size_saved == default_hidden_size_for_bump:
                ckpt_expected_size = bumped_hidden_size
            # Current size is already potentially bumped
            current_expected_size = config.model.model_size_dims

            # Perform compatibility checks
            if ckpt_loss != config.training.loss:
                 current_loss = getattr(config.training, 'loss', 'l1')
                 raise ValueError(f"Loss mode mismatch: ckpt='{ckpt_loss}', requested='{current_loss}'")
            if ckpt_expected_size != current_expected_size:
                 raise ValueError(f"Model size mismatch: ckpt effective={ckpt_expected_size}, requested={current_expected_size}")
            if is_main_process(): logging.info("Checkpoint configuration compatible.")
            # <-- END CHANGED SECTION -->

            # Load State
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if config.training.use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            # Load global_step...
            if 'global_step' in checkpoint:
                start_iteration = checkpoint['global_step']
                if is_main_process(): logging.info(f"Resuming training from iteration {start_iteration + 1}.")
            else: # Fallback...
                 start_epoch_legacy = checkpoint.get('epoch', 0)
                 if start_epoch_legacy > 0:
                     start_iteration = start_epoch_legacy * config.training.iterations_per_epoch
                     logging.warning(f"Checkpoint missing 'global_step'. Found legacy 'epoch' {start_epoch_legacy}.")
                     logging.warning(f"Estimating resume iteration as {start_iteration + 1} based on current iterations_per_epoch={config.training.iterations_per_epoch}.")
                 else:
                     logging.warning("Checkpoint missing 'global_step' and 'epoch'. Starting from iteration 0.")
                     start_iteration = 0
            del checkpoint;
        except Exception as e:
            logging.error(f"Failed load/resume from {latest_checkpoint_path}: {e}", exc_info=True)
            logging.warning("Starting from scratch (Iteration 0)."); start_iteration = 0
    else:
        if is_main_process(): logging.info(f"No checkpoint found at {latest_checkpoint_path}. Starting from iteration 0.")

    # --- Wrap model with DDP ---
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # --- Preview State ---
    # 
    fixed_src_slices = None; fixed_dst_slices = None; preview_save_count = 0
    if start_iteration == 0 and is_main_process():
         if config.logging.preview_batch_interval > 0:
             logging.info("Capturing initial fixed batch for previews...")
             fixed_src_slices, fixed_dst_slices = capture_preview_batch(config, transform)
             if fixed_src_slices is None: logging.warning("Could not capture initial preview batch.")


    # --- Training Loop ---
    # 
    global_step = start_iteration
    current_epoch = start_iteration // config.training.iterations_per_epoch
    last_logged_batch_loss = {}
    if is_main_process(): logging.info(f"Starting training loop from global step: {global_step + 1}")

    try: # Wrap main loop
        while True: # Loop indefinitely
            model.train()
            sampler.set_epoch(current_epoch)
            if is_main_process() and global_step % config.training.iterations_per_epoch == 0:
                logging.info(f"Starting Epoch {current_epoch + 1} (Dataset Pass approx {(global_step // batches_per_dataset_pass) + 1 if batches_per_dataset_pass > 0 else 1})")

            batch_iter_start_time = time.time()
            for i, batch_data in enumerate(dataloader):
                # Batch processing...
                if batch_data is None:
                     if is_main_process(): logging.warning(f"Step[{global_step+1}]: Received None data, skipping.")
                     continue
                try: src_slices, dst_slices = batch_data
                except Exception as e:
                    if is_main_process(): logging.error(f"Step[{global_step+1}]: Batch data error: {e}, skipping.")
                    continue
                iter_data_end_time = time.time()
                src_slices=src_slices.to(device, non_blocking=True); dst_slices=dst_slices.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=device_type, enabled=config.training.use_amp):
                    outputs = model(src_slices); l1_loss = criterion_l1(outputs, dst_slices); lpips_loss = torch.tensor(0.0, device=device)
                    if use_lpips and loss_fn_lpips is not None:
                        try: lpips_loss = loss_fn_lpips(outputs, dst_slices).mean()
                        except Exception as e:
                             if is_main_process() and global_step % (config.logging.log_interval * 10) == 0:
                                  logging.warning(f"Step[{global_step+1}]: LPIPS calc failed: {e}. Skip LPIPS this batch.")
                             lpips_loss=torch.tensor(0.0,device=device)
                    loss = l1_loss + config.training.lambda_lpips * lpips_loss
                if torch.isnan(loss):
                    if is_main_process(): logging.error(f"Step[{global_step+1}]: NaN loss detected! L1={l1_loss.item():.4f}, LPIPS={lpips_loss.item():.4f}. Skipping update.")
                    global_step += 1; continue
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update();
                iter_compute_end_time = time.time()
                global_step += 1

                # Logging, Preview, Checkpoint on main process
                if is_main_process():
                    # Batch Loss Logging...
                    dist.all_reduce(l1_loss, op=dist.ReduceOp.AVG); batch_l1_loss_avg = l1_loss.item()
                    batch_lpips_loss_avg = 0.0
                    if use_lpips: dist.all_reduce(lpips_loss, op=dist.ReduceOp.AVG); batch_lpips_loss_avg = lpips_loss.item()
                    if global_step % config.logging.log_interval == 0:
                        current_lr=optimizer.param_groups[0]['lr']; data_time=iter_data_end_time-batch_iter_start_time; compute_time=iter_compute_end_time-iter_data_end_time
                        total_step_time=time.time()-batch_iter_start_time
                        log_msg = (f'Epoch[{current_epoch+1}], Step[{global_step}], L1:{batch_l1_loss_avg:.4f}')
                        if use_lpips: log_msg += (f', LPIPS:{batch_lpips_loss_avg:.4f}')
                        log_msg += (f', LR:{current_lr:.1e}, T/Step:{total_step_time:.3f}s (D:{data_time:.3f},C:{compute_time:.3f})')
                        logging.info(log_msg)
                        batch_iter_start_time=time.time()
                    # Batch Preview Generation...
                    if config.logging.preview_batch_interval > 0 and global_step % config.logging.preview_batch_interval == 0:
                        refresh_preview = (fixed_src_slices is None) or \
                                          (config.logging.preview_refresh_rate > 0 and preview_save_count % config.logging.preview_refresh_rate == 0)
                        if refresh_preview and (fixed_src_slices is None or preview_save_count > 0):
                            new_src, new_dst = capture_preview_batch(config, transform)
                            if new_src is not None and new_dst is not None: fixed_src_slices, fixed_dst_slices = new_src, new_dst
                            elif fixed_src_slices is None: logging.warning(f"Step {global_step}: Skipping preview (batch capture failed).")
                        if fixed_src_slices is not None:
                            save_previews(model, fixed_src_slices, fixed_dst_slices, config.data.output_dir,
                                          current_epoch, global_step, device,
                                          preview_save_count, config.logging.preview_refresh_rate)
                            preview_save_count += 1
                    # Checkpoint Saving & Management...
                    if global_step > 0 and global_step % config.training.iterations_per_epoch == 0:
                         logging.info(f"Epoch {current_epoch + 1} finished at step {global_step}.")
                         save_checkpoint(model, optimizer, scaler, config, global_step, current_epoch, config.data.output_dir)
                         manage_checkpoints(config.data.output_dir, config.saving.keep_last_checkpoints)

                # Check if epoch completed to break inner loop
                if global_step > 0 and global_step % config.training.iterations_per_epoch == 0:
                     if world_size > 1: dist.barrier()
                     current_epoch += 1
                     break
            # End of inner dataloader loop

    except KeyboardInterrupt:
        if is_main_process(): logging.warning("Training interrupted by user (KeyboardInterrupt).")
    finally:
        if is_main_process(): logging.info("Cleaning up DDP...")
        cleanup_ddp()
        if is_main_process(): logging.info("Training script finished.")

# --- Config Merging Helper ---
# 
def merge_dicts(base, user):
    merged = copy.deepcopy(base)
    for key, value in user.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_dicts(merged[key], value)
        else: merged[key] = value
    return merged

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TuNet using YAML config with DDP (Iteration-Based)')
    parser.add_argument('--config', type=str, required=True, help='Path to the USER configuration file')
    # Define CLI overrides matching the nested structure
    parser.add_argument('--data.src_dir', type=str, dest='data_src_dir')
    parser.add_argument('--data.dst_dir', type=str, dest='data_dst_dir')
    parser.add_argument('--data.output_dir', type=str, dest='data_output_dir')
    parser.add_argument('--data.resolution', type=int, dest='data_resolution')
    # <-- CHANGED: CLI override name -->
    parser.add_argument('--model.model_size_dims', type=int, dest='model_model_size_dims')
    # <-- REMOVED old CLI override -->
    # parser.add_argument('--model.unet_hidden_size', type=int, dest='model_unet_hidden_size')
    parser.add_argument('--training.iterations_per_epoch', type=int, dest='training_iterations_per_epoch')
    parser.add_argument('--training.batch_size', type=int, dest='training_batch_size')
    parser.add_argument('--training.lr', type=float, dest='training_lr')
    parser.add_argument('--training.loss', type=str, choices=['l1', 'l1+lpips'], dest='training_loss')
    parser.add_argument('--training.lambda_lpips', type=float, dest='training_lambda_lpips')
    parser.add_argument('--training.use_amp', type=lambda x: (str(x).lower() == 'true'), dest='training_use_amp')
    parser.add_argument('--logging.preview_batch_interval', type=int, dest='logging_preview_batch_interval')
    parser.add_argument('--saving.keep_last_checkpoints', type=int, dest='saving_keep_last_checkpoints')

    cli_args = parser.parse_args()

    # --- Load Base Config ---
    base_config_dict = {}
    try:
        # Look for base.yaml in base/ directory relative to user config OR script
        user_config_dir = os.path.dirname(os.path.abspath(cli_args.config))
        base_path1 = os.path.join(user_config_dir, 'base', 'base.yaml')

        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_path2 = os.path.join(script_dir, 'base', 'base.yaml')

        base_config_path = None
        if os.path.exists(base_path1):
            base_config_path = base_path1
        elif os.path.exists(base_path2):
            base_config_path = base_path2
        else:
            print(f"ERROR: Base configuration file ('base/base.yaml') not found relative to user config ('{user_config_dir}/base/') or script ('{script_dir}/base/').")
            exit(1)

        print(f"Loading base config from: {base_config_path}")
        with open(base_config_path, 'r') as f:
            base_config_dict = yaml.safe_load(f)
        if base_config_dict is None: base_config_dict = {}
        print("Base config loaded successfully.")
    except yaml.YAMLError as e: print(f"ERROR: Parsing base YAML file {base_config_path}: {e}"); exit(1)
    except Exception as e: print(f"ERROR: Reading base config file {base_config_path}: {e}"); exit(1)

    # --- Load User Config ---
    user_config_dict = {}
    try:
        print(f"Loading user config from: {cli_args.config}")
        with open(cli_args.config, 'r') as f:
            user_config_dict = yaml.safe_load(f)
        if user_config_dict is None: user_config_dict = {}
        print("User config loaded successfully.")
    except FileNotFoundError: print(f"ERROR: User configuration file not found: {cli_args.config}"); exit(1)
    except yaml.YAMLError as e: print(f"ERROR: Parsing user YAML file {cli_args.config}: {e}"); exit(1)
    except Exception as e: print(f"ERROR: Reading user config file {cli_args.config}: {e}"); exit(1)

    # --- Merge Configs ---
    merged_config_dict = merge_dicts(base_config_dict, user_config_dict)
    print("Configs merged.")

    # --- Apply CLI Overrides ---
    # <-- override_map -->
    override_map = {
        'data_src_dir': ['data', 'src_dir'],
        'data_dst_dir': ['data', 'dst_dir'],
        'data_output_dir': ['data', 'output_dir'],
        'data_resolution': ['data', 'resolution'],
        'model_model_size_dims': ['model', 'model_size_dims'], # New name
        'training_iterations_per_epoch': ['training', 'iterations_per_epoch'],
        'training_batch_size': ['training', 'batch_size'],
        'training_lr': ['training', 'lr'],
        'training_loss': ['training', 'loss'],
        'training_lambda_lpips': ['training', 'lambda_lpips'],
        'training_use_amp': ['training', 'use_amp'],
        'logging_preview_batch_interval': ['logging', 'preview_batch_interval'],
        'saving_keep_last_checkpoints': ['saving', 'keep_last_checkpoints'],
    }
    def set_nested(dic, keys, value):
        for key in keys[:-1]: dic = dic.setdefault(key, {})
        dic[keys[-1]] = value
    for arg_key, nested_keys in override_map.items():
        value = getattr(cli_args, arg_key, None)
        if value is not None:
            set_nested(merged_config_dict, nested_keys, value)
            print(f"Applied CLI override: {' -> '.join(nested_keys)} = {value}")

    # Convert final merged dict to Namespace
    config = dict_to_namespace(merged_config_dict)

    # --- Basic Validation ---
    # <-- required_keys -->
    required_keys = {
        'data': ['src_dir', 'dst_dir', 'output_dir', 'resolution', 'overlap_factor'],
        'model': ['model_size_dims'], # Check for the new name
        'training': ['iterations_per_epoch', 'batch_size', 'lr', 'loss', 'lambda_lpips', 'use_amp', 'num_workers', 'prefetch_factor'],
        'logging': ['log_interval', 'preview_batch_interval', 'preview_refresh_rate'],
        'saving': ['keep_last_checkpoints']
    }
    missing = False; error_msgs = []
    for section, keys in required_keys.items():
        if not hasattr(config, section): error_msgs.append(f"Missing config section '{section}'"); missing = True; continue
        section_obj = getattr(config, section)
        for key in keys:
            if not hasattr(section_obj, key): error_msgs.append(f"Missing configuration key '{section}.{key}'"); missing = True
    if missing:
        print("ERROR: Missing required configuration keys:")
        for msg in error_msgs: print(f"  - {msg}")
        exit(1)

    # Validate specific values
    if not (0.0 <= config.data.overlap_factor < 1.0): print(f"ERROR: data.overlap_factor must be [0.0, 1.0)"); exit(1)
    if config.logging.preview_batch_interval < 0: config.logging.preview_batch_interval = 0
    if config.logging.preview_refresh_rate < 0: config.logging.preview_refresh_rate = 0
    if config.saving.keep_last_checkpoints < 0: print("WARNING: saving.keep_last_checkpoints cannot be negative. Setting to 0."); config.saving.keep_last_checkpoints = 0
    lr_check_value = getattr(config.training, 'lr', None)
    if not isinstance(lr_check_value, (int, float)) or lr_check_value <= 0: print(f"ERROR: training.lr must be a positive number. Got: {lr_check_value!r} (type: {type(lr_check_value)})"); exit(1)
    if config.training.lambda_lpips < 0: print(f"ERROR: training.lambda_lpips must be non-negative. Got: {config.training.lambda_lpips}"); exit(1)
    if config.training.iterations_per_epoch <= 0: print(f"ERROR: training.iterations_per_epoch must be positive."); exit(1)
    if config.data.resolution not in [512, 1024]: print(f"WARNING: data.resolution ({config.data.resolution}) not 512 or 1024.")
    if config.training.loss not in ['l1', 'l1+lpips']: print(f"ERROR: training.loss must be 'l1' or 'l1+lpips'"); exit(1)

    # --- DDP Check & Launch ---
    if "LOCAL_RANK" not in os.environ:
         print("="*40 + "\nERROR: DDP environment variables not found! \nPlease launch using 'torchrun'. \nExample: torchrun --nproc_per_node=NUM_GPUS train.py --config path/to/user_config.yaml\n" + "="*40)
         exit(1)

    train(config) # Pass the final merged config object

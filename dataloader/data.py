# dataloader/data.py
import os
from glob import glob
from PIL import Image, UnidentifiedImageError # Import specific error
import logging
import importlib # To dynamically import augmentation classes
import time # For debug timing
from types import SimpleNamespace 

import torch
from torch.utils.data import Dataset
import numpy as np
import albumentations
from albumentations.pytorch import ToTensorV2 # Useful, but we stick to torchvision T.ToTensor for now

# --- Base Dataset (Loads PIL Slices) - WITH FILE FILTERING ---
class BaseImagePairSlicingDataset(Dataset):
    """
    Original dataset logic for finding and slicing image pairs.
    Loads PIL images. Includes filtering for common image types.
    """
    def __init__(self, src_dir, dst_dir, resolution, overlap_factor=0.0):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.resolution = resolution
        self.slice_info = []
        self.overlap_factor = overlap_factor

        if not (0.0 <= overlap_factor < 1.0):
            raise ValueError("overlap_factor must be in the range [0.0, 1.0)")

        overlap_pixels = int(resolution * overlap_factor)
        self.stride = max(1, resolution - overlap_pixels)

        # --- MODIFICATION: Filter for image files ---
        image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.webp', '*.tiff'] # Common image patterns
        src_files_raw = []
        logging.debug(f"Searching for images in {src_dir} with patterns: {image_patterns}")
        for pattern in image_patterns:
            found_files = glob(os.path.join(src_dir, pattern))
            if found_files:
                 logging.debug(f"  Found {len(found_files)} files matching '{pattern}'")
                 src_files_raw.extend(found_files)

        # Remove duplicates if multiple patterns match the same file and sort
        src_files = sorted(list(set(src_files_raw)))
        logging.debug(f"Total unique potential source image files found: {len(src_files)}")

        if not src_files:
             # Check if the directory exists at all
             if not os.path.isdir(src_dir):
                  raise FileNotFoundError(f"Source directory not found: {src_dir}")
             # Log if glob just found nothing
             logging.error(f"No image files matching patterns {image_patterns} found in {src_dir}.")
             # Raise specific error if no images found after filtering
             raise FileNotFoundError(f"No valid source image files found in {src_dir}. Searched patterns: {image_patterns}")
        # --- END MODIFICATION ---

        self.skipped_count = 0
        self.processed_files = 0
        self.total_slices_generated = 0
        self.skipped_paths = []

        logging.info(f"Processing {len(src_files)} potential source images...")
        for src_path in src_files: # Now iterates over the filtered list
            basename = os.path.basename(src_path)
            # Try common extensions for the destination, be more robust
            dst_path = None
            base_name_no_ext, src_ext = os.path.splitext(basename)
            # Build a list of potential destination file names
            potential_dst_names = [
                 basename, # Exact match first
                 base_name_no_ext + '.png',
                 base_name_no_ext + '.jpg',
                 base_name_no_ext + '.jpeg',
                 base_name_no_ext + '.webp',
                 base_name_no_ext + '.tiff',
                 base_name_no_ext + src_ext # Also try original extension if different
            ]
            # Remove duplicates from potential names and create full paths
            potential_dst_paths = [os.path.join(dst_dir, name) for name in list(dict.fromkeys(potential_dst_names))]

            found_dst = False
            for p in potential_dst_paths:
                 if os.path.exists(p):
                     dst_path = p
                     found_dst = True
                     break # Found a matching destination file

            if not found_dst:
                self.skipped_count += 1
                self.skipped_paths.append((src_path, "Dst Missing"))
                logging.debug(f"Skipping {basename}: Destination file not found.")
                continue

            try:
                # Check source image validity and get dimensions
                with Image.open(src_path) as img:
                    w, h = img.size

                # Check destination image validity and dimensions
                with Image.open(dst_path) as dst_img_check:
                    dw, dh = dst_img_check.size

                # Check for size mismatch between source and destination
                if w != dw or h != dh:
                     self.skipped_count += 1
                     self.skipped_paths.append((src_path, f"Size Mismatch (Src: {w}x{h}, Dst: {dw}x{dh})"))
                     logging.warning(f"Skipping {basename}: Size mismatch (Src: {w}x{h}, Dst: {dw}x{dh})")
                     continue

                # Check if image is large enough for slicing
                if w < resolution or h < resolution:
                    self.skipped_count += 1
                    self.skipped_paths.append((src_path, f"Too Small (Needs {resolution}x{resolution}, Got {w}x{h})"))
                    logging.warning(f"Skipping {basename}: Too small (Needs {resolution}x{resolution}, Got {w}x{h})")
                    continue

                # --- Slicing Logic ---
                num_slices_for_file = 0
                y_coords = list(range(0, h - resolution, self.stride)) + ([h - resolution] if h > resolution else [0])
                x_coords = list(range(0, w - resolution, self.stride)) + ([w - resolution] if w > resolution else [0])
                unique_y_coords = sorted(list(set(y_coords)))
                unique_x_coords = sorted(list(set(x_coords)))

                for y in unique_y_coords:
                    for x in unique_x_coords:
                        coords = (x, y, x + resolution, y + resolution)
                        self.slice_info.append((src_path, dst_path, coords))
                        num_slices_for_file += 1
                # --- End Slicing Logic ---

                if num_slices_for_file > 0:
                    self.processed_files += 1
                    self.total_slices_generated += num_slices_for_file
                    logging.debug(f"Processed {basename}: Generated {num_slices_for_file} slices.")

            except UnidentifiedImageError as e:
                 # Catch error if PIL cannot identify a file that passed the glob filter (e.g., corrupted)
                 self.skipped_count += 1
                 self.skipped_paths.append((src_path, f"Unidentified Image Error: {e}"))
                 logging.warning(f"Skipping {basename}: Unidentified Image Error - {e}")
                 continue
            except Exception as e:
                # Catch other potential errors during image loading or checking
                self.skipped_count += 1
                self.skipped_paths.append((src_path, f"Error Processing: {e}"))
                logging.warning(f"Skipping {basename}: Error during processing - {e}", exc_info=True) # Log traceback for unexpected errors
                continue

        # --- Final Check After Loop ---
        if self.total_slices_generated == 0:
             logging.error(f"Failed to generate any slices after processing {len(src_files)} potential source files.")
             if self.processed_files == 0:
                  logging.error("No source files could be fully processed (check skip reasons).")
             # Log skip reasons summary if available
             if self.skipped_paths:
                  logging.error(f"Total skipped source files/pairs: {self.skipped_count}. Reasons summary:")
                  reason_counts = {}
                  for _, reason in self.skipped_paths:
                      reason_base = reason.split('(')[0].split(':')[0].strip() # Get base reason type
                      reason_counts[reason_base] = reason_counts.get(reason_base, 0) + 1
                  for reason, count in reason_counts.items():
                       logging.error(f"  - {reason}: {count} times")
             # Raise an error if really nothing worked
             raise ValueError("Dataset could not be created. No valid image pairs or slices generated. Check logs.")
        else:
             logging.info(f"Base dataset initialization complete. Generated {self.total_slices_generated} slices from {self.processed_files} files.")
             if self.skipped_count > 0:
                  logging.info(f"Skipped {self.skipped_count} source files/pairs during initialization.")


    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, idx):
        src_path, dst_path, coords = self.slice_info[idx]
        try:
            # Load as PIL images - Should succeed now due to checks in __init__
            src_img = Image.open(src_path).convert('RGB')
            dst_img = Image.open(dst_path).convert('RGB')

            # Crop the slices
            src_slice_pil = src_img.crop(coords)
            dst_slice_pil = dst_img.crop(coords)

            # Ensure correct size (should match resolution, but resize just in case coords were edge cases)
            if src_slice_pil.size != (self.resolution, self.resolution):
                src_slice_pil = src_slice_pil.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
            if dst_slice_pil.size != (self.resolution, self.resolution):
                dst_slice_pil = dst_slice_pil.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)

            # Return PIL images - augmentation and final transform happen later
            return src_slice_pil, dst_slice_pil

        except Exception as e:
            logging.error(f"Error loading/processing slice idx {idx} ({src_path}, coords={coords}): {e}", exc_info=True)
            # Return None or raise error? Let's raise to prevent downstream issues.
            raise RuntimeError(f"Error loading/processing slice idx {idx} ({src_path}, coords={coords})") from e


# --- Augmentation Helper Function ---

def create_augmentations(aug_config_list):
    """
    Creates an Albumentations Compose pipeline from a list of config dictionaries.
    """
    transforms_list = []
    if not aug_config_list:
        return albumentations.Compose([]) # Return empty Compose if no augs

    for item in aug_config_list:
        # Ensure item is a dict (or SimpleNamespace converted to dict)
        if isinstance(item, SimpleNamespace):
             item_dict = vars(item)
        elif isinstance(item, dict):
             item_dict = item
        else:
             logging.warning(f"Skipping invalid augmentation config item (must be dict/mapping): {item}")
             continue

        if '_target_' not in item_dict:
            logging.warning(f"Skipping invalid augmentation config item (missing '_target_'): {item_dict}")
            continue

        target_str = item_dict['_target_']
        kwargs = {k: v for k, v in item_dict.items() if k not in ['_target_', 'p']}
        probability = item_dict.get('p', 1.0) # Default probability is 1.0

        try:
            aug_class = None
            if target_str == 'color_jitter_transform':
                 # Map your custom name to albumentations.ColorJitter
                 brightness_limit = kwargs.get('brightness', 0.2)
                 contrast_limit = kwargs.get('contrast', 0.2)
                 saturation_limit = kwargs.get('saturation', 0.2)
                 hue_shift_limit = kwargs.get('hue', 0.1)
                 aug_class = albumentations.ColorJitter
                 # Update kwargs for ColorJitter specifically
                 kwargs = {
                     'brightness': brightness_limit,
                     'contrast': contrast_limit,
                     'saturation': saturation_limit,
                     'hue': hue_shift_limit,
                 }
                 logging.debug(f"  Mapping 'color_jitter_transform' to ColorJitter with kwargs: {kwargs}, p={probability}")

            elif '.' in target_str: # Handle potential module path like albumentations.HorizontalFlip
                 module_name, class_name = target_str.rsplit('.', 1)
                 module = importlib.import_module(module_name)
                 aug_class = getattr(module, class_name)
                 logging.debug(f"  Loading augmentation {class_name} from {module_name}")
            else: # Try finding in albumentations directly
                 if hasattr(albumentations, target_str):
                      aug_class = getattr(albumentations, target_str)
                      logging.debug(f"  Loading augmentation {target_str} from albumentations base")
                 else: # Try finding in albumentations.augmentations.transforms
                      try:
                           module = importlib.import_module('albumentations.augmentations.transforms')
                           if hasattr(module, target_str):
                                aug_class = getattr(module, target_str)
                                logging.debug(f"  Loading augmentation {target_str} from albumentations.augmentations.transforms")
                      except ImportError:
                           pass # Ignore if this submodule doesn't exist or class isn't there

                 if aug_class is None:
                      logging.warning(f"Could not find augmentation class for target: {target_str}. Skipping.")
                      continue

            # Handle specific arguments like interpolation if needed
            if 'interpolation' in kwargs and 'interpolation' in aug_class.__init__.__code__.co_varnames:
                 # Albumentations uses cv2 flags (int): INTER_NEAREST=0, INTER_LINEAR=1, INTER_CUBIC=2, INTER_LANCZOS4=4
                 # Assume integer value from YAML is correct. No conversion needed unless YAML uses strings.
                 pass

            # Instantiate the augmentation
            aug = aug_class(**kwargs, p=probability)
            transforms_list.append(aug)
            logging.debug(f"    Added {target_str} with p={probability} and kwargs={kwargs}")

        except ImportError:
            logging.error(f"Could not import module for augmentation: {target_str}. Skipping.")
        except AttributeError:
            logging.error(f"Could not find class {target_str} in its module or attribute error within class. Skipping.")
        except Exception as e:
            logging.error(f"Error instantiating augmentation {target_str} with kwargs {kwargs}: {e}. Skipping.", exc_info=True)

    return albumentations.Compose(transforms_list)


# --- New Augmented Dataset - WITH DEBUG PRINTS ---

class AugmentedImagePairSlicingDataset(Dataset):
    """
    A wrapper dataset that applies Albumentations augmentations.
    Includes debug prints for shared augmentation application.
    """
    def __init__(self, src_dir, dst_dir, resolution, overlap_factor=0.0,
                 src_transforms=None, dst_transforms=None, shared_transforms=None,
                 final_transform=None):
        """
        Args:
            src_dir (str): Directory containing source images.
            dst_dir (str): Directory containing target images.
            resolution (int): The resolution of the slices to extract.
            overlap_factor (float): The overlap factor for slicing [0.0, 1.0).
            src_transforms (albumentations.Compose): Augmentations applied only to source.
            dst_transforms (albumentations.Compose): Augmentations applied only to destination.
            shared_transforms (albumentations.Compose): Augmentations applied to both src and dst.
            final_transform (callable, optional): Torchvision transform (e.g., ToTensor, Normalize)
                                                 applied AFTER augmentations.
        """
        # Initialize the base dataset to get PIL slices
        self.base_dataset = BaseImagePairSlicingDataset(src_dir, dst_dir, resolution, overlap_factor)

        self.src_transforms = src_transforms if src_transforms else albumentations.Compose([])
        self.dst_transforms = dst_transforms if dst_transforms else albumentations.Compose([])
        self.shared_transforms = shared_transforms if shared_transforms else albumentations.Compose([])
        self.final_transform = final_transform

        # Expose some properties from base_dataset for logging in train.py
        self.resolution = self.base_dataset.resolution
        self.overlap_factor = self.base_dataset.overlap_factor
        self.stride = self.base_dataset.stride
        self.skipped_count = self.base_dataset.skipped_count
        self.processed_files = self.base_dataset.processed_files
        self.total_slices_generated = self.base_dataset.total_slices_generated
        self.skipped_paths = self.base_dataset.skipped_paths


    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 1. Get PIL image slices from the base dataset
        src_pil, dst_pil = self.base_dataset[idx]

        # 2. Convert PIL images to NumPy arrays (HWC format)
        src_np = np.array(src_pil)
        dst_np = np.array(dst_pil)

        # --- DEBUG Vars ---
        is_shared_applied_flag = False
        src_checksum_before = 0
        dst_checksum_before = 0
        shared_transform_time = 0
        # Calculate checksums only if we intend to log this index
        log_this_index = (idx % 100 == 0) # Example: Log every 100th sample
        if log_this_index:
            src_checksum_before = src_np.sum() if src_np is not None else 0
            dst_checksum_before = dst_np.sum() if dst_np is not None else 0
        # --- End DEBUG Vars ---

        # 3. Apply shared transformations
        #    Albumentations applies geometric transforms consistently if 'mask' is provided
        if self.shared_transforms.transforms: # Check if there are any shared transforms
             try:
                  transform_start_time = time.time()
                  transformed = self.shared_transforms(image=src_np, mask=dst_np)
                  transform_end_time = time.time()
                  shared_transform_time = transform_end_time - transform_start_time
                  is_shared_applied_flag = True # Mark as tried

                  src_np = transformed['image']
                  dst_np = transformed['mask']
             except Exception as e:
                  logging.error(f"Error applying shared transform on index {idx}: {e}", exc_info=True)
                  # Fallback or re-raise? Re-raise is safer to stop bad data.
                  raise RuntimeError(f"Error in shared augmentation for index {idx}") from e

        # --- DEBUG LOG ---
        if log_this_index:
            log_msg = f"[DEBUG Dataset idx={idx}] Shared Aug Pipeline Executed: {is_shared_applied_flag}."
            if is_shared_applied_flag:
                 src_checksum_after = src_np.sum() if src_np is not None else 0
                 dst_checksum_after = dst_np.sum() if dst_np is not None else 0
                 log_msg += f" Time: {shared_transform_time:.4f}s."
                 src_changed = src_checksum_before != src_checksum_after
                 dst_changed = dst_checksum_before != dst_checksum_after
                 log_msg += f" Src Checksum Changed: {src_changed} ({src_checksum_before} -> {src_checksum_after})."
                 log_msg += f" Dst Checksum Changed: {dst_changed} ({dst_checksum_before} -> {dst_checksum_after})."
                 # Note: Checksum might not change for some valid augmentations (e.g., flip on symmetrical image)
                 if not src_changed and not dst_changed:
                      log_msg += " (Note: Checksum unchanged - may be normal for this aug/image)."
            logging.info(log_msg)
        # --- END DEBUG LOG ---

        # 4. Apply source-specific transformations
        if self.src_transforms.transforms:
            try:
                 src_np = self.src_transforms(image=src_np)['image']
            except Exception as e:
                 logging.error(f"Error applying source transform on index {idx}: {e}", exc_info=True)
                 raise RuntimeError(f"Error in source-specific augmentation for index {idx}") from e

        # 5. Apply destination-specific transformations
        if self.dst_transforms.transforms:
            try:
                 dst_np = self.dst_transforms(image=dst_np)['image']
            except Exception as e:
                 logging.error(f"Error applying destination transform on index {idx}: {e}", exc_info=True)
                 raise RuntimeError(f"Error in destination-specific augmentation for index {idx}") from e

        # 6. Apply final torchvision transforms (ToTensor, Normalize)
        if self.final_transform:
            try:
                # Ensure uint8 HWC format before ToTensor
                # Albumentations usually returns float32, need conversion if ToTensor expects uint8
                # However, torchvision ToTensor CAN handle float32 HWC numpy arrays.
                # Let's assume ToTensor handles it. If normalization errors occur, convert here:
                # src_np_uint8 = src_np.astype(np.uint8)
                # dst_np_uint8 = dst_np.astype(np.uint8)
                # src_tensor = self.final_transform(src_np_uint8)
                # dst_tensor = self.final_transform(dst_np_uint8)

                # Assuming ToTensor handles float32 input correctly
                src_tensor = self.final_transform(src_np)
                dst_tensor = self.final_transform(dst_np)

            except TypeError as e:
                 # Common error if ToTensor receives unexpected dtype/shape
                 logging.error(f"TypeError applying final transform on index {idx}: {e}", exc_info=True)
                 logging.error(f"  src_np shape: {src_np.shape}, dtype: {src_np.dtype}")
                 logging.error(f"  dst_np shape: {dst_np.shape}, dtype: {dst_np.dtype}")
                 raise RuntimeError(f"TypeError applying final transform for index {idx}. Check input array format.") from e
            except Exception as e:
                 logging.error(f"Error applying final transform on index {idx}: {e}", exc_info=True)
                 raise RuntimeError(f"Error applying final transform for index {idx}") from e

        else:
            # If no final transform, convert to tensor manually (CHW, float 0-1)
            src_tensor = torch.from_numpy(src_np.transpose(2, 0, 1)).float() / 255.0 if src_np.max() > 1 else torch.from_numpy(src_np.transpose(2, 0, 1)).float()
            dst_tensor = torch.from_numpy(dst_np.transpose(2, 0, 1)).float() / 255.0 if dst_np.max() > 1 else torch.from_numpy(dst_np.transpose(2, 0, 1)).float()


        return src_tensor, dst_tensor

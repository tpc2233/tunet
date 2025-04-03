Infernece:

Loads Trained Checkpoint: Reads the saved .pth file.
Reads Training Config: Automatically determines the resolution used during training from the checkpoint's saved arguments.
Tiled Inference: Slices large input images into patches matching the training resolution.
Overlapping Slices: Uses a configurable overlap (--overlap_factor) during slicing.
Blending: Implements Hann window blending during stitching to reduce edge artifacts between slices significantly compared to simple averaging.
Batch Processing: Processes slices in batches for GPU efficiency.
Preprocessing/Postprocessing: Applies the same normalization (ToTensor, Normalize) as training and reverses it (denormalize) for saving.
Device Selection: Supports CPU or CUDA inference.
AMP Support: Optional mixed-precision inference (--use_amp).

Training:

Image-to-Image Learning (UNet): Trains a standard UNet model to learn the transformation required to convert source (src) images into target (dst) images based on paired examples.
Multi-GPU Training (DDP): Utilizes PyTorch's Distributed Data Parallel (DDP), launched via torchrun, for efficient, synchronized training across multiple GPUs on a single machine (or multiple machines with appropriate setup).
Tiled Training Data: Employs a custom dataset (ImagePairSlicingDataset) to handle large images by breaking them down into smaller patches suitable for GPU memory.
Configurable Slice Resolution: Supports training on fixed-size image patches, typically 512x512 or 1024x1024 (--resolution).
Overlapping Training Slices: Allows configuring an overlap factor (--overlap_factor) between training patches. This provides more training data per image and can help the model learn edge consistency, aligning training more closely with the recommended inference approach.
Input Preprocessing: Applies necessary transformations like converting images to tensors (ToTensor) and normalizing pixel values (Normalize) before feeding them to the UNet.
Mixed Precision Training (AMP): Offers optional support (--use_amp) for Automatic Mixed Precision, potentially accelerating training speed and reducing GPU memory consumption on compatible hardware (NVIDIA Tensor Cores).
Checkpointing: Periodically saves the model's state (state_dict), optimizer state, and training arguments (args) to .pth files, enabling training resumption or loading the model for later inference.
Training Previews: Generates and saves a preview image (training_preview.jpg) at regular intervals (--preview_interval), comparing source slices, the model's current predictions, and target slices from a fixed batch to visually monitor learning progress.
Configurable Hyperparameters: Uses command-line arguments (argparse) to allow easy configuration of crucial parameters like learning rate (--lr), batch size per GPU (--batch_size), number of epochs (--epochs), data paths, overlap, etc.

<div align="center">

# TUNET
Learn a pixel-wise mapping from source (src) images to destination (dst) images using U-Net — a more direct, suitable, and less complex approach. 
Supports both training and inference.


</div>




Features:
```
Multi-GPU training native
Up to 8K native plates
Multi-node for scale compute Ie: 8+ GPUs 
Control of overlap
```

✅ Install using Miniconda or Anaconda:
```
git clone https://github.com/tpc2233/tunet.git
cd tunet

conda create -n tunet python=3.8
conda activate tunet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnx onnxruntime Pillow
```

✅ How to use:
```
TRAINING:
cd to tunet folder
make sure conda activate tunet
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --src_dir /path/to/your/src --dst_dir /path/to/your/dst --output_dir /path/to/your/model --resolution 512 --overlap_factor 0.25 --log_interval 5 --save_interval 10 --epochs 5000 --batch_size 4


For model size use: --unet_hidden_size 128 (default=64)
--nproc_per_node=8 (use 8 GPUs)
--nnodes=1 (use one machine)
```


## Preview of the training:
```
left: Src Original
Middle: Dst GT
Right: Test Inference 
```

<img width="1554" alt="Screenshot 2025-04-03 at 10 15 42" src="https://github.com/user-attachments/assets/7b188b50-8414-48e5-8710-0ddb699a69e3" />

## Video:
[[video 🤗](https://youtu.be/UyMq0bsny-A)]



Inference:

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




## License

The source code is licensed under the Apache License, Version 2.0.
Permissions
Commercial use
Modification
Distribution
Patent use
Private use

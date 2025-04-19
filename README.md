<div align="center">

# TUNET
A pixel-wise mapping from source (src) images to destination (dst) images using U-Net — a more direct, suitable, and less complex approach. 
Supports both training and inference.


</div>






✅ Install using Miniconda or Anaconda:
###[[Install Video](https://youtu.be/QaAca_LiwKc))]

```
git clone --branch multi-gpu --single-branch https://github.com/tpc2233/tunet.git
cd tunet

conda create -n tunet python=3.8
conda activate tunet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnx pyyaml lpips onnxruntime Pillow albumentations
```

✅ How to use: 
###[[Training Video](https://youtu.be/gRwQRJPaX7U)] 
```
TRAINING:
1- Choose one of the config_templates for your project (simple or advanced)
2- Set the input plate paths 
You are good to go!
```

Training (Single or Multi-GPU)  
Run the training script. The number of GPUs is set via --nproc_per_node:
```
# Single GPU
torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py --config /path/to/your/config.yaml

# Multi-GPU (support: 2, 4, or 8 GPUs)
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py --config /path/to/your/config.yaml

```


Note:
batch_size is per GPU.  
For example, batch_size=2 with 8 GPUs results in an effective batch size of 16.



## Training screen:
```
left: Src Original plate
Middle: Dst Modified plate
Right: Model Inference 
```
![Screenshot 2025-04-06 at 18 37 45](https://github.com/user-attachments/assets/bc4ab4b4-d636-4b7b-9003-aaed1b213d02)




## Inference:
```
python inference.py --overlap_factor 0.25 --checkpoint /path/to/your/tunet_latest.pth --input_dir /path/to/your/plate --output_dir /path/to/output/folder
```

### Inference converters:

✅ Foundry NUKE Converter:
```
Convert the model to run native inside Nuke, CAT:

python utils/convert_nuke.py --generate_nk --checkpoint_pth /path/to/model/tunet_latest.pth --method script
```
<img width="899" alt="Screenshot 2025-04-06 at 19 31 15" src="https://github.com/user-attachments/assets/e8b4c620-93a3-4f50-8789-09f88326c2b6" />



✅ Autodesk FLAME Converter:
```
WIP TODO
Convert the model to run native inside Flame, ONNX:

python utils/convert_flame.py --checkpoint /path/to/model/tunet_latest.pth --use_gpu
```
<img width="893" alt="Screenshot 2025-04-06 at 19 28 21" src="https://github.com/user-attachments/assets/0eec9a04-eb3b-4e1a-94bb-b23f9d441690" />


## Video:
[[video 🤗](https://youtu.be/UyMq0bsny-A)]


### Citation

Consider cite TUNET in your project.
```
@article{tpo2025tunet,
  title={TuNet},
  author={Thiago Porto},
  year={2025}
}
```

## License

The source code is licensed under the Apache License, Version 2.0.
Commercial use Permission 


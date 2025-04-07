<div align="center">

# TUNET
A pixel-wise mapping from source (src) images to destination (dst) images using U-Net — a more direct, suitable, and less complex approach. 
Supports both training and inference.


</div>






✅ Install using Miniconda or Anaconda:
```
git clone https://github.com/tpc2233/tunet.git
cd tunet

conda create -n tunet python=3.8
conda activate tunet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnx pyyaml lpips onnxruntime Pillow
```

✅ How to use:
```
TRAINING:
Copy one of the config_templates to your project [simple or advanced]
input your plates and configuration
Run:
torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py --config /path/to/your/config.yaml
```


## Preview of the training:
```
left: Src Original plate
Middle: Dst Modified plate
Right: Model Inference 
```

<img width="1554" alt="Screenshot 2025-04-03 at 10 15 42" src="https://github.com/user-attachments/assets/7b188b50-8414-48e5-8710-0ddb699a69e3" />


✅ Foundry NUKE CAT Converter:
```
python utils/convert_nuke.py --generate_nk --checkpoint_pth /path/to/your/002/model//test_simple_2/tunet_latest.pth --method script
```
<img width="899" alt="Screenshot 2025-04-06 at 19 31 15" src="https://github.com/user-attachments/assets/e8b4c620-93a3-4f50-8789-09f88326c2b6" />



✅ Autodesk FLAME ONNX Converter:
```
WIP TODO
python utils/convert_flame.py --checkpoint /path/to/your/002/model/tunet_latest.pth --use_gpu
```
<img width="893" alt="Screenshot 2025-04-06 at 19 28 21" src="https://github.com/user-attachments/assets/0eec9a04-eb3b-4e1a-94bb-b23f9d441690" />


## Video:
[[video 🤗](https://youtu.be/UyMq0bsny-A)]



## License

The source code is licensed under the Apache License, Version 2.0.
Commercial use Permission 


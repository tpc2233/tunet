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
soon
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



✅ Autodesk FLAME ONNX Converter:
```
WIP TODO
python utils/convert_flame.py --checkpoint /path/to/your/002/model/tunet_latest.pth --use_gpu
```
<img width="893" alt="Screenshot 2025-04-06 at 19 28 20" src="https://github.com/user-attachments/assets/95c314f2-1ad5-4f15-ab25-e8eebc892c5b" />


✅ Foundry NUKE CAT Converter:
```
WIP TODO
python utils/convert_nuke.py --checkpoint_pth /path/to/your/002/model//test_simple_2/tunet_latest.pth --method script
```
<img width="899" alt="Screenshot 2025-04-06 at 19 31 15" src="https://github.com/user-attachments/assets/e8b4c620-93a3-4f50-8789-09f88326c2b6" />


## License

The source code is licensed under the Apache License, Version 2.0.
Permissions
Commercial use
Modification
Distribution
Patent use
Private use

## Statble Diffusion 3 + HyperSD + Controlnet Support with OpenVINO

### 1. Setup Python Environment
```bash
conda create -n sd3-controlnet-ov python=3.10
conda activate sd3-controlnet-ov
pip install -r requirements.txt
```

### 2. Download model
```bash
set HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download stabilityai/stable-diffusion-3-medium-diffusers --local-dir stabilityai/stable-diffusion-3-medium-diffusers

wget https://hf-mirror.com/ByteDance/Hyper-SD/resolve/main/Hyper-SD3-4steps-CFG-lora.safetensors -P Hyper-SD

huggingface-cli download --resume-download InstantX/SD3-Controlnet-Canny --local-dir InstantX/SD3-Controlnet-Canny
huggingface-cli download --resume-download InstantX/SD3-Controlnet-Pose --local-dir InstantX/SD3-Controlnet-Pose
huggingface-cli download --resume-download InstantX/SD3-Controlnet-Tile --local-dir InstantX/SD3-Controlnet-Tile
```

### 3. Convert SD3 + HyperSD + Controlnet Pytorch Model to OpenVINO Model
```bash
python convert_sd3_controlnet.py -m stabilityai/stable-diffusion-3-medium-diffusers -l Hyper-SD/Hyper-SD3-4steps-CFG-lora.safetensors -c InstantX 
```

### 4. Run Stable Diffusion 3 + HyperSD + Controlnet Canny with Inference OpenVINO
```bash
python sd3_controlnet_ov.py -m stable-diffusion-3-controlnet-ov --width 512 --height 512 --controlnet_type canny --image_path assets/canny.jpg --device GPU 
```

### 5. Run Stable Diffusion 3 + HyperSD + Controlnet Pose Inference with OpenVINO
```bash
python sd3_controlnet_ov.py -m stable-diffusion-3-controlnet-ov --width 512 --height 512 --controlnet_type pose --image_path assets/pose.jpg --device GPU
```

### 6. Run Stable Diffusion 3 + HyperSD + Controlnet Tile Inference with OpenVINO
```bash
python sd3_controlnet_ov.py -m stable-diffusion-3-controlnet-ov --width 512 --height 512 --controlnet_type tile --image_path assets/tile.jpg --device GPU
```

### (Optional) 7. Run Stable Diffusion 3 + HyperSD + Controlnet Canny Inference with Pytorch
```bash
python sd3_controlnet_pt.py --width 512 --height 512 --controlnet_path InstantX/SD3-Controlnet-Canny --image_path assets/canny.jpg
```

### (Optional) 8. Run Stable Diffusion 3 + HyperSD + Controlnet Pose Inference with Pytorch
```bash
python sd3_controlnet_pt.py --width 512 --height 512 --controlnet_path InstantX/SD3-Controlnet-Pose --image_path assets/pose.jpg
```

### (Optional) 9. Run Stable Diffusion 3 + HyperSD + Controlnet Tile Inference with Pytorch
```bash
python sd3_controlnet_pt.py --width 512 --height 512 --controlnet_path InstantX/SD3-Controlnet-Tile --image_path assets/tile.jpg
```

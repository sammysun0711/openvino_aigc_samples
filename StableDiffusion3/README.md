## Statble Diffusion 3 + HyperSD + Controlnet Support with OpenVINO

### 1. Setup Python Environment
```bash
conda create -n sd3-controlnet-ov python=3.10
conda activate sd3-controlnet-ov
pip install -r requirments.txt
```

### 2. Download model
```bash
set HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download stabilityai/stable-diffusion-3-medium-diffusers --local-dir stabilityai/stable-diffusion-3-medium-diffusers

wget https://hf-mirror.com/ByteDance/Hyper-SD/resolve/main/Hyper-SD3-4steps-CFG-lora.safetensors -P Hyper-SD

huggingface-cli download --resume-download InstantX/SD3-Controlnet-Canny --local-dir InstantX/SD3-Controlnet-Canny
huggingface-cli download --resume-download InstantX/SD3-Controlnet-Pose --local-dir InstantX/SD3-Controlnet-Canny
huggingface-cli download --resume-download InstantX/SD3-Controlnet-Tile --local-dir InstantX/SD3-Controlnet-Canny
```

### 3. Convert SD3 + HyperSD + Contorlnet Pytorch Model to OpenVINO Model
```bash
python convert_sd3_controlnet.py -m stabilityai/stable-diffusion-3-medium-diffusers --lora_path Hyper-SD/Hyper-SD3-4steps-CFG-lora.safetensors -c InstantX 
```

### 4.Stable Diffusion 3 + HyperSD + Controlnet Canny with OpenVINO
```bash
python run_sd3_controlnet_ov.py --controlnet_type canny --image_path assets/canny.jpg
```

### 5.Stable Diffusion 3 + HyperSD + Controlnet Pose with OpenVINO
```bash
python sd3_controlnet_ov.py --controlnet_type pose --image_path assets/pose.jpg
```

### 6.Stable Diffusion 3 + HyperSD + Controlnet Tile with OpenVINO
```bash
python sd3_controlnet_ov.py --controlnet_type tile --image_path assets/tile.jpg
```

### (optional) 7.Stable Diffusion 3 + HyperSD + Controlnet Canny with Pytorch
```bash
python sd3_controlnet_pt.py --controlnet_path InstantX/SD3-Controlnet-Canny --image_path assets/canny.jpg
```

### (optional) 8.Stable Diffusion 3 + HyperSD + Controlnet Pose with Pytorch
```bash
python sd3_controlnet_pt.py --controlnet_path InstantX/SD3-Controlnet-Pose --image_path assets/pose.jpg
```

### (optional) 9.Stable Diffusion 3 + HyperSD + Controlnet Tile with Pytorch
```bash
python sd3_controlnet_pt.py --controlnet_path InstantX/SD3-Controlnet-Tile --image_path assets/tile.jpg
```

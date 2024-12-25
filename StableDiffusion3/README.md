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
huggingface-cli download --resume-download stabilityai/stable-diffusion-3-medium-diffusers --local-dir stable-diffusion-3-medium-diffusers
huggingface-cli download --resume-download ByteDance/Hyper-SD --local-dir Hyper-SD
```

### 3.Stable Diffusion 3 + HyperSD + Controlnet Canny with OpenVINO
```bash
huggingface-cli download --resume-download InstantX/SD3-Controlnet-Canny --local-dir SD3-Controlnet-Canny
python sd3_controlnet.py -c SD3-Controlnet-Canny
```

### 4.Stable Diffusion 3 + HyperSD + Controlnet Pose with OpenVINO
```bash
huggingface-cli download --resume-download InstantX/SD3-Controlnet-Pose --local-dir SD3-Controlnet-Pose
python sd3_controlnet.py -c SD3-Controlnet-Pose
```

### 5.Stable Diffusion 3 + HyperSD + Controlnet Tile with OpenVINO
```bash
huggingface-cli download --resume-download InstantX/SD3-Controlnet-Tile --local-dir SD3-Controlnet-Tile
python sd3_controlnet.py -c SD3-Controlnet-Tile
```
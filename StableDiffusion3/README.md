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
huggingface-cli download --resume-download InstantX/SD3-Controlnet-Depth --local-dir SD3-Controlnet-Depth
```

### 3. Convert SD3 + HyperSD + Controlnet Pytorch Model to OpenVINO Model
```python
python sd3_controlnet.py
```

### 4. Run SD3 + HyperSD + Controlnet with OpenVINO runtime
```
python sd3_controlnet.py
```
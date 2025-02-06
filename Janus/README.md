# DeekSeek Janus-Pro Model Enabling with OpenVINO

### 1. Setup Environment
```bash
conda create janus-ov python=3.10
conda activate janus-ov
pip install -r requirements.txt
```
### 2. Download Janus Pytorch model
```bash
modelscope download --model deepseek-ai/Janus-Pro-1B --local_dir Janus-Pro-1B
```
 
### 3. Convert Pytorch model to OpenVINO model
```python
python convert_janus.py
```

### 4. Run Janus for multimodal understanding task with OpenVINO
```python
python run_janus_vlm_ov.py
```
### 5. Run Janus for text-to-image task with OpenVINO
```python
python run_janus_t2i_ov.py
```

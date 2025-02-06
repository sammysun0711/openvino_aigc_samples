# DeekSeek Janus-Pro Model Enabling with OpenVINO

### 1. Setup Environment
```bash
conda create -n janus-ov python=3.10
conda activate janus-ov
pip install -r requirements.txt
```
### 2. Download Janus Pytorch model
```bash
modelscope download --model deepseek-ai/Janus-Pro-1B --local_dir Janus-Pro-1B
```
 
### 3. Convert Pytorch model to OpenVINO model
```python
python convert_janus.py -m Janus-Pro-1B -o Janus-Pro-1B-OV
```

### 4. Run Janus for multimodal understanding task with OpenVINO
```python
 python run_janus_vlm_ov.py -m Janus-Pro-1B-OV -p "Describe image in details" -i cat_in_box.png -d CPU
```
### 5. Run Janus for text-to-image task with OpenVINO
```python
python run_janus_t2i_ov.py -m Janus-Pro-1B-OV -d CPU
```

# DeekSeek Janus-Pro Model Enabling with OpenVINO

### 0. Verified Model Lists
- [Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B)
- [Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B)
- [Janus-1.3B](https://huggingface.co/deepseek-ai/Janus-1.3B)

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
 
### 3. Convert Pytorch Model to OpenVINO Model
```python
python convert_janus.py -m Janus-Pro-1B -o Janus-Pro-1B-OV
```

### 4. Run Janus-Pro for Multimodal Understanding Task with OpenVINO
```python
python run_janus_vlm_ov.py -m Janus-Pro-1B-OV -p "Describe image in details" -i images/cat_in_box.png -d GPU
```

### 5. Run Janus-Pro for Text-to-Image Task with OpenVINO
```python
python run_janus_t2i_ov.py -m Janus-Pro-1B-OV -d GPU
```

### 6. Benchmark Janus-Pro for Multimodal Understanding Task with OpenVINO
```
python benchmark_janus_vlm_ov.py -m Janus-Pro-1B-OV/ -d GPU
```

### 7. Benchmark Janus-Pro for Text-to-Image Task with OpenVINO
```
python benchmark_janus_t2i_ov.py -m Janus-Pro-1B-OV/ -d GPU
```

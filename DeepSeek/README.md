# DeepSeek-R1 Distill Model Enabling with OpenVINO

### 0. Verfied models with prompts from DeepSeek [prompt-library](https://api-docs.deepseek.com/zh-cn/prompt-library):
- [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)


### 1. Setup Python Environment
```bash
conda create -n deepseek-ov python=3.10
conda activate deepseek-ov
pip install -r requirements.txt 
```

### 2. Download DeepSeek-R1 Distill Pytorch Model
```bash
modelscope download --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local_dir DeepSeek-R1-Distill-Qwen-1.5B
```

### 3. Convert Pytorch Model to OpenVINO FP16 Model
```bash
optimum-cli export openvino --model DeepSeek-R1-Distill-Qwen-1.5B --weight-format fp16 DeepSeek-R1-Distill-Qwen-1.5B-OV-FP16 --task text-generation-with-past --trust-remote-code
```

### 4. Convert Pytorch Model to OpenVINO INT4 Model
```bash
optimum-cli export openvino --model DeepSeek-R1-Distill-Qwen-1.5B --weight-format int4 --group-size 64 --ratio 1.0 DeepSeek-R1-Distill-Qwen-1.5B-OV-INT4 --task text-generation-with-past --trust-remote-code
```

### 5. Run inference with OpenVINO FP16 model with GPU
```python
python test_deepseek_ov.py -m DeepSeek-R1-Distill-Qwen-1.5B-OV-FP16 -d GPU
```

### 6. Run inference with OpenVINO INT4 model with GPU
```python
python test_deepseek_ov.py -m DeepSeek-R1-Distill-Qwen-1.5B-OV-INT4 -d GPU
```

# OpenVINO_AIGC_Samples
OpenVINO Samples for Popular AIGC Applications

## Setup Environment
```bash
conda create -n aigc python=3.10
conda activate aigc
pip install -r requirements.txt
```

## LlaMA
### 1. Convert Pytorch Model to OpenVINO Model
Convert Pytorch Model to OpenVINO FP32 Model
```python
python Tools/convert.py --model_id ziqingyang/chinese-alpaca-2-7b --output_dir LlaMA/chinese-alpaca-2-7b --precision FP32
```
Convert Pytorch Model to OpenVINO FP16 Model
```python
python Tools/convert.py --model_id ziqingyang/chinese-alpaca-2-7b --output_dir LlaMA/chinese-alpaca-2-7b --precision FP16
```
Convert Pytorch Model to OpenVINO INT8 Model with Weight Only Compression
```
python Tools/convert.py --model_id ziqingyang/chinese-alpaca-2-7b --output_dir LlaMA/chinese-alpaca-2-7b --precision FP16 --compress_weights
```
### 2. Run inference with LlaMA OpenVINO Model
Run LlaMA OpenVINO FP32 Model with CPU
```bash
cd LlaMA
python run_llm.py -c chinese-alpaca-2-7b/openvino/FP32 -p ”为什么北京是中国的首都？" -d CPU
```
Run LlaMA OpenVINO FP16 Model with GPU using Model Cache
```python
python run_llm.py -c chinese-alpaca-2-7b/openvino/FP16 -p ”为什么北京是中国的首都？" -d GPU --cache_dir model_cache
```
Run LlaMA OpenVINO INT8 Model with GPU using Model Cache
```python
python run_llm.py -c chinese-alpaca-2-7b/openvino/INT8 -p ”为什么北京是中国的首都？" -d GPU --cache_dir model_cache
```

## Stable Diffusion
### 1. Convert Pytorch Model to OpenVINO Model
Convert Pytorch Model to OpenVINO FP32 Model
```python
python Tools/convert.py --model_id runwayml/stable-diffusion-v1-5 --output_dir StableDiffusion/stable-diffusion-v1-5 --precision FP32
```
Convert Pytorch Model to OpenVINO FP16 Model
```python
python Tools/convert.py --model_id runwayml/stable-diffusion-v1-5 --output_dir StableDiffusion/stable-diffusion-v1-5 --precision FP16
```
Convert Pytorch Model to OpenVINO INT8 Model with Weight Only Compression
```
python Tools/convert.py --model_id runwayml/stable-diffusion-v1-5 --output_dir StableDiffusion/stable-diffusion-v1-5 --precision FP16 --compress_weights
```

### 2. Run inference with Stable Diffusion OpenVINO Model
Run Stable Diffusion OpenVINO FP32 Model with CPU
```bash
cd StableDiffusion
python run_sd.py -c stable-diffusion-v1-5/openvino/FP32 -p ”A cute cat" -d CPU
```
Run Stable Diffusion OpenVINO FP16 Model using Model Cache
```python
python run_sd.py -c stable-diffusion-v1-5/openvino/FP16 -p ”A cute cat" -d GPU --cache_dir model_cache
```
Run Stable Diffusion OpenVINO INT8 Model using Model Cache
```python
python run_sd.py -c stable-diffusion-v1-5/openvino/INT8 -p ”A cute cat" -d GPU --cache_dir model_cache
```

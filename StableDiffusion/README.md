# Stable Diffusion
Here is the example for Stable Diffusion model conversion and inference with OpenVINO runtime.

## 1. Setup Environment
```bash
conda create -n aigc python=3.10
conda activate aigc
pip install -r ../requirements.txt
sudo apt-get install git-lfs
```

## 2. Donwload Pytorch Model with GIT LFS
```bash
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

## 3. Convert Pytorch Model to OpenVINO Model
Convert Pytorch Model to OpenVINO FP32 Model
```python
python ../Tools/convert.py --model_id stable-diffusion-v1-5 \
    --output_dir stable-diffusion-v1-5-ov --precision FP32
```
Convert Pytorch Model to OpenVINO FP16 Model
```python
python ../Tools/convert.py --model_id stable-diffusion-v1-5 \
    --output_dir stable-diffusion-v1-5-ov --precision FP16
```
Convert Pytorch Model to OpenVINO INT8 Model with Weight Only Compression
```python
python ../Tools/convert.py --model_id stable-diffusion-v1-5 \
    --output_dir stable-diffusion-v1-5-ov --precision FP16 --compress_weights
```

## 4. Run inference with Stable Diffusion OpenVINO Model
Run Stable Diffusion OpenVINO FP32 Model on Intel CPU
```python
python run_sd.py -c stable-diffusion-v1-5-ov/FP32 -p "A cute cat" -d CPU
```
Run Stable Diffusion OpenVINO FP16 Model on Intel iGPU with static shape using model cache 
```python
python run_sd.py -c stable-diffusion-v1-5-ov/FP16 -p "A cute cat" -d GPU.0 \
     --static_shape --cache_dir model_cache
```
Run Stable Diffusion OpenVINO INT8 Model on Intel dGPU with static shape using Model Cache
```python
python run_sd.py -c stable-diffusion-v1-5-ov/INT8 -p "A cute cat" -d GPU.1 \
    --static_shape --cache_dir model_cache
```

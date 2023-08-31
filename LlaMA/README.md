# LlaMA
Here is the example for LlaMA model conversion and inference with OpenVINO runtime.

## 1. Setup Environment
```bash
conda create -n aigc python=3.10
conda activate aigc
pip install -r ../requirements.txt
sudo apt-get install git-lfs
```
## 2. Donwload Pytorch Model with GIT LFS
```bash
git clone https://huggingface.co/ziqingyang/chinese-alpaca-2-7b
```

## 3. Convert Pytorch Model to OpenVINO Model
Convert Pytorch Model to OpenVINO FP32 Model
```python
python ../Tools/convert.py --model_id chinese-alpaca-2-7b \
    --output_dir chinese-alpaca-2-7b-ov --precision FP32
```
Convert Pytorch Model to OpenVINO FP16 Model
```python
python ../Tools/convert.py --model_id chinese-alpaca-2-7b \
    --output_dir chinese-alpaca-2-7b-ov --precision FP16
```
Convert Pytorch Model to OpenVINO INT8 Model with Weight Only Compression
```python
python ../Tools/convert.py --model_id chinese-alpaca-2-7b \
    --output_dir chinese-alpaca-2-7b-ov --precision FP16 --compress_weights
```

## 4. Run inference with LlaMA OpenVINO Model
Run LlaMA OpenVINO FP32 Model on Intel CPU
```python
python run_llm.py -c chinese-alpaca-2-7b-ov/FP32 \
    -p "为什么北京是中国的首都？" -d CPU
```
Run LlaMA OpenVINO FP16 Model on Intel iGPU using Model Cache
```python
python run_llm.py -c chinese-alpaca-2-7b-ov/FP16 \
    -p "为什么北京是中国的首都？" -d GPU.0 --cache_dir model_cache
```
Run LlaMA OpenVINO INT8 Model on Intel dGPU using Model Cache
```python
python run_llm.py -c chinese-alpaca-2-7b-ov/INT8 \
    -p "为什么北京是中国的首都？" -d GPU.1 --cache_dir model_cache
```

## 5. Run benchmark with LlaMA OpenVINO Model
Run benchmark with LlaMA OpenVINO FP32 Model on CPU
```python
python benchmark_llm.py -c chinese-alpaca-2-7b-ov/FP32 -d CPU \
    -ps prompt_set/prompt_causallm_cn.json --cache_dir model_cache
```

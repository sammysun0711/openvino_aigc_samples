# LlaMA
Here is the example for LlaMA model conversion and inference with OpenVINO runtime.

## 1. Setup Environment
```bash
conda create -n aigc python=3.10
conda activate aigc
pip install -r ../requirements.txt
```

## 2. Convert Pytorch Model to OpenVINO Model
Convert Pytorch Model to OpenVINO FP32 Model
```python
python ../Tools/convert.py --model_id ziqingyang/chinese-alpaca-2-7b --output_dir LlaMA/chinese-alpaca-2-7b --precision FP32
```
Convert Pytorch Model to OpenVINO FP16 Model
```python
python ../Tools/convert.py --model_id ziqingyang/chinese-alpaca-2-7b --output_dir LlaMA/chinese-alpaca-2-7b --precision FP16
```
Convert Pytorch Model to OpenVINO INT8 Model with Weight Only Compression
```python
python ../Tools/convert.py --model_id ziqingyang/chinese-alpaca-2-7b --output_dir LlaMA/chinese-alpaca-2-7b --precision FP16 --compress_weights
```

## 3. Run inference with LlaMA OpenVINO Model
Run LlaMA OpenVINO FP32 Model on CPU
```python
python run_llm.py -c chinese-alpaca-2-7b/openvino/FP32 -p ”为什么北京是中国的首都？" -d CPU
```
Run LlaMA OpenVINO FP16 Model on GPU using Model Cache
```python
python run_llm.py -c chinese-alpaca-2-7b/openvino/FP16 -p ”为什么北京是中国的首都？" -d GPU --cache_dir model_cache
```
Run LlaMA OpenVINO INT8 Model on GPU using Model Cache
```python
python run_llm.py -c chinese-alpaca-2-7b/openvino/INT8 -p ”为什么北京是中国的首都？" -d GPU --cache_dir model_cache
```

## 4. Run benchmark with LlaMA OpenVINO Model
Run benchmark with LlaMA OpenVINO FP32 Model on CPU
```python
python benchmark_llm.py -c chinese-alpaca-2-7b/openvino/FP32 -d CPU \
    -ps prompt_set/prompt_causallm_cn.json --cache_dir model_cache
```

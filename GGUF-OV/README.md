# Run GGUF Model Inference with OpenVINO GenAI On-the-Fly

This sample aims to show how to run GGUF model inference from [llama.cpp](https://github.com/ggml-org/llama.cpp) with [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) on-the-fly, this method provides following benefits: 

- Direct GGUF Compressed Model supported (Q4_K_M/Q4_0/Q8_0/F16) without GGUF Model->Pytorch Model->OpenVINO IR conversion, which can significate reduce model conversion time and memory usage.
- Smoothless leverage Large Language Model (LLM) pipeline optimization across intel xPU (CPU/GPU/NPU) platform (MTL/LNL/ARL and beyond).

### 0. Verfied models:
- [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
- [Qwen/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF)
- [Qwen/Qwen2.5-3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF)
- [Qwen/Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF)


### 1. Setup Python Environment
```bash
conda create -n deepseek-ov python=3.10
conda activate gguf-ov
pip install -r requirements.txt 
```

### 2. Download GGUF model and convert OpenVINO Tokenizer models
```bash
python run_gguf_with_ov_genai.py -m Qwen/Qwen2.5-1B-Instruct-GGUF -f qwen2.5-1b-instruct-q4_k_m.gguf -d GPU -o gguf_models -p "What is OpenVINO?" -cd "model_cache"
```
```
Download GGUF model from model_id: Qwen/Qwen2.5-1.5B-Instruct-GGUF, filename: qwen2.5-1.5b-instruct-q4_k_m.gguf to gguf_models ...
qwen2.5-1.5b-instruct-q4_k_m.gguf: 100%|███████████████████████████████████████████████████| 1.12G/1.12G [00:33<00:00, 33.6MB/s]
Download GGUF model finished, elapsed: 35554.653 ms

Convert OpenVINO tokenizer from GGUF model: Qwen/Qwen2.5-1.5B-Instruct-GGUF, filename: qwen2.5-1.5b-instruct-q4_k_m.gguf to gguf_models ...
Convert OpenVINO tokenizer finished, elapsed: 36055.765 ms
Loading and unpacking model from: gguf_models\qwen2.5-1.5b-instruct-q4_k_m.gguf
Loading and unpacking model done. Time: 5859ms
Start generating OV model...
Model generation done. Time: 721ms

Prompt: Who are you?
Start generation ...

Response:
I am Qwen, a large language model created by Alibaba Cloud. I am designed to assist with a wide range of tasks and provide helpful responses to your queries. How can I assist you today?
```
### 3. Run inference with GGUF with OpenVINO GenAI with GPU on-the-fly
```bash
python run_gguf_with_ov_genai.py -m Qwen/Qwen2.5-1.5B-Instruct-GGUF -f qwen2.5-1.5b-instruct-q4_0.gguf -d GPU -o gguf_models -p "What is OpenVINO?" -cd "model_cache"
```
```
Found GGUF model in path: gguf_models\qwen2.5-1.5b-instruct-q4_k_m.gguf, skip downloading.
Found converted OpenVINO tokenizer models in path: gguf_models\openvino_tokenizer.xml.
Loading and unpacking model from: gguf_models\qwen2.5-1.5b-instruct-q4_k_m.gguf
Loading and unpacking model done. Time: 5604ms
Start generating OV model...
Model generation done. Time: 728ms

Prompt: Who are you?
Start generation ...

Response:
I am Qwen, a large language model created by Alibaba Cloud. I am designed to assist with a wide range of tasks and provide helpful responses to your queries. How can I assist you today?
```

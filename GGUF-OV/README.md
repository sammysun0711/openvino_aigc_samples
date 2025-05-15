# Run GGUF Model Inference with OpenVINO GenAI via GGUF Reader On-the-Fly

This sample aims to show how to run GGUF model inference from [llama.cpp](https://github.com/ggml-org/llama.cpp) with [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) on-the-fly, this method provides following benefits: 

- Direct GGUF Compressed Model supported (Q4_K_M/Q4_0/F16) without requied GGUF->Pytorch->OpenVINO conversion, which takes long process time and significant memory usage
- Smoothless leverage OpenVINO GenAI large language model pipeline optimization across intel xPU (CPU/GPU/NPU) platform (MTL/LNL/ARL and beyond)

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

### 2. Download GGUF model and converted OpenVINO Tokenizer models
```bash
python run_gguf_with_ov_genai.py -m Qwen/Qwen2.5-0.5B-Instruct-GGUF -f qwen2.5-0.5b-instruct-q4_0.gguf -d CPU -o gguf_models -p "Who are you?" -cd "model_cache" -mnt 128
```
Download GGUF model from model_id: Qwen/Qwen2.5-0.5B-Instruct-GGUF, filename: qwen2.5-0.5b-instruct-q4_0.gguf to gguf_models ...
Download GGUF model finished, elapsed: 1654.478 ms

Convert OpenVINO tokenizer from GGUF model: Qwen/Qwen2.5-0.5B-Instruct-GGUF, filename: qwen2.5-0.5b-instruct-q4_0.gguf to gguf_models ...
Convert OpenVINO tokenizer finished, elapsed: 17003.100 ms
Loading and unpacking model from: gguf_models/qwen2.5-0.5b-instruct-q4_0.gguf
Loading and unpacking model done. Time: 261ms
Start generating OV model...
Model generation done. Time: 314ms

Prompt:  Who are you?
Start generation ...

Response:  I am Qwen, a large language model created by Alibaba Cloud. I am a language model designed to assist users in generating human-like text, such as writing articles, stories, and even writing books. I am trained on a vast corpus of text data, including books, articles, and other written works. I am also trained on a large corpus of human language data, including written and spoken language. I am designed to provide information and insights to users, and to assist them in their tasks and goals. I am a tool that can be used to help users with their needs and to provide them with a better understanding of the world.

```

```
### 3. Run inference with GGUF with OpenVINO GenAI on-the-fly
```bash
python run_gguf_with_ov_genai.py -m Qwen/Qwen2.5-0.5B-Instruct-GGUF -f qwen2.5-0.5b-instruct-q4_0.gguf -d CPU -o gguf_models -p "Who are you?" -cd "model_cache" -mnt 128
```
```
Found GGUF model in path: gguf_models/qwen2.5-0.5b-instruct-q4_0.gguf, skip downloading.
Found converted OpenVINO tokenizer models in path: gguf_models/openvino_tokenizer.xml.
Loading and unpacking model from: gguf_models/qwen2.5-0.5b-instruct-q4_0.gguf
Loading and unpacking model done. Time: 209ms
Start generating OV model...
Model generation done. Time: 298ms

Prompt:  Who are you?
Start generation ...

Response:  I am Qwen, a large language model created by Alibaba Cloud. I am a language model designed to assist users in generating human-like text, such as writing articles, stories, and even writing books. I am trained on a vast corpus of text data, including books, articles, and other written works. I am also trained on a large corpus of internet text, including web pages, blogs, and other written content. I am designed to provide information and insights to users, and to help users with their queries and tasks. I am a language model that can understand and generate human-like text, and can be used to create content,

```

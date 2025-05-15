import argparse
from pathlib import Path
import time

from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

import openvino as ov
import openvino_genai
from openvino_tokenizers import convert_tokenizer

chat_template = {
    "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"}

def streamer(subword):
    print(subword, end='', flush=True)
    # Return flag corresponds whether generation should be stopped.
    return openvino_genai.StreamingStatus.RUNNING

def convert_ov_tokenizers(output_dir, gguf_file):
    hf_tokenizer = AutoTokenizer.from_pretrained(
        output_dir, gguf_file=gguf_file, trust_remote_code=True)
    # hf_tokenizer_chat_template = hf_tokenizer.get_chat_template()
    # Qwen2.5 GGUF model stored chat_template is different from HF tokenizers_config.json, replace as chat_tempalte with HF tokenizers_config
    ov_tokenizer, ov_detokenizer = convert_tokenizer(
        hf_tokenizer, with_detokenizer=True)
    ov_tokenizer.set_rt_info(chat_template, "chat_template")

    ov.save_model(ov_tokenizer, output_dir / "openvino_tokenizer.xml")
    ov.save_model(ov_detokenizer, output_dir / "openvino_detokenizer.xml")


def main():
    parser = argparse.ArgumentParser(
        description="Run GGUF model inference with OpenVINO GenAI")
    parser.add_argument('-m', '--model_id', type=str, default="Qwen/Qwen2.5-3B-Instruct-GGUF", required=False,
                        help="Specify path to Hugging Face checkpoint. Default is 'Qwen/Qwen2.5-3B-Instruct-GGUF'")
    parser.add_argument('-f', '--filename', type=str, default="qwen2.5-3b-instruct-q4_k_m.gguf", required=False,
                        help="Specify path to Hugging Face checkpoint. Default is 'qwen2.5-3b-instruct-q4_k_m.gguf'")
    parser.add_argument('-d', '--device', type=str, default="CPU", required=False,
                        help="Specify device used for OpenVINO inference.")
    parser.add_argument('-p', '--prompt', type=str, default="Who are you?", required=False,
                        help="Specify input prompt. Default is 'Who are you?'")
    parser.add_argument('-o', '--output_dir', type=str, default="gguf_models", required=False,
                        help="Specify save directory for GGUF models and OpenVINO tokenizers models, default is 'gguf_models'.")
    parser.add_argument("-mnt", "--max_new_tokens", default=40, type=int, required=False,
                        help="Specify maximum generated tokens counter, default is 40")
    parser.add_argument('-cd', '--cache_dir', type=str, default="model_cache", required=False,
                        help="Specify save directory of OpenVINO model cache, model cache can be disable by setting ''.")
    args = parser.parse_args()

    print("Arguments: ", args)

    output_dir = Path(args.output_dir)
    gguf_path = output_dir / args.filename
    ov_tokenizer_path = output_dir / "openvino_tokenizer.xml"

    if not gguf_path.exists():
        print(
            f"\nDownload GGUF model from model_id: {args.model_id}, filename: {args.filename} to {output_dir} ...")
        start = time.perf_counter()
        gguf_path = hf_hub_download(
            repo_id=args.model_id, filename=args.filename, local_dir=output_dir)
        end = time.perf_counter()
        print(
            f"Download GGUF model finished, elapsed: {(end - start) *1e3:.3f} ms")
    else:
        print(f"\nFound GGUF model in path: {gguf_path}, skip downloading.")

    if not ov_tokenizer_path.exists():
        print(
            f"\nConvert OpenVINO tokenizer from GGUF model: {args.model_id}, filename: {args.filename} to {output_dir} ...")
        start = time.perf_counter()
        convert_ov_tokenizers(output_dir, args.filename)
        end = time.perf_counter()
        print(
            f"Convert OpenVINO tokenizer finished, elapsed: {(end - start) *1e3:.3f} ms")
    else:
        print(
            (f"Found converted OpenVINO tokenizer models in path: {ov_tokenizer_path}."))

    tokenizer = openvino_genai.Tokenizer(output_dir)
    pipe = openvino_genai.LLMPipeline(gguf_path, tokenizer, args.device)

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens
    pipe.start_chat()
    print("\nPrompt: ", args.prompt)
    print("Start generation ...")
    print("\nResponse: \n")
    pipe.generate(args.prompt, config, streamer)
    pipe.finish_chat()


if '__main__' == __name__:
    main()

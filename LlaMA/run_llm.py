import argparse
import time
from pathlib import Path
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""
prompt_template = ("[INST] <<SYS>>\n"
                  "{system_prompt}\n"
                  "<</SYS>>\n\n"
                  "{instruction} [/INST]")

def question_answer(args):
    print("Input text: ", args.prompt)
    instruction = prompt_template.format_map({'instruction': args.prompt,'system_prompt':DEFAULT_SYSTEM_PROMPT})
    inputs = tokenizer(instruction, return_tensors="pt")
    gen_sequence_start = time.time()
    print("Start generate sequence ...")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    output_ids = ov_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=args.max_new_tokens)
    gen_sequence_end = time.time()
    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    gen_sequence_length = len(output_ids[0]) - len(input_ids[0])
    gen_latency = gen_sequence_end - gen_sequence_start
    if gen_sequence_length > 0:
        gen_latency_per_token = 1000.0 * gen_latency / gen_sequence_length
    else: 
        gen_latency_per_token = 0

    print(f"Generation {gen_sequence_length} tokens took {gen_latency:.3f} s")
    print(f"Generation latency per token took {gen_latency_per_token:.3f} ms")
    print(f"Predicted Sequence: {output_text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('-c', '--checkpoint', type=str, default="llama-7b", required=False,
                         help="Specify path to Hugging Face checkpoint. Default is 'llama-7b'")
    parser.add_argument('-d', '--device', type=str, default="CPU", required=False,
                         help="Specify device used for OpenVINO inference.")
    parser.add_argument('-p','--prompt', type=str, default="What is OpenVINO?", required=False,
                         help="Specify input prompt. Default is 'What is OpenVINO'")  
    parser.add_argument('-mnt', '--max_new_tokens', type=int, default=128, required=False,
                         help="Specify maximum number of new tokens to be generated.")
    parser.add_argument('-cd', '--cache_dir', type=str, default="model_cache", required=False,
                         help="Specify save directory of OpenVINO model cache, model cache can be disable by setting ''")
    # Parse the argument
    args = parser.parse_args()

    # initialize model and tokenizer
    model_init_start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    print("Init OpenVINO model ...")
    model_path = Path(args.checkpoint + "/openvino_model.xml")
    if model_path.exists():
        ov_model = OVModelForCausalLM.from_pretrained(args.checkpoint, 
                                                      export=False, 
                                                      compile=False,
                                                      ov_config={"CACHE_DIR":args.cache_dir})
        if "GPU" in args.device:
            ov_model.half()
        ov_model.to(args.device.lower())
        ov_model.compile()
    else:
        print("Error! Please convert OpenVINO model with Tools/convert.py at first!")
    print("Load OpenVINO model in device {} finished".format(ov_model._device))
    model_init_end = time.time()
    model_init_latency = model_init_end - model_init_start 
    print(f"Read/Compile model took {model_init_latency:.3f} s")
    question_answer(args)

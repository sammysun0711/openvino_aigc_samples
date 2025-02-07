from pathlib import Path
import time
import argparse
from janus.models import VLChatProcessor
from PIL import Image
from transformers import TextStreamer
from janus.utils.io import load_pil_images
from memory_profile import MemConsumption

from ov_janus_helper import OVJanusModel, ChunkStreamer

parser = argparse.ArgumentParser(
    "Benchmark TooL of DeepSeek Janus-Pro OpenVINO Inference for Multimodal Understanding Task",
    add_help=True,
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "-m",
    "--model_id",
    type=str,
    default="Janus-Pro-1B-OV",
    help="Model folder including Janus-Pro OpenVINO Models",
)

parser.add_argument("-d", "--device", default="CPU", type=str, help="Inference device")
parser.add_argument(
    "-cd",
    "--cache_dir",
    default="model_cache",
    type=str,
    help="Folder to save model cache",
)

parser.add_argument(
    "-p",
    "--prompt",
    default="Describe image in details",
    type=str,
    help="Input prompt for multimodal understanding",
)

parser.add_argument(
    "-i",
    "--image_path",
    type=str,
    default="images/cat_in_box.png",
    help="Path to input image",
)

parser.add_argument(
    "-mnt",
    "--max_new_tokens",
    default=256,
    type=int,
    help="Specify maximum generated tokens counter",
)

parser.add_argument(
    "-niter",
    "--num_interation",
    default=5,
    type=int,
    help="Specify number of interation for infernence",
)

parser.add_argument(
    "--use_chunk_streamer",
    action='store_true',
    help="Whether to use chunk streaming",
)

parser.add_argument(
    "--token_length",
    default=10,
    type=int,
    help="Specify token length for chunk streaming ",
)
args = parser.parse_args()

model_id = args.model_id
device = args.device
cache_dir = args.cache_dir
input_prompt = args.prompt
image_path = Path(args.image_path)
max_new_tokens = args.max_new_tokens
num_interation = args.num_interation

ov_config = {"CACHE_DIR": cache_dir}
start = time.time()
llm_times = []
input_token_length = []

ov_model = OVJanusModel(
    model_id, device=device, ov_config=ov_config, llm_times=llm_times
)

pipe_init_duration = time.time() - start
print(f"Init Pipeline took: {pipe_init_duration:.3f} s")

processor = VLChatProcessor.from_pretrained(model_id, trust_remote_code=True)

# Multimodal understanding
image = Image.open(image_path)

conversation = [
    {
        "role": "User",
        "content": f"<image_placeholder>{input_prompt}\n",
        "images": [str(image_path)],
    },
    {"role": "Assistant", "content": ""},
]
pil_images = load_pil_images(conversation)

prepare_inputs = processor(
    conversations=conversation, images=pil_images, force_batchify=True
)


# run image encoder to get the image embeddings
inputs_embeds = ov_model.prepare_inputs_embeds(**prepare_inputs)
input_token_length = inputs_embeds.shape[1]

if args.use_chunk_streamer:
    streamer = ChunkStreamer(processor.tokenizer, skip_prompt=True, tokens_len=args.token_length, skip_special_tokens=True)
else:
    streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)


print(f"Question:\n{input_prompt}")
first_token_t = []
avg_token_t = []
max_rss_mem, max_shared_mem, max_uss_mem = None, None, None
pipeline_latency = []
max_rss_mem_list = []
output_token_lengths = []
mem_consumption = MemConsumption()
mem_consumption.start_collect_mem_consumption_thread()

for i in range(num_interation):
    mem_consumption.start_collect_memory_consumption()
    print("== Genereate output: ")
    start_time = time.time()
    answer_token_ids = ov_model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=processor.tokenizer.eos_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        max_new_tokens=128,
        do_sample=False,
        streamer=streamer,
    )

    end_time = time.time()
    pipeline_latency.append((end_time - start_time))
    mem_consumption.end_collect_momory_consumption()

    print("\n--------------------------------------------")
    if len(llm_times) > 1:
        first_token_t.append(llm_times[0])
        avg_token = sum(llm_times[1:]) / (len(llm_times) - 1)
        avg_token_t.append(avg_token)
        output_token_lengths.append(len(llm_times))
        print(f"First input token size: ", input_token_length)
        print(
            f"VLM Model First token latency: {llm_times[0]:.2f} ms, Output len: {len(llm_times)}, Average 2nd+ token latency: {avg_token:.2f} ms"
        )
        max_rss_mem, max_shared_mem, max_uss_mem = (
            mem_consumption.get_max_memory_consumption()
        )
        max_rss_mem_list.append(max_rss_mem)
        print("max_rss_mem: {:.2f} MB".format(max_rss_mem))
        mem_consumption.clear_max_memory_consumption()

print("--------------------------------------------")
print("")
print(f"== Performance metrics from {num_interation} times run: ")
print(f"Pipeline intialization time: {pipe_init_duration:.3f} s")
print(f"First input token size: ", input_token_length)
print(f"Generated output token size: ", output_token_lengths[-1])
avg_token_ft = sum(first_token_t) / len(first_token_t)
print(f"Average VLM first token latency: {avg_token_ft:.2f} ms")
avg_token_av = sum(avg_token_t) / len(avg_token_t)
print(f"Average VLM 2nd+ token latency: {avg_token_av:.2f} ms")
avg_token_rate = 1000 / avg_token_av
print(f"Average VLM token rate: {avg_token_rate:.2f} t/s")
print(f"Max RSS Memory Usage: {sum(max_rss_mem_list)/num_interation:.2f} MB")
print(
    f"Average E2E pipeline inference time of {num_interation} iteration: {sum(pipeline_latency)/num_interation:.2f} s"
)
print("")
print("--------------------------------------------")

mem_consumption.end_collect_mem_consumption_thread()

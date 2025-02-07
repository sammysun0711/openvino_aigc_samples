from pathlib import Path
import time
import argparse
from janus.models import VLChatProcessor
from ov_janus_helper import OVJanusModel, generate_image
from memory_profile import MemConsumption

parser = argparse.ArgumentParser(
    "Benchmark Tools of DeepSeek Janus-Pro OpenVINO Inference for Text-to-Image Generation Task",
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
    default="A cute and adorable baby fox with big brown eyes, autumn leaves in the background enchanting,immortal,fluffy, shiny mane,Petals,fairyism,unreal engine 5 and Octane Render,highly detailed, photorealistic, cinematic, natural colors.",
    type=str,
    help="Input prompt for text-to-image generation",
)

parser.add_argument(
    "-o",
    "--save_path",
    type=str,
    default="generated_images.png",
    help="Path to save generated image",
)

parser.add_argument(
    "-niter",
    "--num_interation",
    default=5,
    type=int,
    help="Specify number of interation for infernence",
)

args = parser.parse_args()

model_id = args.model_id
save_path = args.save_path
device = args.device
cache_dir = args.cache_dir
image_gen_prompt = args.prompt
num_interation = args.num_interation

ov_config = {"CACHE_DIR": cache_dir}
print("Inference device: ", device)
input_token_lengths = []
max_rss_mem_list = []
pipeline_latency = []
img_size = 384

start = time.time()
ov_model = OVJanusModel(model_id, device=device, ov_config=ov_config)
pipe_init_duration = time.time() - start
print(f"Init Pipeline took: {pipe_init_duration:.3f} s")
processor = VLChatProcessor.from_pretrained(model_id, trust_remote_code=True)
print("Prompt: ", image_gen_prompt)
# Text to image generation
print("Image generation ...")

mem_consumption = MemConsumption()
mem_consumption.start_collect_mem_consumption_thread()
for i in range(num_interation):
    mem_consumption.start_collect_memory_consumption()
    print("\n--------------------------------------------")
    start = time.time()
    images = generate_image(
        ov_model,
        processor,
        image_gen_prompt,
        output_dir=None,
        parallel_size=1,
        img_size=img_size,
    )
    generation_duration = time.time() - start
    input_token_lengths.append(ov_model.language_model.input_token_length)
    pipeline_latency.append(generation_duration)
    
    mem_consumption.end_collect_momory_consumption()
    max_rss_mem, max_shared_mem, max_uss_mem = (
            mem_consumption.get_max_memory_consumption()
        )
    max_rss_mem_list.append(max_rss_mem)
    mem_consumption.clear_max_memory_consumption()

    print(f"First input token size: ", input_token_lengths[-1])
    print(f"E2E pipeline for {img_size}x{img_size} Image generation took: {generation_duration:.3f} s")
    print("Max RSS Memory Usage: {:.2f} MB".format(max_rss_mem))

print(f"Save generated image in path: {save_path}")
images[0].save(save_path)

print("--------------------------------------------")
print("")
print(f"== Performance metrics from {num_interation} times run: ")
print(f"Pipeline intialization time: {pipe_init_duration:.3f} s")
print(f"First input token size: ", input_token_lengths[0])
print(f"Max RSS Memory Usage: {sum(max_rss_mem_list)/num_interation:.2f} MB")
print(
    f"Average E2E pipeline for {img_size}x{img_size} image generation of {num_interation} iteration took: {sum(pipeline_latency)/num_interation:.2f} s"
)
print("")
print("--------------------------------------------")

mem_consumption.end_collect_mem_consumption_thread()

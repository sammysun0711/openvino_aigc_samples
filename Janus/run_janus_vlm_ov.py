from pathlib import Path
import argparse
from janus.models import VLChatProcessor
from PIL import Image
from transformers import TextStreamer

from optimum.intel.openvino import OVModelForVisualCausalLM

parser = argparse.ArgumentParser(
    "DeepSeek Janus-Pro OpenVINO Inference for Multimodal Understanding Task",
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

parser.add_argument(
        "-d", "--device", default="CPU", type=str, help="Inference device"
    )
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
    default="cat_in_box.png",
    help="Path to input image",
)

parser.add_argument(
    "-mnt",
    "--max_new_tokens",
    default=100,
    type=int,
    help="Specify maximum generated tokens counter",
)

args = parser.parse_args()

model_id = args.model_id
device = args.device
cache_dir = args.cache_dir
input_prompt = args.prompt
image_path = Path(args.image_path)
max_new_tokens = args.max_new_tokens

ov_config = {"CACHE_DIR": cache_dir}
model = OVModelForVisualCausalLM.from_pretrained(model_id, comiple=False, trust_remote_code=True, ov_config=ov_config)
model.to(device)
model.compile()

processor = VLChatProcessor.from_pretrained(model_id, trust_remote_code=True)

# Multimodal understanding
image = Image.open(image_path)

print(f"Prompt: {input_prompt}\n")
inputs = model.preprocess_inputs(input_prompt, image, processor)

print(f"Response: \n")
streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

model.generate(**inputs, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=False)
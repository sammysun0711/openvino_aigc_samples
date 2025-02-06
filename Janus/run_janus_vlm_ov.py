from io import BytesIO
from pathlib import Path

import requests
from janus.models import VLChatProcessor
from PIL import Image
from transformers import TextStreamer

from optimum.intel.openvino import OVModelForVisualCausalLM

model_id = "Janus-Pro-1B-OV"

model = OVModelForVisualCausalLM.from_pretrained(model_id, trust_remote_code=True)
processor = VLChatProcessor.from_pretrained(model_id, trust_remote_code=True)

# Multimodal understanding
input_prompt = "Describe image in details"
image_path = Path("cat_in_box.png")

image = Image.open(image_path)

print(f"Prompt: {input_prompt}\n")
inputs = model.preprocess_inputs(input_prompt, image, processor)

print(f"Response: \n")
streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

model.generate(**inputs, streamer=streamer, max_new_tokens=100, do_sample=False)

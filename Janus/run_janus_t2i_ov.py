from io import BytesIO
from pathlib import Path

import requests
from janus.models import VLChatProcessor
from PIL import Image
from transformers import TextStreamer

from optimum.intel.openvino import OVModelForVisualCausalLM

model_id = "Janus-Pro-1B-OV"
save_path = "fox.png"
model = OVModelForVisualCausalLM.from_pretrained(model_id, trust_remote_code=True)
processor = VLChatProcessor.from_pretrained(model_id, trust_remote_code=True)

# Text to image generation

image_gen_prompt = "A cute and adorable baby fox with big brown eyes, autumn leaves in the background enchanting,immortal,fluffy, shiny mane,Petals,fairyism,unreal engine 5 and Octane Render,highly detailed, photorealistic, cinematic, natural colors."
print("Image generation ...")
images = model.generate_image(processor, image_gen_prompt, parallel_size=1)

print(f"Save generate image in {save_path}")
images[0].save(save_path)

from io import BytesIO
from pathlib import Path

import requests
from janus.models import VLChatProcessor
from PIL import Image
from transformers import TextStreamer, AutoTokenizer

from optimum.intel.openvino import OVModelForVisualCausalLM

model_id = "Janus-Pro-1B"
save_dir = "Janus-Pro-1B-OV"


model = OVModelForVisualCausalLM.from_pretrained(model_id, trust_remote_code=True)
model.save_pretrained(save_dir)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.save_pretrained(save_dir)

processor = VLChatProcessor.from_pretrained(model_id, trust_remote_code=True)
processor.save_pretrained(save_dir)

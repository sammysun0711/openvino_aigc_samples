from pathlib import Path
import argparse
from janus.models import VLChatProcessor
from optimum.intel.openvino import OVModelForVisualCausalLM

parser = argparse.ArgumentParser(
    "DeepSeek Janus-Pro OpenVINO Inference for Text-to-Image Generation Task",
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
    default="A cute and adorable baby fox with big brown eyes, autumn leaves in the background enchanting,immortal,fluffy, shiny mane,Petals,fairyism,unreal engine 5 and Octane Render,highly detailed, photorealistic, cinematic, natural colors.",
    type=str,
    help="Input prompt for text-to-image generation",
)

parser.add_argument(
    "-o",
    "--save_path",
    type=str,
    default="fox.png",
    help="Path to save generated image",
)

args = parser.parse_args()

model_id = args.model_id
save_path = args.save_path
device = args.device
cache_dir = args.cache_dir
image_gen_prompt = args.prompt

ov_config = {"CACHE_DIR": cache_dir}
model = OVModelForVisualCausalLM.from_pretrained(model_id, comiple=False, trust_remote_code=True, ov_config=ov_config)
model.to(device)
model.compile()

processor = VLChatProcessor.from_pretrained(model_id, trust_remote_code=True)
print("Prompt: ", image_gen_prompt)
# Text to image generation
print("Image generation ...")
images = model.generate_image(processor, image_gen_prompt, parallel_size=1)

print(f"Save generated image in path: {save_path}")
images[0].save(save_path)

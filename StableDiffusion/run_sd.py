import argparse
import numpy as np
import time
from pathlib import Path
from optimum.intel import OVStableDiffusionPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('-c', '--checkpoint', type=str, default="stable-diffusion-v1-5/openvino/FP16", required=False,
                         help="Specify path to Hugging Face checkpoint. Default is 'stable-diffusion-v1-5/openvino/FP16'")
    parser.add_argument('-d', '--device', type=str, default="CPU", required=False,
                         help="Specify device used for OpenVINO inference.")
    parser.add_argument('-p','--prompt', type=str, default="Sailing ship in storm by Rembrandt", required=False,
                         help="Specify input prompt. Default is 'Sailing ship in storm by Rembrandt'")  
    parser.add_argument('-cd', '--cache_dir', type=str, default="model_cache", required=False,
                         help="Specify maximum number of new tokens to be generated.")
    # Parse the argument
    args = parser.parse_args()

    # Define the shapes related to the inputs and desired outputs
    batch_size = 1
    num_images_per_prompt = 1
    height = 512
    width = 512

    # initialize stable diffusion pipeline
    model_init_start = time.time()
    pipeline = OVStableDiffusionPipeline.from_pretrained(args.checkpoint, 
                                                         export=False, 
                                                         compile=False,
                                                         ov_config={"CACHE_DIR":args.cache_dir})
    # Statically reshape the model
    pipeline.reshape(batch_size=batch_size, height=height, width=width, num_images_per_prompt=num_images_per_prompt)
    # Compile the model before the first inference
    if "GPU" in args.device:
        pipeline.half()
        pipeline.to(args.device.lower())
    pipeline.compile()
    print("Load OpenVINO model in device {} finished".format(args.device))
    model_init_end = time.time()
    model_init_latency = model_init_end - model_init_start 
    print(f"Initialize stable diffusion pipeline took {model_init_latency:.3f} s")
    
    # Run inference
    image = pipeline(args.prompt, height=height, width=width, num_images_per_prompt=num_images_per_prompt).images[0]
    image.save("generated_result.png")

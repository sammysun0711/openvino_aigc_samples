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
                         help="Specify save directory of OpenVINO model cache, model cache can be disable by setting ''.")
    parser.add_argument('-sd', '--seed', type=int, required=False,
                         help="Specify random seeds for image generation")
    parser.add_argument('--static_shape', action="store_true",
                        help="Whether reshape stable diffusion pipeline with static shape")
    parser.add_argument('-b', '--batch_size', type=int, default=1, required=False,
                        help="Specifiy batch size for stable diffusion pipeline")
    parser.add_argument('--num_images_per_prompt', type=int, default=1, required=False,
                        help="Specifiy number of image per prompt for stable diffusion pipeline")
    parser.add_argument('--height', type=int, default=512, required=False,
                        help="Specifiy desired generated image height for stable diffusion pipeline")
    parser.add_argument('--width', type=int, default=512, required=False,
                        help="Specifiy desired generated image width for stable diffusion pipeline")

    # Parse the argument
    args = parser.parse_args()

    batch_size = args.batch_size
    num_images_per_prompt = args.num_images_per_prompt
    height = args.height
    width = args.width
    if args.seed:
        print(f"Set random seed: {args.seed}")
        np.random.seed(args.seed)

    # initialize stable diffusion pipeline
    model_init_start = time.time()
    pipeline = OVStableDiffusionPipeline.from_pretrained(args.checkpoint, 
                                                         export=False, 
                                                         compile=False,
                                                         ov_config={"CACHE_DIR":args.cache_dir})
    if args.static_shape:
        # Statically reshape the model
        pipeline.reshape(batch_size=batch_size, height=height, width=width, num_images_per_prompt=num_images_per_prompt)
        print("Statically reshape the stable diffusion pipeline for better performance while reduce memory usage.")

    # Compile the model before the first inference
    if "GPU" in args.device:
        pipeline.half()
    pipeline.to(args.device.lower())
    pipeline.compile()
    model_init_latency = time.time() - model_init_start
    print(f"Initialize stable diffusion pipeline took {model_init_latency:.3f} s")
    
    # Run inference
    generation_start = time.time()
    images = pipeline(args.prompt, height=height, width=width, num_images_per_prompt=num_images_per_prompt).images
    generation_latency = time.time() - generation_start
    print(f"Stable diffusion pipeline generation took {generation_latency:.3f} s")
    for idx, image in enumerate(images):
        image.save(f"generated_result_{idx}.png")

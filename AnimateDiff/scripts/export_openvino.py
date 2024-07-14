import argparse
import datetime
import inspect
import os
from functools import partial

from omegaconf import OmegaConf

import torch
import torchvision.transforms as transforms

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.models.sparse_controlnet import SparseControlNetModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.pipelines.pipeline_animation import TEXT_ENCODER_OV_PATH, VAE_ENCODER_OV_PATH, VAE_DECODER_OV_PATH, CONTROLNET_OV_PATH, UNET_OV_PATH

from animatediff.utils.util import save_videos_grid
from animatediff.utils.util import load_weights

from einops import rearrange, repeat

import csv, pdb, glob, math
from pathlib import Path
from PIL import Image
import numpy as np
import openvino as ov

from typing import Tuple
import gc

dtype_mapping = {
    torch.float32: ov.Type.f32,
    torch.float64: ov.Type.f64,
    torch.int32: ov.Type.i32,
    torch.int64: ov.Type.i64,
}


class UnetWrapper(torch.nn.Module):
    def __init__(
        self,
        unet,
        sample_dtype=torch.float32,
        timestep_dtype=torch.int64,
        encoder_hidden_states=torch.float32,
        down_block_additional_residuals=torch.float32,
        mid_block_additional_residual=torch.float32,
    ):
        super().__init__()
        self.unet = unet
        self.sample_dtype = sample_dtype
        self.timestep_dtype = timestep_dtype
        self.encoder_hidden_states_dtype = encoder_hidden_states
        self.down_block_additional_residuals_dtype = down_block_additional_residuals
        self.mid_block_additional_residual_dtype = mid_block_additional_residual

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        down_block_additional_residuals: Tuple[torch.Tensor],
        mid_block_additional_residual: torch.Tensor,
    ):
        sample.to(self.sample_dtype)
        timestep.to(self.timestep_dtype)
        encoder_hidden_states.to(self.encoder_hidden_states_dtype)
        down_block_additional_residuals = [res.to(self.down_block_additional_residuals_dtype) for res in down_block_additional_residuals]
        mid_block_additional_residual.to(self.mid_block_additional_residual_dtype)
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        )

def flattenize_inputs(inputs):
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(flattenize_inputs(input_data))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs

def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()

def convert_unet(unet, ir_path, down_block_additional_residuals, mid_block_additional_residual):
    """
    Convert Controlnet Model.
    Function accepts controlnet model, and prepares example inputs for conversion,
    Parameters:
        controlnet (torch.nn.Module): controlnet model from Animatediff pipeline
        ir_path (Path): File for storing model
    Returns:
        None
    """
    controlnet_inputs = {"sample": torch.ones([2, 4, 16, 32, 48]),
                         "timestep": torch.Tensor(1),
                         "encoder_hidden_states": torch.zeros([2, 77, 768]),
                         "controlnet_cond": torch.zeros([1, 4, 16, 32, 48]),
                         "conditioning_mask": torch.ones([1, 1, 16, 32, 48])}
    """
    latent_model_input.shape:  torch.Size([2, 4, 16, 32, 48])
    t:  tensor(961)
    text_embeddings.shape:  torch.Size([2, 77, 768])
    len(down_block_additional_residuals):  12
    down_block_additional_residuals[0].shape:  torch.Size([2, 320, 16, 32, 48])
    mid_block_additional_residual.shape:  torch.Size([2, 1280, 16, 4, 6])
    """
    
    unet_inputs = {"sample": torch.ones([2, 4, 16, 32, 48]),
                   "timestep": torch.Tensor(1),
                   "encoder_hidden_states": torch.zeros([2, 77, 768]),
                   "down_block_additional_residuals": down_block_additional_residuals,
                   "mid_block_additional_residual": mid_block_additional_residual}

    unet = UnetWrapper(unet)
    unet.eval()
    with torch.no_grad():
        ov_model = ov.convert_model(unet, example_input=unet_inputs)
        flatten_inputs = flattenize_inputs(unet_inputs.values())
        for input_data, input_tensor in zip(flatten_inputs, ov_model.inputs):
            pshape = ov.PartialShape(input_data.shape)
            if len(pshape) == 5:
                pshape[3] = -1
                pshape[4] = -1
            input_tensor.get_node().set_partial_shape(pshape)
            input_tensor.get_node().set_element_type(dtype_mapping[input_data.dtype])
        ov_model.validate_nodes_and_infer_types()
        ov.save_model(ov_model, ir_path)
        del ov_model
        del unet
        cleanup_torchscript_cache()
        gc.collect()
        print("Unet successfully converted to IR")

    
def convert_controlnet(controlnet, ir_path, controlnet_conditioning_scale=1.0, guess_mode=False, return_dict=False):
    """
    Convert Controlnet Model.
    Function accepts controlnet model, and prepares example inputs for conversion,
    Parameters:
        controlnet (torch.nn.Module): controlnet model from Animatediff pipeline
        ir_path (Path): File for storing model
    Returns:
        down_block_additional_residuals: down block addtional residuals for unet inference
        mid_block_additional_residual: middle block addtional residual for unet inference
    """
    controlnet_inputs = {"sample": torch.ones([2, 4, 16, 32, 48]),
                         "timestep": torch.Tensor(1),
                         "encoder_hidden_states": torch.zeros([2, 77, 768]),
                         "controlnet_cond": torch.zeros([1, 4, 16, 32, 48]),
                         "conditioning_mask": torch.ones([1, 1, 16, 32, 48])}
    with torch.no_grad():
        down_block_additional_residuals, mid_block_additional_residual = controlnet(
            controlnet_inputs["sample"], controlnet_inputs["timestep"],
            encoder_hidden_states=controlnet_inputs["encoder_hidden_states"],
            controlnet_cond=controlnet_inputs["controlnet_cond"],
            conditioning_mask=controlnet_inputs["conditioning_mask"],
            conditioning_scale=controlnet_conditioning_scale,
            guess_mode=guess_mode, return_dict=return_dict,
        )
    controlnet_input_info = []
    for name, inp in controlnet_inputs.items():
        pshape = ov.PartialShape(inp.shape)
        if len(pshape) == 5:
            pshape[3] = -1
            pshape[4] = -1
        controlnet_input_info.append((name, pshape))
    with torch.no_grad():
        controlnet.forward = partial(controlnet.forward, 
                                        conditioning_scale=controlnet_conditioning_scale, 
                                        guess_mode=guess_mode, return_dict=return_dict)
        ov_model = ov.convert_model(controlnet, example_input=controlnet_inputs, input=controlnet_input_info)
        ov.save_model(ov_model, ir_path)
        del ov_model
        cleanup_torchscript_cache()
        gc.collect()
        print("ControlNet successfully converted to IR")
    
    return down_block_additional_residuals, mid_block_additional_residual

def convert_encoder(text_encoder: torch.nn.Module, ir_path: Path):
    """
    Convert Text Encoder mode.
    Function accepts text encoder model, and prepares example inputs for conversion,
    Parameters:
        text_encoder (torch.nn.Module): text_encoder model from Stable Diffusion pipeline
        ir_path (Path): File for storing model
    Returns:
        None
    """
    input_ids = torch.ones((1, 77), dtype=torch.long)
    # switch model to inference mode
    text_encoder.eval()

    # disable gradients calculation for reducing memory consumption
    with torch.no_grad():
        # Export model to IR format
        ov_model = ov.convert_model(
            text_encoder,
            example_input=input_ids,
            input=([-1, -1]),
        )
    ov.save_model(ov_model, ir_path)
    del ov_model
    cleanup_torchscript_cache()
    print(f"Text Encoder successfully converted to IR and saved to {ir_path}")

def convert_vae_encoder(vae: torch.nn.Module, ir_path: Path):
    """
    Convert VAE model for encoding to IR format.
    Function accepts vae model, creates wrapper class for export only necessary for inference part,
    prepares example inputs for conversion,
    Parameters:
        vae (torch.nn.Module): VAE model from StableDiffusio pipeline
        ir_path (Path): File for storing model
    Returns:
        None
    """
    class VAEEncoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, image):
            return self.vae.encode(x=image)["latent_dist"].sample()
    vae_encoder = VAEEncoderWrapper(vae)
    vae_encoder.eval()
    image = torch.zeros((1, 3, 512, 512))
    with torch.no_grad():
        ov_model = ov.convert_model(vae_encoder, example_input=image, input=[((1, 3, -1, -1),)])
    ov.save_model(ov_model, ir_path)
    del ov_model
    cleanup_torchscript_cache()
    print(f"VAE encoder successfully converted to IR and saved to {ir_path}")

def convert_vae_decoder(vae: torch.nn.Module, ir_path: Path):
    """
    Convert VAE model for decoding to IR format.
    Function accepts vae model, creates wrapper class for export only necessary for inference part,
    prepares example inputs for conversion,
    Parameters:
        vae (torch.nn.Module): VAE model frm StableDiffusion pipeline
        ir_path (Path): File for storing model
    Returns:
        None
    """
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            return self.vae.decode(latents)

    vae_decoder = VAEDecoderWrapper(vae)
    latents = torch.zeros((1, 4, 64, 64))

    vae_decoder.eval()
    with torch.no_grad():
        ov_model = ov.convert_model(vae_decoder, example_input=latents, input=[((1, 4, -1, -1),)])
    ov.save_model(ov_model, ir_path)
    del ov_model
    cleanup_torchscript_cache()
    print(f"VAE decoder successfully converted to IR and saved to {ir_path}")

@torch.no_grad()
def main(args):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir)

    config  = OmegaConf.load(args.config)
    samples = []

    # create validation pipeline
    tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder").cpu()
    if not TEXT_ENCODER_OV_PATH.exists():
        print(f"Convert Text Encoder Pytorch Model to OpenVINO IR, and save as {TEXT_ENCODER_OV_PATH} ...")
        convert_encoder(text_encoder, TEXT_ENCODER_OV_PATH)
    else:
        print(f"Text encoder will be loaded from {TEXT_ENCODER_OV_PATH}")
    vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").cpu()
    if not VAE_ENCODER_OV_PATH.exists():
        print(f"Convert VAE Encoder Pytorch Model to OpenVINO IR, and save as {VAE_ENCODER_OV_PATH} ...")
        convert_vae_encoder(vae, VAE_ENCODER_OV_PATH)
    else:
        print(f"VAE encoder will be loaded from {VAE_ENCODER_OV_PATH}")
    if not VAE_DECODER_OV_PATH.exists():
        print(f"Convert VAE Decoder Pytorch Model to OpenVINO IR, and save as {VAE_DECODER_OV_PATH} ...")
        convert_vae_decoder(vae, VAE_DECODER_OV_PATH)
    else:
        print(f"VAE decoder will be loaded from {VAE_DECODER_OV_PATH}")
    sample_idx = 0
    for model_idx, model_config in enumerate(config):
        model_config.W = model_config.get("W", args.W)
        model_config.H = model_config.get("H", args.H)
        model_config.L = model_config.get("L", args.L)

        inference_config = OmegaConf.load(model_config.get("inference_config", args.inference_config))
        unet = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs)).cpu()
        # load controlnet model
        controlnet = controlnet_images = None
        if model_config.get("controlnet_path", "") != "":
            assert model_config.get("controlnet_images", "") != ""
            assert model_config.get("controlnet_config", "") != ""
            
            unet.config.num_attention_heads = 8
            unet.config.projection_class_embeddings_input_dim = None

            controlnet_config = OmegaConf.load(model_config.controlnet_config)
            controlnet = SparseControlNetModel.from_unet(unet, controlnet_additional_kwargs=controlnet_config.get("controlnet_additional_kwargs", {}))
            print(f"loading controlnet checkpoint from {model_config.controlnet_path} ...")
            controlnet_state_dict = torch.load(model_config.controlnet_path, map_location="cpu")
            controlnet_state_dict = controlnet_state_dict["controlnet"] if "controlnet" in controlnet_state_dict else controlnet_state_dict
            controlnet_state_dict.pop("animatediff_config", "")
            controlnet.load_state_dict(controlnet_state_dict)
            controlnet.cpu()
            image_paths = model_config.controlnet_images
            if isinstance(image_paths, str): image_paths = [image_paths]

            print(f"controlnet image paths:")
            for path in image_paths: print(path)
            assert len(image_paths) <= model_config.L

            image_transforms = transforms.Compose([
                transforms.RandomResizedCrop(
                    (model_config.H, model_config.W), (1.0, 1.0), 
                    ratio=(model_config.W/model_config.H, model_config.W/model_config.H)
                ),
                transforms.ToTensor(),
            ])

            if model_config.get("normalize_condition_images", False):
                def image_norm(image):
                    image = image.mean(dim=0, keepdim=True).repeat(3,1,1)
                    image -= image.min()
                    image /= image.max()
                    return image
            else: image_norm = lambda x: x
                
            controlnet_images = [image_norm(image_transforms(Image.open(path).convert("RGB"))) for path in image_paths]

            os.makedirs(os.path.join(savedir, "control_images"), exist_ok=True)
            for i, image in enumerate(controlnet_images):
                Image.fromarray((255. * (image.numpy().transpose(1,2,0))).astype(np.uint8)).save(f"{savedir}/control_images/{i}.png")

            controlnet_images = torch.stack(controlnet_images).unsqueeze(0).cpu()
            controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")

            if controlnet.use_simplified_condition_embedding:
                num_controlnet_images = controlnet_images.shape[2]
                controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
                controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
                controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)

        pipeline = AnimationPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            controlnet=controlnet,
            scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
        ).to("cpu")

        pipeline = load_weights(
            pipeline,
            # motion module
            motion_module_path         = model_config.get("motion_module", ""),
            motion_module_lora_configs = model_config.get("motion_module_lora_configs", []),
            # domain adapter
            adapter_lora_path          = model_config.get("adapter_lora_path", ""),
            adapter_lora_scale         = model_config.get("adapter_lora_scale", 1.0),
            # image layers
            dreambooth_model_path      = model_config.get("dreambooth_path", ""),
            lora_model_path            = model_config.get("lora_model_path", ""),
            lora_alpha                 = model_config.get("lora_alpha", 0.8),
        ).to("cpu")

        down_block_additional_residuals, mid_block_additional_residual = None, None
        if not CONTROLNET_OV_PATH.exists():
            print(f"Convert ControlNet Pytorch Model to OpenVINO IR, and save as {CONTROLNET_OV_PATH} ...")
            down_block_additional_residuals, mid_block_additional_residual = convert_controlnet(pipeline.controlnet, CONTROLNET_OV_PATH)
        else:
            print(f"Controlnet will be loaded from {CONTROLNET_OV_PATH}")

        if not UNET_OV_PATH.exists():
            print(f"Convert UNet Pytorch Model to OpenVINO IR, and save as {UNET_OV_PATH} ...")
            convert_unet(pipeline.unet, UNET_OV_PATH, down_block_additional_residuals, mid_block_additional_residual)
        else:
            print(f"UNet will be loaded from {UNET_OV_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
    parser.add_argument("--inference-config",      type=str, default="configs/inference/inference-v1.yaml")    
    parser.add_argument("--config",                type=str, required=True)
    
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)

    args = parser.parse_args()
    main(args)

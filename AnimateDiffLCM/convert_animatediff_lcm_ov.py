import argparse
import copy
from functools import partial
import gc
import inspect
import os
import numpy as np
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union, Any
from tqdm import tqdm

import torch
import openvino as ov
import nncf

from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter, LCMScheduler

MODEL_DIR = Path("animatediff-lcm-ov")

TEXT_ENCODER_PATH = MODEL_DIR / "text_encoder.xml"
UNET_PATH = MODEL_DIR / "unet.xml"
VAE_DECODER_PATH = MODEL_DIR / "vae_decoder.xml"

core = ov.Core()

def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()

def compress_model(model, save_path, group_size=128, ratio=1.0):
    if not save_path.exists():
        print("*** Model compression started ***")
        print(
            f"Compression parameters:\n\tmode = {nncf.CompressWeightsMode.INT4_SYM}\n\tratio = {ratio}\n\tgroup_size = {group_size}"
        )
        compressed_model = nncf.compress_weights(
            model,
            mode=nncf.CompressWeightsMode.INT4_SYM,
            ratio=ratio,
            group_size=group_size,
        )
        ov.save_model(compressed_model, save_path)
        del compressed_model
        print("*** Model compression end ***")


def convert_text_encoder(text_encoder):
    with torch.no_grad():
        ov_model = ov.convert_model(
            text_encoder, example_input=torch.ones([1, 77], dtype=torch.long)
        )
    ov.save_model(ov_model, TEXT_ENCODER_PATH)
    del ov_model
    cleanup_torchscript_cache()


def convert_vae_decoder(vae):
    vae_decoder = copy.deepcopy(vae)
    with torch.no_grad():
        vae_decoder.forward = vae_decoder.decode
        ov_model = ov.convert_model(
            vae_decoder, example_input=torch.ones([16, 4, 64, 64])
        )
    ov.save_model(ov_model, VAE_DECODER_PATH)
    del ov_model
    del vae_decoder
    cleanup_torchscript_cache()

def convert_unet(unet):
    inputs = {
        "sample": torch.randn((2, 4, 16, 64, 64), dtype=torch.float32),
        "encoder_hidden_states": torch.randn((32, 77, 768), dtype=torch.float32),
        "timestep": torch.ones(1, dtype=torch.int64)
    }

    with torch.no_grad():
        unet.forward = partial(unet.forward, cross_attention_kwargs=None, added_cond_kwargs=None)
        ov_model = ov.convert_model(unet, example_input=inputs)
    ov.save_model(ov_model, UNET_PATH)
    del ov_model
    del unet
    cleanup_torchscript_cache()

def get_pipeline_components(
    model_id="Realistic_Vision_V5.1_noVAE",
    adapter_path="animatediff-motion-adapter-v1-5-2"
):
    pipe_kwargs = {"trust_remote_code": True}
    print("adapter_path: ", adapter_path)
    
    adapter = MotionAdapter.from_pretrained(adapter_path, trust_remote_code=True) #torch_dtype=torch.float16)
    # load SD 1.5 based finetuned model
    pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, **pipe_kwargs)# torch_dtype=torch.float16)

    scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipe.scheduler = scheduler

    pipe.tokenizer.save_pretrained(MODEL_DIR / "tokenizer")
    pipe.scheduler.save_pretrained(MODEL_DIR / "scheduler")
    unet, vae, text_encoder = (
        None,
        None,
        None,
    )

    if not TEXT_ENCODER_PATH.exists():
        text_encoder = pipe.text_encoder
        text_encoder.eval()

    if not UNET_PATH.exists():
        unet = pipe.unet
        unet.eval()

    if not VAE_DECODER_PATH.exists():
        vae = pipe.vae
        vae.eval()

    return unet, vae, text_encoder

def get_lcm_pipeline_components(
    #model_id="Realistic_Vision_V5.1_noVAE",
    #adapter_path="animatediff-motion-adapter-v1-5-2"
    model_id="epiCRealism",
    lora_path= "AnimateLCM",
):
    pipe_kwargs = {"trust_remote_code": True}
    print("adapter_path: ", lora_path)

    adapter = MotionAdapter.from_pretrained("AnimateLCM")
    pipe = AnimateDiffPipeline.from_pretrained("epiCRealism", motion_adapter=adapter)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

    pipe.load_lora_weights("AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
    pipe.load_lora_weights("animatediff-motion-lora-tilt-up", adapter_name="tilt-up")

    pipe.set_adapters(["lcm-lora", "tilt-up"], [1.0, 0.8])
    """
    adapter = MotionAdapter.from_pretrained(adapter_path, trust_remote_code=True) #torch_dtype=torch.float16)
    # load SD 1.5 based finetuned model
    pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, **pipe_kwargs)# torch_dtype=torch.float16)

    scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipe.scheduler = scheduler
    """
    pipe.tokenizer.save_pretrained(MODEL_DIR / "tokenizer")
    pipe.scheduler.save_pretrained(MODEL_DIR / "scheduler")
    unet, vae, text_encoder = (
        None,
        None,
        None,
    )

    if not TEXT_ENCODER_PATH.exists():
        text_encoder = pipe.text_encoder
        text_encoder.eval()

    if not UNET_PATH.exists():
        unet = pipe.unet
        unet.eval()

    if not VAE_DECODER_PATH.exists():
        vae = pipe.vae
        vae.eval()

    return unet, vae, text_encoder

def convert_animatediff(
    model_id="Realistic_Vision_V5.1_noVAE",
    adapter_path="animatediff-motion-adapter-v1-5-2",
):
    conversion_statuses = [
        UNET_PATH.exists(),
        VAE_DECODER_PATH.exists(),
        TEXT_ENCODER_PATH.exists(),
    ]

    requires_conversion = not all(conversion_statuses)

    unet, vae, text_encoder = (
        None,
        None,
        None
    )

    if requires_conversion:
        unet, vae, text_encoder = (
            get_pipeline_components(model_id, adapter_path)
        )
    else:
        print("AnimateDiff model already converted")
        return

    if not TEXT_ENCODER_PATH.exists():
        print("*** T5 encoder model conversion started ***")
        convert_text_encoder(text_encoder)
        del text_encoder
        gc.collect()
        print("*** T5 encoder conversion finished ***")
    else:
        print("Found converted T5 encoder model")

    if not VAE_DECODER_PATH.exists():
        print("*** VAE decoder conversion started ***")
        convert_vae_decoder(vae)
        # del vae
        gc.collect()
        print("*** VAE decoder conversion end ***")
    else:
        print("Found converted vae decoder model")

    if not UNET_PATH.exists():
        print("*** Unet model conversion started ***")
        convert_unet(unet)
        del unet
        gc.collect()
        print("*** Unet model conversion finished ***")
    else:
        print("Found converted unet model")

def convert_animatediff_lcm(
    model_id="epiCRealism",
    lora_path = "AnimateLCM"
):
    conversion_statuses = [
        UNET_PATH.exists(),
        VAE_DECODER_PATH.exists(),
        TEXT_ENCODER_PATH.exists(),
    ]

    requires_conversion = not all(conversion_statuses)

    unet, vae, text_encoder = (
        None,
        None,
        None
    )

    if requires_conversion:
        unet, vae, text_encoder = (
            get_lcm_pipeline_components(model_id, lora_path)
        )
    else:
        print("AnimateDiff model already converted")
        return

    if not TEXT_ENCODER_PATH.exists():
        print("*** T5 encoder model conversion started ***")
        convert_text_encoder(text_encoder)
        del text_encoder
        gc.collect()
        print("*** T5 encoder conversion finished ***")
    else:
        print("Found converted T5 encoder model")

    if not VAE_DECODER_PATH.exists():
        print("*** VAE decoder conversion started ***")
        convert_vae_decoder(vae)
        # del vae
        gc.collect()
        print("*** VAE decoder conversion end ***")
    else:
        print("Found converted vae decoder model")

    if not UNET_PATH.exists():
        print("*** Unet model conversion started ***")
        convert_unet(unet)
        del unet
        gc.collect()
        print("*** Unet model conversion finished ***")
    else:
        print("Found converted unet model")

if __name__ == "__main__":
    model_id="epiCRealism"
    lora_path = "AnimateLCM"
    convert_animatediff_lcm(model_id, lora_path)

    #pipe.enable_vae_slicing()
    #pipe.enable_model_cpu_offload()
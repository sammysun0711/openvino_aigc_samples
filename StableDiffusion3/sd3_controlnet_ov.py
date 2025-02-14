import argparse
import copy
from functools import partial
import gc
import inspect
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Union, Any

import torch
import openvino as ov

from diffusers import SD3Transformer2DModel, StableDiffusion3ControlNetPipeline
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel
from diffusers.pipelines.stable_diffusion_3.pipeline_output import (
    StableDiffusion3PipelineOutput,
)
from diffusers.loaders import SD3LoraLoaderMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
    load_image,
)

from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
    AutoTokenizer,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


MODEL_DIR = Path("stable-diffusion-3-controlnet-ov")

TEXT_ENCODER_PATH = MODEL_DIR / "text_encoder.xml"
TEXT_ENCODER_2_PATH = MODEL_DIR / "text_encoder_2.xml"
TEXT_ENCODER_3_PATH = MODEL_DIR / "text_encoder_3.xml"
TEXT_ENCODER_3_INT4_PATH = MODEL_DIR / "text_encoder_3_int4.xml"

VAE_ENCODER_PATH = MODEL_DIR / "vae_encoder.xml"
CONTROLNET_PATH_POSE = MODEL_DIR / "controlnet_pose.xml"
CONTROLNET_PATH_TILE = MODEL_DIR / "controlnet_tile.xml"
CONTROLNET_PATH_CANNY = MODEL_DIR / "controlnet_canny.xml"
CONTROLNET_PATH = ""
TRANSFORMER_PATH = MODEL_DIR / "transformer.xml"
TRANSFORMER_INT8_PATH = MODEL_DIR / "transformer_int8.xml"

VAE_DECODER_PATH = MODEL_DIR / "vae_decoder.xml"

CONTROL_BLOCK_SAMPLE_LARER_PREFIX = "control_block_samples."

dtype_mapping = {
    torch.float32: ov.Type.f32,
    torch.float64: ov.Type.f64,
    torch.int32: ov.Type.i32,
    torch.int64: ov.Type.i64,
}

core = ov.Core()

def get_pipeline_components(
    use_hypersd,
    load_t5,
    model_id="stable-diffusion-3-medium-diffusers",
    lora_path="Hyper-SD/Hyper-SD3-4steps-CFG-lora.safetensors",
    controlnet_path="SD3-Controlnet-Tile",
):
    pipe_kwargs = {"trust_remote_code": True}
    if not load_t5:
        pipe_kwargs.update({"text_encoder_3": None, "tokenizer_3": None})

    controlnet = SD3ControlNetModel.from_pretrained(
        controlnet_path, trust_remote_code=True
    )
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet, **pipe_kwargs
    )

    if use_hypersd:
        pipe.load_lora_weights(lora_path, trust_remote_code=True)
        pipe.fuse_lora(lora_scale=0.125)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            Path(model_id) / "scheduler", trust_remote_code=True
        )
        pipe.scheduler = scheduler

    pipe.tokenizer.save_pretrained(MODEL_DIR / "tokenizer")
    pipe.tokenizer_2.save_pretrained(MODEL_DIR / "tokenizer_2")
    if load_t5:
        pipe.tokenizer_3.save_pretrained(MODEL_DIR / "tokenizer_3")
    pipe.scheduler.save_pretrained(MODEL_DIR / "scheduler")
    transformer, vae, text_encoder, text_encoder_2, text_encoder_3, controlnet = (
        None,
        None,
        None,
        None,
        None,
        None,
    )

    if not TEXT_ENCODER_PATH.exists():
        text_encoder = pipe.text_encoder
        text_encoder.eval()
    if not TEXT_ENCODER_2_PATH.exists():
        text_encoder_2 = pipe.text_encoder_2
        text_encoder_2.eval()
    if not TEXT_ENCODER_3_PATH.exists() and load_t5:
        text_encoder_3 = pipe.text_encoder_3
        text_encoder_3.eval()

    if not VAE_ENCODER_PATH.exists():
        vae = pipe.vae
        vae.eval()

    if not CONTROLNET_PATH.exists():
        controlnet = pipe.controlnet
        controlnet.eval()
    if not TRANSFORMER_PATH.exists():
        transformer = pipe.transformer
        transformer.eval()

    if not VAE_DECODER_PATH.exists():
        vae = pipe.vae
        vae.eval()

    return transformer, vae, text_encoder, text_encoder_2, text_encoder_3, controlnet


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


inputs = {
    "hidden_states": torch.randn((2, 16, 128, 128), dtype=torch.float32),
    "encoder_hidden_states": torch.randn((2, 333, 4096), dtype=torch.float32),
    "pooled_projections": torch.randn((2, 2048), dtype=torch.float32),
    "timestep": torch.ones(1, dtype=torch.int64),
    "controlnet_cond": torch.randn((2, 16, 128, 128), dtype=torch.float32),
    # "conditioning_scale": 0.5
}

control_block_samples = None
conditioning_scale = 0.5


def convert_controlnet(controlnet):
    print("Convert controlnet start...")
    if not CONTROLNET_PATH.exists():
        with torch.no_grad():
            controlnet.forward = partial(
                controlnet.forward,
                conditioning_scale=conditioning_scale,
                joint_attention_kwargs=None,
                return_dict=False,
            )
            global control_block_samples
            control_block_samples = controlnet(**inputs)
            # ov_model = ov.convert_model(controlnet, example_input=inputs, input=input_info)
            ov_model = ov.convert_model(controlnet, example_input=inputs)
            for i, output_tensor in enumerate(ov_model.outputs):
                r_name = output_tensor.get_node().get_friendly_name()
                # print("============")
                # print(r_name)

                n_name = CONTROL_BLOCK_SAMPLE_LARER_PREFIX + str(i)
                output_tensor.get_node().set_friendly_name(n_name)
                output_tensor.set_names({n_name})
                # print(output_tensor.get_node().get_friendly_name())
                # print("============")

            ov_model.validate_nodes_and_infer_types()
            ov.save_model(ov_model, CONTROLNET_PATH)
            del ov_model
            cleanup_torchscript_cache()
        print("ControlNet successfully converted to IR")
        del controlnet
    else:
        print(f"ControlNet will be loaded from {CONTROLNET_PATH}")


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


def convert_sd3_transformer(transformer):
    class TransformerWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            control_block_samples,
            joint_attention_kwargs=None,
            return_dict=False,
            skip_layers=None,
        ):
            block_controlnet_hidden_states = [
                res.to(torch.float32) for res in control_block_samples
            ]
            return self.model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                block_controlnet_hidden_states=block_controlnet_hidden_states,
            )

    inputs.pop("controlnet_cond", None)
    inputs["control_block_samples"] = control_block_samples[0]
    transformer = TransformerWrapper(transformer)
    transformer.forward = partial(
        transformer.forward,
        joint_attention_kwargs=None,
        return_dict=False,
        skip_layers=None,
    )

    with torch.no_grad():
        ov_model = ov.convert_model(
            transformer,
            example_input=inputs,
        )
    flatten_inputs = flattenize_inputs(inputs.values())
    i = 1
    for input_data, input_tensor in zip(flatten_inputs, ov_model.inputs):

        r_name = input_tensor.get_node().get_friendly_name()
        r_shape = ov.PartialShape(input_data.shape)
        # print("============")
        # print(r_name, r_shape)
        r_shape[0] = -1
        if len(r_shape) == 4:
            r_shape[2] = -1
            r_shape[3] = -1
        elif len(r_shape) == 3:
            r_shape[1] = -1
        elif len(r_shape) == 2:
            r_shape[1] = -1
        if (
            r_name
            not in [
                "hidden_states",
                "encoder_hidden_states",
                "pooled_projections",
                "timestep",
            ]
            and len(r_shape) == 3
        ):
            n_name = CONTROL_BLOCK_SAMPLE_LARER_PREFIX + str(i)
            input_tensor.get_node().set_friendly_name(n_name)
            input_tensor.set_names({n_name})
            i += 1
        # print(input_tensor.get_node().get_friendly_name(), r_shape)
        # print("============")
        input_tensor.get_node().set_partial_shape(r_shape)
        input_tensor.get_node().set_element_type(dtype_mapping[input_data.dtype])

    ov_model.validate_nodes_and_infer_types()
    ov.save_model(ov_model, TRANSFORMER_PATH)
    del ov_model
    cleanup_torchscript_cache()


def convert_t5_model(text_encoder):
    with torch.no_grad():
        ov_model = ov.convert_model(
            text_encoder, example_input=torch.ones([1, 77], dtype=torch.long)
        )
    ov.save_model(ov_model, TEXT_ENCODER_3_PATH)
    del ov_model
    cleanup_torchscript_cache()


def convert_clip_model(text_encoder, text_encoder_path):
    text_encoder.forward = partial(
        text_encoder.forward, output_hidden_states=True, return_dict=False
    )
    with torch.no_grad():
        ov_model = ov.convert_model(
            text_encoder, example_input=torch.ones([1, 77], dtype=torch.long)
        )
    ov.save_model(ov_model, text_encoder_path)
    del ov_model
    cleanup_torchscript_cache()


def convert_vae_encoder(vae):
    vae_encoder = copy.deepcopy(vae)
    with torch.no_grad():
        vae_encoder.forward = lambda sample: {
            "latent_sample": vae_encoder.encode(x=sample)["latent_dist"].sample()
        }
        # vae.forward = vae.encode
        ov_model = ov.convert_model(
            vae_encoder,
            example_input=torch.ones([2, 3, 512, 512]),
            input=[-1, 3, -1, -1],
        )
    ov.save_model(ov_model, VAE_ENCODER_PATH)
    del ov_model
    del vae_encoder
    cleanup_torchscript_cache()


def convert_vae_decoder(vae):
    vae_decoder = copy.deepcopy(vae)
    with torch.no_grad():
        vae_decoder.forward = vae_decoder.decode
        ov_model = ov.convert_model(
            vae_decoder, example_input=torch.ones([1, 16, 64, 64])
        )
    ov.save_model(ov_model, VAE_DECODER_PATH)
    del ov_model
    del vae_decoder
    cleanup_torchscript_cache()


def convert_sd3(
    load_t5,
    use_hypersd,
    model_id="stable-diffusion-3-medium-diffusers",
    lora_path="Hyper-SD/Hyper-SD3-4steps-CFG-lora.safetensors",
    controlnet_path="SD3-Controlnet-Tile",
):
    conversion_statuses = [
        TRANSFORMER_PATH.exists(),
        CONTROLNET_PATH.exists(),
        VAE_ENCODER_PATH.exists(),
        VAE_DECODER_PATH.exists(),
        TEXT_ENCODER_PATH.exists(),
        TEXT_ENCODER_2_PATH.exists(),
    ]

    if load_t5:
        conversion_statuses.append(TEXT_ENCODER_3_PATH.exists())

    requires_conversion = not all(conversion_statuses)

    transformer, vae, text_encoder, text_encoder_2, text_encoder_3, controlnet = (
        None,
        None,
        None,
        None,
        None,
        None,
    )

    if requires_conversion:
        transformer, vae, text_encoder, text_encoder_2, text_encoder_3, controlnet = (
            get_pipeline_components(
                use_hypersd, load_t5, model_id, lora_path, controlnet_path
            )
        )
    else:
        print("SD3 model already converted")
        return

    if load_t5 and not TEXT_ENCODER_3_PATH.exists():
        print("T5 encoder model conversion started")
        convert_t5_model(text_encoder_3)
        del text_encoder_3
        gc.collect()
        print("T5 encoder conversion finished")
    elif load_t5:
        print("Found converted T5 encoder model")

    if not TEXT_ENCODER_PATH.exists():
        print("Clip Text encoder 1 conversion started")
        convert_clip_model(text_encoder, TEXT_ENCODER_PATH)
        del text_encoder
        gc.collect()
        print("Clip Text encoder 1 conversion finished")
    else:
        print("Found converted Clip Text encoder 1")

    if not TEXT_ENCODER_2_PATH.exists():
        print("Clip Text encoder 2 conversion started")
        convert_clip_model(text_encoder_2, TEXT_ENCODER_2_PATH)
        del text_encoder_2
        gc.collect()
        print("Clip Text encoder 2 conversion finished")
    else:
        print("Found converted Clip Text encoder 2")

    if not VAE_ENCODER_PATH.exists():
        print("VAE encoder conversion started")
        convert_vae_encoder(vae)
        # del vae
        gc.collect()

    if not VAE_DECODER_PATH.exists():
        print("VAE decoder conversion started")
        convert_vae_decoder(vae)
        # del vae
        gc.collect()

    if not CONTROLNET_PATH.exists():
        print("Controlnet conversion started")
        convert_controlnet(controlnet)
        del controlnet
        gc.collect()

    if not TRANSFORMER_PATH.exists():
        print("Transformer model conversion started")
        convert_sd3_transformer(transformer)
        del transformer
        gc.collect()
        print("Transformer model conversion finished")

    else:
        print("Found converted transformer model")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class OVStableDiffusion3ControlNetPipeline(DiffusionPipeline):
    r"""
    Args:
        transformer ([`SD3Transformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant,
            with an additional added projection layer that is initialized with a diagonal matrix with the `hidden_size`
            as its dimension.
        text_encoder_2 ([`CLIPTextModelWithProjection`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection),
            specifically the
            [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
            variant.
        text_encoder_3 ([`T5EncoderModel`]):
            Frozen text-encoder. Stable Diffusion 3 uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`CLIPTokenizer`):
            Second Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_3 (`T5TokenizerFast`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        controlnet ([`SD3ControlNetModel`] or `List[SD3ControlNetModel]` or [`SD3MultiControlNetModel`]):
            Provides additional conditioning to the `unet` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined
            additional conditioning.
    """

    model_cpu_offload_seq = (
        "text_encoder->text_encoder_2->text_encoder_3->transformer->vae"
    )
    _optional_components = []
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
        "negative_pooled_prompt_embeds",
    ]

    def __init__(
        self,
        transformer: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae_encoder: AutoencoderKL,
        vae_decoder: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
        controlnet: SD3ControlNetModel,
        text_encoder_3_dim=4096,
        joint_attention_dim=4096,
        force_zeros_for_pooled_projection=True,
    ):
        super().__init__()
        """
        if isinstance(controlnet, (list, tuple)):
            controlnet = SD3MultiControlNetModel(controlnet)
        if isinstance(controlnet, SD3MultiControlNetModel):
            for controlnet_model in controlnet.nets:
                # for SD3.5 8b controlnet, it shares the pos_embed with the transformer
                if (
                    hasattr(controlnet_model.config, "use_pos_embed")
                    and controlnet_model.config.use_pos_embed is False
                ):
                    pos_embed = controlnet_model._get_pos_embed_from_transformer(transformer)
                    controlnet_model.pos_embed = pos_embed.to(controlnet_model.dtype).to(controlnet_model.device)
        elif isinstance(controlnet, SD3ControlNetModel):
            if hasattr(controlnet.config, "use_pos_embed") and controlnet.config.use_pos_embed is False:
                pos_embed = controlnet._get_pos_embed_from_transformer(transformer)
                controlnet.pos_embed = pos_embed.to(controlnet.dtype).to(controlnet.device)
        """

        self.register_modules(
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            transformer=transformer,
            scheduler=scheduler,
            controlnet=controlnet,
        )
        """
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = (
            self.transformer.config.sample_size
            if hasattr(self, "transformer") and self.transformer is not None
            else 128
        )
        """
        self.vae_scale_factor = 2**3
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length
            if hasattr(self, "tokenizer") and self.tokenizer is not None
            else 77
        )
        self.vae_scaling_factor = 1.5305
        self.vae_shift_factor = 0.0609
        self.default_sample_size = 64
        self._text_encoder_3_dim = text_encoder_3_dim
        self._joint_attention_dim = joint_attention_dim
        self._force_zeros_for_pooled_projection = force_zeros_for_pooled_projection

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size * num_images_per_prompt,
                    self.tokenizer_max_length,
                    self.joint_attention_dim,
                ),
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_3(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer_3.batch_decode(
                untruncated_ids[:, self.tokenizer_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = torch.from_numpy(self.text_encoder_3(text_input_ids)[0])
        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        return prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_clip_prompt_embeds
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        clip_model_index: int = 0,
    ):
        device = device or self._execution_device

        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(text_input_ids)
        pooled_prompt_embeds = torch.from_numpy(prompt_embeds[0])
        hidden_states = list(prompt_embeds.values())[1:]

        if clip_skip is None:
            prompt_embeds = torch.from_numpy(hidden_states[-2])
        else:
            prompt_embeds = torch.from_numpy(hidden_states[-(clip_skip + 2)])

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            batch_size * num_images_per_prompt, -1
        )

        return prompt_embeds, pooled_prompt_embeds

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used in all the text-encoders.
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used in both text-encoders
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, SD3LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds,
                (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat(
                [pooled_prompt_embed, pooled_prompt_2_embed], dim=-1
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt_2 or negative_prompt
            negative_prompt_3 = negative_prompt_3 or negative_prompt

            # normalize str to list
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )
            negative_prompt_2 = (
                batch_size * [negative_prompt_2]
                if isinstance(negative_prompt_2, str)
                else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3]
                if isinstance(negative_prompt_3, str)
                else negative_prompt_3
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embed, negative_pooled_prompt_embed = (
                self._get_clip_prompt_embeds(
                    negative_prompt,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    clip_skip=None,
                    clip_model_index=0,
                )
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = (
                self._get_clip_prompt_embeds(
                    negative_prompt_2,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    clip_skip=None,
                    clip_model_index=1,
                )
            )
            negative_clip_prompt_embeds = torch.cat(
                [negative_prompt_embed, negative_prompt_2_embed], dim=-1
            )

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (
                    0,
                    t5_negative_prompt_embed.shape[-1]
                    - negative_clip_prompt_embeds.shape[-1],
                ),
            )

            negative_prompt_embeds = torch.cat(
                [negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2
            )
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        if self.text_encoder is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        return (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def check_inputs(
        self,
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=None,
        negative_prompt_2=None,
        negative_prompt_3=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_3 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_3`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )
        elif prompt_2 is not None and (
            not isinstance(prompt_2, str) and not isinstance(prompt_2, list)
        ):
            raise ValueError(
                f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}"
            )
        elif prompt_3 is not None and (
            not isinstance(prompt_3, str) and not isinstance(prompt_3, list)
        ):
            raise ValueError(
                f"`prompt_3` has to be of type `str` or `list` but is {type(prompt_3)}"
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_2 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_2`: {negative_prompt_2} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )
        elif negative_prompt_3 is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt_3`: {negative_prompt_3} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if negative_prompt_embeds is not None and negative_pooled_prompt_embeds is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_pooled_prompt_embeds` also have to be passed. Make sure to generate `negative_pooled_prompt_embeds` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(
                f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}"
            )

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        # device,
        generator,
        latents=None,
    ):
        """
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        """
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = randn_tensor(shape, generator=generator)

        return latents

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        if isinstance(image, torch.Tensor):
            pass
        else:
            image = self.image_processor.preprocess(image, height=height, width=width)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        control_image: PipelineImageInput = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        controlnet_pooled_projections: Optional[torch.FloatTensor] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            control_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.Tensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.Tensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be accepted
                as an image. The dimensions of the output image defaults to `image`'s dimensions. If height and/or
                width are passed, `image` is resized accordingly. If multiple ControlNets are specified in `init`,
                images must be passed as a list such that each element of the list can be correctly batched for input
                to a single ControlNet.
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            controlnet_pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`):
                Embeddings projected from the embeddings of controlnet input conditions.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
                `text_encoder_2`. If not defined, `negative_prompt` is used instead
            negative_prompt_3 (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
                `text_encoder_3`. If not defined, `negative_prompt` is used instead
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
                input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            clip_skip=self.clip_skip,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )

        # 3. Prepare control image
        if self._force_zeros_for_pooled_projection:
            # instantx sd3 controlnet does not apply shift factor
            vae_shift_factor = 0
        else:
            vae_shift_factor = self.vae_shift_factor

        control_image = self.prepare_image(
            image=control_image,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            guess_mode=False,
        )
        height, width = control_image.shape[-2:]

        control_image = torch.from_numpy(self.vae_encoder(control_image.numpy())[0])
        control_image = (control_image - vae_shift_factor) * self.vae_scaling_factor

        # 4. Prepare timesteps
        # timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, sigmas=sigmas
        )

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        # num_channels_latents = self.transformer.config.in_channels
        num_channels_latents = 16
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            # device,
            generator,
            latents,
        )

        # 6. Create tensor stating which controlnets to keep
        if self._force_zeros_for_pooled_projection:
            # instantx sd3 controlnet used zero pooled projection
            controlnet_pooled_projections = torch.zeros_like(pooled_prompt_embeds)
        else:
            controlnet_pooled_projections = (
                controlnet_pooled_projections or pooled_prompt_embeds
            )

        if self._joint_attention_dim is not None:
            controlnet_encoder_hidden_states = prompt_embeds
        else:
            # SD35 official 8b controlnet does not use encoder_hidden_states
            controlnet_encoder_hidden_states = None

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                # start = time.perf_counter()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # controlnet(s) inference
                controlnet_inputs = {
                    "hidden_states": latent_model_input,
                    "controlnet_cond": control_image,
                    "encoder_hidden_states": controlnet_encoder_hidden_states,
                    "pooled_projections": controlnet_pooled_projections,
                    "timestep": timestep,
                }
                """
                print("============================ Controlnet input shape ====================================")
                print("hidden_states.shape: ", latent_model_input.shape)
                print("controlnet_cond.shape: ", control_image.shape)
                print("controlnet_encoder_hidden_states.shape: ", controlnet_encoder_hidden_states.shape)
                print("pooled_projections.shape: ", controlnet_pooled_projections.shape)
                print("timestep.shape: ", timestep)
                """
                # print("controlnet_inputs: ", controlnet_inputs)
                """
                prepare_controlnet_input_dur = time.perf_counter() - start
                print(
                    f"Prepare controlnet input duration, elapsed {prepare_controlnet_input_dur:.03f} secs. "
                )
                """
                start = time.perf_counter()
                controlnet_outputs = self.controlnet(controlnet_inputs)
                # print("controlnet_outputs: ", controlnet_outputs)
                controlnet_inference_duration = time.perf_counter() - start
                # print(f" Controlnet inference duration, elapsed {controlnet_inference_duration:.03f} secs. ")

                #start = time.perf_counter()
                # print("self.transformer.inputs:", self.transformer.inputs)
                transformer_inputs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": prompt_embeds,
                    "pooled_projections": pooled_prompt_embeds,
                }
                """
                print("============================ Transformer input shape ====================================")
                print("hidden_states.shape: ", latent_model_input.shape)
                print("timestep.shape: ", timestep.shape)
                print("encoder_hidden_states.shape: ", prompt_embeds.shape)
                print("pooled_projections.shape: ", pooled_prompt_embeds.shape)
                print("controlnet_outputs[0].shape: ", controlnet_outputs[0].shape)
                """
                for i in range(len(controlnet_outputs)):
                    transformer_inputs[
                        CONTROL_BLOCK_SAMPLE_LARER_PREFIX + str(i + 1)
                    ] = controlnet_outputs[i]
                """
                prepare_transformer_input_dur = time.perf_counter() - start
                print(
                    f"Prepare transformer input duration, elapsed {prepare_transformer_input_dur:.03f} secs. "
                )
                """
                start = time.perf_counter()
                noise_pred = torch.from_numpy(self.transformer(transformer_inputs)[0])
                transformer_inference_duration = time.perf_counter() - start

                # print(f"Transformer inference duration, elapsed {transformer_inference_duration:.03f} secs. ")

                # print("noise_pred: ", noise_pred)

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae_scaling_factor) + self.vae_shift_factor

            # image = self.vae.decode(latents, return_dict=False)[0]
            image = torch.from_numpy(self.vae_decoder(latents)[0])
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)

import openvino.properties as props

def init_sd3_controlnet_pipeline(
    models_dict: Dict[str, Any], device: str, text_encoder_3_dim=4096
):
    pipeline_args = {}
    ov_config = {}
    if args.cache_mode.upper() == "OPTIMIZE_SIZE":
        print("Model cahce mode: OPTIMIZE_SIZE")
        core.set_property({"CACHE_DIR": "model_cache_optimize_size", "CACHE_MODE": "OPTIMIZE_SIZE"})
    elif args.cache_mode.upper() == "OPTIMIZE_SPEED":
        print("Model cahce mode: OPTIMIZE_SPEED")
        core.set_property({"CACHE_DIR": "model_cache_optimzie_speed", "CACHE_MODE": "OPTIMIZE_SPEED"})
    else:
        print("Model cache: default")
        core.set_property({"CACHE_DIR": "model_cache"})
    if "GPU" in device:
        ov_config["INFERENCE_PRECISION_HINT"] = "f16"
        ov_config["ACTIVATIONS_SCALE_FACTOR"] = "10"

    print("Models compilation started ...")
    for model_name, model_path in models_dict.items():
        # core.set_property({props.cache_dir: f"model_cache_speed/{model_name}_cache.blob"})
        pipeline_args[model_name] = core.compile_model(
            model_path, device, ov_config if "text_encoder_3" in model_name else {}
        )
        print(f"{model_name} - Done!")
    print("Models compilation finished!")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_DIR / "scheduler")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "tokenizer")
    tokenizer_2 = AutoTokenizer.from_pretrained(MODEL_DIR / "tokenizer_2")
    tokenizer_3 = (
        AutoTokenizer.from_pretrained(MODEL_DIR / "tokenizer_3")
        if "text_encoder_3" in models_dict
        else None
    )

    pipeline_args["scheduler"] = scheduler
    pipeline_args["tokenizer"] = tokenizer
    pipeline_args["tokenizer_2"] = tokenizer_2
    pipeline_args["tokenizer_3"] = tokenizer_3
    pipeline_args["text_encoder_3_dim"] = text_encoder_3_dim
    ov_pipe = OVStableDiffusion3ControlNetPipeline(**pipeline_args)

    return ov_pipe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Stable Diffusion 3 Controlnet with OpenVNIO",
        add_help=True,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="stable-diffusion-3-controlnet-ov",
        help="Model folder including Stable Diffusion 3 Medium OpenVINO model",
    )
    parser.add_argument(
        "-c",
        "--controlnet_type",
        default="tile",
        help="Controlnet type for Stable Diffusion 3 Medium Controlnet Pipeline with OpenVINO runtime, supported 'canny', 'pose', 'tile'",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default="Anime style illustration of a girl wearing a suit. A moon in sky. In the background we see a big rain approaching.",
        type=str,
        help="positve prompt",
    )
    parser.add_argument(
        "-np",
        "--n_prompt",
        default="NSFW, nude, naked, porn, ugly",
        type=str,
        help="negativ_prompt",
    )
    parser.add_argument(
        "-d", "--device", default="CPU", type=str, help="Inference device"
    )
    parser.add_argument(
        "--load_t5",
        default=True,
        type=bool,
        help="Whether use T5XXL as Text Encoder 3 for image generation",
    )
    parser.add_argument(
        "--guidance_scale",
        default=3.0,
        type=float,
        help="Classifier-Free Guidance (CFG) scale to control image generation process",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Random seed for image generation"
    )
    parser.add_argument(
        "--height", default=512, type=int, help="Specify target generated image height"
    )
    parser.add_argument(
        "--width", default=512, type=int, help="Specify target generated image width"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="assets/tile.jpg",
        help="Path of Input Image for Controlnet for Stable Diffusion 3 Medium Pytorch models",
    )
    parser.add_argument(
        "--use_t5_int4",
        default=False,
        action="store_true",
        help="Whether use INT4 weight compressed T5XXL",
    )
    parser.add_argument(
        "--use_transformer_int8",
        default=False,
        action="store_true",
        help="Whether use INT8 quantized transformer model",
    )
    parser.add_argument(
        "--num_image_generation",
        default=4,
        action="store_true",
        help="Specify number of image generation for performance evaluation",
    )
    parser.add_argument(
        "--num_inference_steps",
        default=4,
        type=int,
        help="Specify how many LCM generation steps",
    )
    parser.add_argument(
        "--cache_mode",
        default="optimize_size",
        type=str,
        help="Specify which cache mode for weightless cache, option `optimize_speed` or 'optimize_size' (by default)",
    )
    args = parser.parse_args()
    print("Args: ", args)
    print("OpenVINO version: ", ov.get_version())

    load_t5 = args.load_t5
    use_t5_int4 = args.use_t5_int4
    device = args.device
    controlnet_image_path = args.image_path
    prompt = args.prompt
    n_prompt = args.n_prompt
    guidance_scale = args.guidance_scale
    height = args.height
    width = args.width
    seed = args.seed
    controlnet_type = args.controlnet_type
    use_transformer_int8 = args.use_transformer_int8

    if "canny" in controlnet_type.lower():
        CONTROLNET_PATH = CONTROLNET_PATH_CANNY
    elif "pose" in controlnet_type.lower():
        CONTROLNET_PATH = CONTROLNET_PATH_POSE
    elif "tile" in controlnet_type.lower():
        CONTROLNET_PATH = CONTROLNET_PATH_TILE
    else:
        print("Error! Found unsupported controlnet for SD3: ", controlnet_type)
        exit(1)

    models_dict = {
        #    "transformer": TRANSFORMER_PATH,
        "controlnet": CONTROLNET_PATH,
        "vae_encoder": VAE_ENCODER_PATH,
        "vae_decoder": VAE_DECODER_PATH,
        "text_encoder": TEXT_ENCODER_PATH,
        "text_encoder_2": TEXT_ENCODER_2_PATH,
    }

    if load_t5 and not use_t5_int4:
        print("------------------- Use Text Encoder FP16 Model -------------------------")
        models_dict["text_encoder_3"] = TEXT_ENCODER_3_PATH

    if load_t5 and use_t5_int4:
        print("------------------- Use Text Encoder INT4 Model -------------------------")
        models_dict["text_encoder_3"] = TEXT_ENCODER_3_INT4_PATH

    if use_transformer_int8:
        print("------------------- Use Transformer INT8 Model -------------------------")
        models_dict["transformer"] = TRANSFORMER_INT8_PATH
    else:
        print("------------------- Use Transformer FP16 Model -------------------------")
        models_dict["transformer"] = TRANSFORMER_PATH

    print("Pipeline initialization start ...")
    start = time.perf_counter()
    ov_pipe = init_sd3_controlnet_pipeline(models_dict, device)
    pipeline_init_duration = time.perf_counter() - start
    print(f"Pipeline initialization end, elapsed {pipeline_init_duration:.03f} secs. ")

    control_image = load_image(controlnet_image_path)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    print(f"Image generation start for {args.num_image_generation} images ...")
    start = time.perf_counter()
    image = ov_pipe(
        prompt,
        negative_prompt=n_prompt,
        control_image=control_image,
        num_inference_steps=args.num_inference_steps,
        controlnet_conditioning_scale=0.5,
        guidance_scale=guidance_scale,
        generator=generator,
        height=height,
        width=width,
    ).images[0]

    inference_duration = time.perf_counter() - start

    print(f"==== First Image generation finished, elapsed {inference_duration:.03f} secs. ====\n")

    infer_count = args.num_image_generation - 1
    start = time.perf_counter()
    for i in range(infer_count):
        image = ov_pipe(
            prompt,
            negative_prompt=n_prompt,
            control_image=control_image,
            num_inference_steps=args.num_inference_steps,
            controlnet_conditioning_scale=0.5,
            guidance_scale=guidance_scale,
            generator=generator,
            height=height,
            width=width,
        ).images[0]
    inference_duration = time.perf_counter() - start

    print(f"==== Average Image generation finished, elapsed {(inference_duration/infer_count):.03f} secs. ====\n")
    image.save(
        f"results/ov_sd3_controlnet_{controlnet_type}_guidance_scale_{guidance_scale}_{height}x{width}_seed_{seed}.jpg"
    )

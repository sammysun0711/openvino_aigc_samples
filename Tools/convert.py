import gc
from argparse import ArgumentParser
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from optimum.intel.openvino import (
    OVModelForCausalLM, OVStableDiffusionPipeline, OVQuantizer, 
    OV_XML_FILE_NAME,
)
from optimum.exporters.openvino import export_models
from optimum.exporters.onnx import __main__ as optimum_main
from nncf import compress_weights
from diffusers import StableDiffusionPipeline
import time

def convert_causal_lm(args):
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    start = time.perf_counter()
    if args.save_orig:
        pt_out_dir = Path(args.output_dir) / "pytorch"
        pt_model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True))
        pt_model.save_pretrained(pt_out_dir)
        try:
            tok.save_pretrained(pt_out_dir)
        except Exception:
            print("tokenizer loading failed")
        del pt_model
        gc.collect()
    if args.compress_weights:
         pt_model = AutoModelForCausalLM.from_pretrained(args.model_id, trust_remote_code=True, config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True))
         feature = "text-generation"
         quantizer = OVQuantizer.from_pretrained(pt_model, task=feature)
         ov_out_dir = Path(args.output_dir) / "INT8"
         quantizer.quantize(save_directory=ov_out_dir, weights_only=True)
         try:
            tok.save_pretrained(ov_out_dir)
         except Exception:
            print("tokenizer loading failed")

    model = OVModelForCausalLM.from_pretrained(args.model_id, export=True, compile=False, trust_remote_code=True, config=AutoConfig.from_pretrained(args.model_id, trust_remote_code=True))
    end = time.perf_counter()
    print(f"Conversion total time {end - start}s")
    start1 = time.perf_counter()
    if args.precision == "FP16":
        model.half()
    ov_out_dir = Path(args.output_dir) / args.precision
    model.save_pretrained(ov_out_dir)
    end1 = time.perf_counter()
    print(f"Serialization total time {end1 - start1}s")

    try:
        tok.save_pretrained(ov_out_dir)
    except Exception:
        print("tokenizer loading failed")

def convert_sd(args):
    start = time.perf_counter()
    pt_model = StableDiffusionPipeline.from_pretrained(args.model_id) if args.save_orig or args.compress_weights else None
    if args.save_orig:
        pt_model.save_pretrained(Path(args.output_dir) / "pytorch")
    if args.compress_weights:
        wc_text_encoder = compress_weights(pt_model.text_encoder)
        wc_unet = compress_weights(pt_model.unet)
        wc_vae = compress_weights(pt_model.vae)
        pt_model.text_encoder = wc_text_encoder
        pt_model.unet = wc_unet
        pt_model.vae = wc_vae
        onnx_config, models_and_onnx_configs = optimum_main._get_submodels_and_onnx_configs(
            model=pt_model,
            task="stable-diffusion",
            monolith=False,
            custom_onnx_configs={},
            custom_architecture=False,
        )
        output = Path(args.output_dir) / "INT8"
        for model_name in models_and_onnx_configs:
            subcomponent = models_and_onnx_configs[model_name][0]
            if hasattr(subcomponent, "save_config"):
                subcomponent.save_config(output / model_name)
            elif hasattr(subcomponent, "config") and hasattr(subcomponent.config, "save_pretrained"):
                subcomponent.config.save_pretrained(output / model_name)

        files_subpaths = [Path(name_dir) / OV_XML_FILE_NAME for name_dir in models_and_onnx_configs]

        # Saving the additional components needed to perform inference.
        pt_model.scheduler.save_pretrained(output.joinpath("scheduler"))

        feature_extractor = getattr(pt_model, "feature_extractor", None)
        if feature_extractor is not None:
            feature_extractor.save_pretrained(output.joinpath("feature_extractor"))

        tokenizer = getattr(pt_model, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.save_pretrained(output.joinpath("tokenizer"))

        tokenizer_2 = getattr(pt_model, "tokenizer_2", None)
        if tokenizer_2 is not None:
            tokenizer_2.save_pretrained(output.joinpath("tokenizer_2"))

        pt_model.save_config(output)

        export_models(
            models_and_onnx_configs=models_and_onnx_configs,
            output_dir=output,
            output_names=files_subpaths
        )

    model = OVStableDiffusionPipeline.from_pretrained(args.model_id, export=True, compile=False)
    if args.precision == "FP16":
        model.half()
    end = time.perf_counter()
    print(f"Conversion total time {end - start}s")
    start1 = time.perf_counter()
    model.save_pretrained(Path(args.output_dir) / args.precision)
    end1 = time.perf_counter()
    print(f"Serialization total time {end1 - start1}s")

converters = {
    "decoder": convert_causal_lm,
    "stable-diffusion": convert_sd,
}

def get_convert_model_type(model_id):
    default = "decoder"
    for key in converters:
        if key in model_id:
            return key
        
    return default

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_id", required=True, 
                        help="Model id of a pretrained Hugging Face model on the Hub or local directory")
    parser.add_argument("--output_dir", required=True, 
                        help="Save directory of converted OpenVINO Model and configurations")
    parser.add_argument("--save_orig", action="store_true", 
                        help="Whether save original Hugging Face Pytorch model on disk or not.")
    parser.add_argument("--precision", choices=["FP32", "FP16"], default="FP32", 
                        help="Specifiy model precision of converted OpenVINO Models")
    parser.add_argument("--compress_weights", action="store_true", 
                        help="Whether quantize Model to INT8 model with compress only method or not")

    args = parser.parse_args()
    model_type = get_convert_model_type(args.model_id)
    converter = converters[model_type]
    converter(args)

if __name__ == "__main__":
    main()

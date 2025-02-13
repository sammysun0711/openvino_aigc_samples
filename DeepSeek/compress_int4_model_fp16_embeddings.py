import openvino as ov
import nncf 
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Compress OpenVINO Model with FP16 Embedding", add_help=True)
    parser.add_argument("-m", "--model_dir", required=True, type=str, help="OpenVINO model for loading")
    parser.add_argument("-o", "--output_dir", required=True, type=str, default="DeepSeek-R1-Distill-Qwen-1.5B-OV-INT4-GS32-FP16-Embeddings", help="output directory for saving model")
    parser.add_argument("-gs", "--group_size", required=False, default=32, type=int, help="Group size for model compression")
    parser.add_argument("--mode", required=False, default="asym", type=str, help="Specify whehter symetric compression or aysmetric compression")

    args = parser.parse_args()
    model_dir = args.model_dir
    output_dir = args.output_dir
    group_size = args.group_size


    fp16_model_path = f"{model_dir}/openvino_model.xml"
    int4_model_path = f"{output_dir}/openvino_model.xml"

    mode = None
    if args.mode == "sym":
        mode = nncf.CompressWeightsMode.INT4_SYM
    elif args.mode == "asym":
        mode = nncf.CompressWeightsMode.INT4_ASYM
    compression_configuration = {
            "mode": mode,
            "group_size": group_size
    }
    print("Compression configuration: ", compression_configuration)

    core = ov.Core()
    ov_model = core.read_model(fp16_model_path)
    ov_compressed_model = nncf.compress_weights(ov_model, **compression_configuration, backup_mode=nncf.BackupMode.NONE)
    ov.save_model(ov_compressed_model, int4_model_path)

import argparse
from pathlib import Path
from janus.models import VLChatProcessor
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForVisualCausalLM
import nncf

parser = argparse.ArgumentParser(
    "DeepSeek Janus-Pro Pytorch to OpenVINO Model Conversion Tool",
    add_help=True,
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    "-m",
    "--model_id",
    type=str,
    default="Janus-Pro-1B",
    help="Model folder including Janus-Pro Pytorch Models",
)

parser.add_argument(
    "-o",
    "--save_dir",
    type=str,
    default="Janus-Pro-1B-OV",
    help="Model folder to save converted Janus-Pro OpenVINO Models",
)

compression_configuration = {
        "mode": nncf.CompressWeightsMode.INT4_ASYM,
        "bits": 4,
        "group_size": 64,
        "ratio": 1.0,
}

args = parser.parse_args()

model_id = args.model_id
save_dir = args.save_dir

model = OVModelForVisualCausalLM.from_pretrained(
    model_id,
    export=True,
    trust_remote_code=True,
    load_in_8bit=False,
    quantization_config=compression_configuration,
)
model.save_pretrained(save_dir)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.save_pretrained(save_dir)

processor = VLChatProcessor.from_pretrained(model_id, trust_remote_code=True)
processor.save_pretrained(save_dir)

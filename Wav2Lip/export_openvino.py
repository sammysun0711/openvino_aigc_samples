import numpy as np
import os
import openvino as ov
from pathlib import Path
import torch
from face_detection.detection.sfd.net_s3fd import s3fd
from models import Wav2Lip

def _load(checkpoint_path):
    checkpoint = torch.load(checkpoint_path,
                            map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    return model.eval()

if __name__ == "__main__":
    models_urls = {
            's3fd': 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth'}
    path_to_detector = "checkpoints/face_detection.pth"
    path_to_wav2lip = "checkpoints/wav2lip.pth"

    OV_FACE_DETECTION_MODEL_PATH = Path("checkpoints/face_detection.xml")
    OV_WAV2LIP_MODEL_PATH = Path("checkpoints/wav2lip.xml")

    # Convert Face Detection Model
    print("Convert Face Detection Model ...")
    if not os.path.isfile(path_to_detector):
        model_weights = load_url(models_urls['s3fd'])
    else:
        model_weights = torch.load(path_to_detector)
    face_detector = s3fd()
    face_detector.load_state_dict(model_weights)

    if not OV_FACE_DETECTION_MODEL_PATH.exists():
        face_detection_dummy_inputs = torch.FloatTensor(np.random.rand(1, 3, 768, 576))
        face_detection_ov_model = ov.convert_model(face_detector, example_input=face_detection_dummy_inputs)
        ov.save_model(face_detection_ov_model, OV_FACE_DETECTION_MODEL_PATH)
    print("Converted face detection OpenVINO model: ", OV_FACE_DETECTION_MODEL_PATH)

    print("Convert Wav2Lip Model ...")
    wav2lip = load_model(path_to_wav2lip)
    img_batch = torch.FloatTensor(np.random.rand(123, 6, 96, 96))
    mel_batch = torch.FloatTensor(np.random.rand(123, 1, 80, 16))

    if not OV_WAV2LIP_MODEL_PATH.exists():
        example_inputs = {"audio_sequences": mel_batch, "face_sequences": img_batch}
        wav2lip_ov_model = ov.convert_model(wav2lip, example_input=example_inputs)
        ov.save_model(wav2lip_ov_model, OV_WAV2LIP_MODEL_PATH)
    print("Converted face detection OpenVINO model: ", OV_FACE_DETECTION_MODEL_PATH)

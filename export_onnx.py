import os
import torch
import numpy as np
import onnxruntime as ort
from onnxruntime.transformers import optimizer

from model import CaptchaTransformer
from dataset import NUM_CLASSES

WEIGHT_PATH = "model_v2.pth"
TEMP_PATH = "_model_v2.onnx"
OUTPUT_PATH = "model_v2.onnx"
OPSET = 18


def export_raw():
    device = torch.device("cpu")

    model = CaptchaTransformer(
        num_classes=NUM_CLASSES,
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.eval()

    dummy = torch.randn(1, 1, 32, 128, device=device)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            TEMP_PATH,
            input_names=["input"],
            output_names=["logits"],
            opset_version=OPSET,
            dynamic_axes={
                "input": {0: "batch"},
                "logits": {0: "batch"},
            },
            do_constant_folding=True,
        )


def optimize():
    opt_model = optimizer.optimize_model(
        TEMP_PATH,
        model_type="bert",
        num_heads=4,
        hidden_size=256,
        opt_level=99,
    )
    opt_model.save_model_to_file(OUTPUT_PATH)


def cleanup():
    for path in [TEMP_PATH, TEMP_PATH + ".data"]:
        if os.path.exists(path):
            os.remove(path)


def main():
    export_raw()
    optimize()
    cleanup()
    size = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"File : {OUTPUT_PATH} ({size:.2f} MB)")


if __name__ == "__main__":
    main()
import argparse

import torch

from model import Model


def export(
    input_path: str = "model/model.pt",
    output_path: str = "model/model.onnx",
    quantitize
):
    model = torch.load(args.input, map_location=torch.device("cpu"))
    model.eval()
    input1 = torch.tensor([[1] * 128])
    input2 = torch.tensor([[1] * 128])
    torch.onnx.export(
        model,
        (input1, input2),
        args.output,
        export_params=True,
        opset_version=14,
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--input",
        type=str,
        default="model/model.pt",
        help="path for pt model",
    )
    argparser.add_argument(
        "--output",
        type=str,
        default="model/text_classifier.onnx",
        help="path for exported model",
    )
    args = argparser.parse_args()

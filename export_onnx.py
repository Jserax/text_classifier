import argparse

import torch

from model import Model


def export(
    input_path: str = "model/model.pt",
    output_path: str = "app/model.onnx",
):
    model = torch.load(input_path, map_location=torch.device("cpu"))
    model.eval()
    input1 = torch.tensor([[1] * 128], dtype=torch.int32)
    input2 = torch.tensor([[1] * 128], dtype=torch.int32)
    torch.onnx.export(
        model,
        (input1, input2),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input1", "input2"],
        output_names=["output"],
        dynamic_axes={
            "input1": {0: "batch_size"},
            "input2": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
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
        default="app/model.onnx",
        help="path for exported model",
    )
    args = argparser.parse_args()
    export(args.input, args.output)

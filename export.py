import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import onnx
from net import UNet,UNet_wavelet,rescspblock,rescspblockA


def main(args):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    net = torch.load(args.weight_path).to(device)
    net.eval()
    torch_input = torch.randn(1, 3, args.TRANSFROM_SCALES,args.TRANSFROM_SCALES)

    torch.onnx.export(
        net,                  # model to export
        (torch_input,),        # inputs of the model,
        args.OUTPUT_ONNX,        # filename of the ONNX model
        input_names=["input"],  # Rename inputs for the ONNX model
        opset_version=16,
    )
    onnx_model = onnx.load(args.OUTPUT_ONNX)
    onnx.checker.check_model(onnx_model)
    # onnx.checker.check_model(onnx_model)

def opt_args():
    args = argparse.ArgumentParser()
    args.add_argument('--weight_path', type=str, default="output/RESIDE-6K_UNet/model_best.pth",
                      help='Path to model weight')
    args.add_argument('--OUTPUT_ONNX', type=str, default="output_UNet.onnx",
                      help='Output onnx')
    args.add_argument('--TRANSFROM_SCALES', type=int, default=256,
                      help='train img size')
    return args.parse_args()


if __name__ == '__main__':
    opt = opt_args()
    main(opt)
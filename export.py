import argparse
import torch
import os
import torch.nn as nn
import torch.nn.functional as F

import onnx


def main(args):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.half:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    model_path = args.weight_path.rsplit("/", 1)[0]
    if os.path.exists(os.path.join(model_path, "net.py")):
        print("Loading weights from {}".format(model_path))
        import importlib
        import sys
        spec = importlib.util.spec_from_file_location("net", os.path.join(model_path, "net.py"))
        net = importlib.util.module_from_spec(spec)
        sys.modules["net"] = net
        spec.loader.exec_module(net)
        if "wavelet" in args.weight_path:
            model = net.UNet_wavelet(block=net.resmspblock)
            model.load_state_dict(torch.load(args.weight_path)["state_dict"])
        else:
            model = net.UNet(block=net.resmspblock)
            model.load_state_dict(torch.load(args.weight_path)["state_dict"])
    else:
        model = torch.load(args.weight_path)
    model.to(device)
    if args.half:
        model = model.half()
    model.eval()
    torch_input = torch.randn(1, 3, args.TRANSFROM_SCALES,args.TRANSFROM_SCALES).to(device)
    if args.half:
        torch_input = torch_input.to(device).half()

    torch.onnx.export(
        model,                  # model to export
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
    args.add_argument('--weight_path', type=str, default="output/RESIDE-6K_UNet_wavelet_12/model_best.pth",
                      help='Path to model weight')
    args.add_argument('--OUTPUT_ONNX', type=str, default="output_UNet_wavelet.onnx",
                      help='Output onnx')
    args.add_argument('--TRANSFROM_SCALES', type=int, default=256,
                      help='train img size')
    args.add_argument('--half', type=bool, default=True,
                      help='use float16')
    return args.parse_args()


if __name__ == '__main__':
    opt = opt_args()
    main(opt)
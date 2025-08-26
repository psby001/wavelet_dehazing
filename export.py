import argparse
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis
import onnx

from collections import OrderedDict


def main(args):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.half:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    # model_path = args.weight_path.rsplit("/", 1)[0]
    # if os.path.exists(os.path.join(model_path, "net.py")):
    #     print("Loading weights from {}".format(model_path))
    #     import importlib
    #     import sys
    #     spec = importlib.util.spec_from_file_location("net", os.path.join(model_path, "net.py"))
    #     net = importlib.util.module_from_spec(spec)
    #     sys.modules["net"] = net
    #     spec.loader.exec_module(net)
    #     model = net.two_order_UNet_wavelet()
    #     if args.nano:
    #         new_state_dict = OrderedDict()
    #         state_dict =torch.load(args.weight_path)["state_dict"]
    #         for k, v in state_dict.items():
    #             if "ifm" in k:
    #                 continue
    #             name = k
    #             new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
    #         del model.ifm
    #         del model.ll_decoder.ifm
    #         del model.detail_decoder.ifm
    #
    #         model.load_state_dict(new_state_dict)
    #         model.idwt = True
    #         model.out_ = True
    #     else:
    #         model.load_state_dict(torch.load(args.weight_path,map_location="cpu")["state_dict"])
    #         model.out_ = True

    import net
    model = net.two_order_UNet_wavelet()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params)
    if args.half:
        model = model.half()
    model.eval()
    # model = model.to(device)
    # script_model = torch.jit.script(model)
    # torch.jit.save(script_model, f"mp_s3_si_litedec3_{device}.zip")
    torch_input = torch.randn(1, 3, 640,480).to(device)
    if args.half:
        torch_input = torch_input.to(device).half()
    flops = FlopCountAnalysis(model, torch_input)
    print(flops.total() / 1000 / 1000 / 1000)
    torch.onnx.export(
        model,                  # model to export
        (torch_input,),        # inputs of the model,
        args.OUTPUT_ONNX,        # filename of the ONNX model
        input_names=["input"],  # Rename inputs for the ONNX model
        # opset_version=16,
    )
    onnx_model = onnx.load(args.OUTPUT_ONNX)
    onnx.checker.check_model(onnx_model)
    # onnx.checker.check_model(onnx_model)

def opt_args():
    args = argparse.ArgumentParser()
    args.add_argument('--weight_path', type=str, default="output/RESIDE-6K_two_order_UNet_wavelet_01/model_best.pth",
                      help='Path to model weight')
    args.add_argument('--OUTPUT_ONNX', type=str, default="super_lite_w.onnx",
                      help='Output onnx')
    args.add_argument('--TRANSFROM_SCALES', type=int, default=256,
                      help='img size')
    args.add_argument('--half', type=bool, default=False,
                      help='use float16')
    args.add_argument('--nano', type=bool, default=False,
                      help='export to nano')
    return args.parse_args()


if __name__ == '__main__':
    opt = opt_args()
    main(opt)
import torch
import onnx

def dcp(img, omega=0.75):
    h, w = img.shape[2:]
    imsz = h * w
    # 要查找的是暗通道中前0.1%的值
    numpx = max(imsz // 1000, 1)
    # 找到暗通道的索引，弄成[batch, 3, numpx]，因为要匹配三个通道，所以需要expand
    dark = torch.min(img, dim=1, keepdim=True)[0]
    indices = torch.topk(dark.view(-1, imsz), k=numpx, dim=1)[1].view(-1, 1, numpx).expand(-1, 3, -1)
    # 用上述索引匹配原图中的3个通道，并求其平均值
    a = (torch.gather(img.view(-1, 3, imsz), 2, indices).sum(2) / numpx).view(-1, 3, 1, 1) + 1e-6  # 怕a为0，加个eps

    # 代公式算tx
    tx = 1 - omega * torch.min(img / a.view(-1, 3, 1, 1), dim=1, keepdim=True)[0]
    # 代公式算jx
    return (img - a) / torch.clamp_min(tx, 0.1) + a

class DCP(torch.nn.Module):
    def __init__(self, omega):
        super().__init__()
        self._omega = omega

    def forward(self, x):
        return dcp(x, self._omega)


if __name__ == '__main__':
    OUTPUT_ONNX = "dcp.onnx"
    model = DCP(omega=0.75)
    model.eval()
    torch_input = torch.randn(1, 3, 640, 480)

    torch.onnx.export(
        model,  # model to export
        (torch_input,),  # inputs of the model,
        OUTPUT_ONNX,  # filename of the ONNX model
        input_names=["input"],  # Rename inputs for the ONNX model
        # opset_version=16,
    )
    onnx_model = onnx.load(OUTPUT_ONNX)
    onnx.checker.check_model(onnx_model)
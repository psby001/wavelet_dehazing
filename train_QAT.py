import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.nn import functional as F
import tqdm
from torch.optim import lr_scheduler
import pytorch_ssim
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler
from torch import autocast
from SWT import SWTForward
from wavelet import wt_m,iwt_m
from IQA_pytorch import CW_SSIM
import shutil
import random
# import pytorch_quantization
# from pytorch_quantization import nn as quant_nn
# from pytorch_quantization import quant_modules
# from pytorch_quantization.tensor_quant import QuantDescriptor
# from pytorch_quantization import calib
from torchao.quantization.prototype.qat import Int8DynActInt4WeightQATQuantizer
def save_state_dict_with_model(state_dict,path,name,net = "net.py"):
    # torch.save(state_dict, os.path.join(path,name))
    torch.save({'state_dict': state_dict}, os.path.join(path, name))
    if not os.path.exists(os.path.join(path,net)):
        print('Saving in',path)
        shutil.copy2(net, os.path.join(path, "net.py"))


def set_seed(seed=2552, loader=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        loader.sampler.generator.manual_seed(seed)
    except AttributeError:
        pass
def check_path(path,n=0):
    if (not os.path.exists(path)) and n==0:
        return path
    elif not os.path.exists(path+"_{:0>2d}".format(n)):
        return path+"_{:0>2d}".format(n)
    else:
        n+=1
        return check_path(path,n)

def save_model(model,path,name):
    if (not os.path.exists(path)):
        os.makedirs(path)
    torch.save(model, os.path.join(path,name))
    # save_state_dict_with_model(model.state_dict(),path,name,net = net)

# def compute_amax(model, **kwargs):
#     # Load calib result
#     for name, module in model.named_modules():
#         if isinstance(module, quant_nn.TensorQuantizer):
#             if module._calibrator is not None:
#                 if isinstance(module._calibrator, calib.MaxCalibrator):
#                     module.load_calib_amax()
#                 else:
#                     module.load_calib_amax(**kwargs)
#             print(F"{name:40}: {module}")
#     model.cuda()
#
# def collect_stats(model, data_loader, num_batches):
#     """Feed data to the network and collect statistics"""
#     # Enable calibrators
#     for name, module in model.named_modules():
#         if isinstance(module, quant_nn.TensorQuantizer):
#             if module._calibrator is not None:
#                 module.disable_quant()
#                 module.enable_calib()
#             else:
#                 module.disable()
#     # Feed data to the network for collecting stats
#     for i, batch in tqdm.tqdm(enumerate(data_loader), total=num_batches):
#         img_idx, label_idx = batch["source"], batch["target"]
#         model(img_idx.cuda())
#         if i >= num_batches:
#             break
#
#     # Disable calibrators
#     for name, module in model.named_modules():
#         if isinstance(module, quant_nn.TensorQuantizer):
#             if module._calibrator is not None:
#                 module.enable_quant()
#                 module.disable_calib()
#             else:
#                 module.enable()
#
# def calibrate_model(model, model_name, data_loader, num_calib_batch, calibrator, hist_percentile, out_dir):
#     """
#         Feed data to the network and calibrate.
#         Arguments:
#             model: classification model
#             model_name: name to use when creating state files
#             data_loader: calibration data set
#             num_calib_batch: amount of calibration passes to perform
#             calibrator: type of calibration to use (max/histogram)
#             hist_percentile: percentiles to be used for historgram calibration
#             out_dir: dir to save state files in
#     """
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#     if num_calib_batch > 0:
#         print("Calibrating model")
#         with torch.no_grad():
#             collect_stats(model, data_loader, num_calib_batch)
#
#         if not calibrator == "histogram":
#             compute_amax(model, method="max")
#             calib_output = os.path.join(
#                 out_dir,
#                 F"{model_name}-max-{num_calib_batch*data_loader.batch_size}.pth")
#             torch.save(model.state_dict(), calib_output)
#         else:
#             for percentile in hist_percentile:
#                 print(F"{percentile} percentile calibration")
#                 compute_amax(model, method="percentile")
#                 calib_output = os.path.join(
#                     out_dir,
#                     F"{model_name}-percentile-{percentile}-{num_calib_batch*data_loader.batch_size}.pth")
#                 torch.save(model.state_dict(), calib_output)
#
#             for method in ["mse", "entropy"]:
#                 print(F"{method} calibration")
#                 compute_amax(model, method=method)
#                 calib_output = os.path.join(
#                     out_dir,
#                     F"{model_name}-{method}-{num_calib_batch*data_loader.batch_size}.pth")
#                 torch.save(model.state_dict(), calib_output)


def main(args,seed = 2552,n = ""):
    set_seed(seed)
    from dz_datasets.loader import PairLoader
    train_data = PairLoader(args.DATA_PATH, 'train', 'train',
                               (args.TRANSFROM_SCALES,args.TRANSFROM_SCALES))
    train_loader = Data.DataLoader(train_data, batch_size=args.BATCH_SIZE,
                                 shuffle=True, num_workers=4, pin_memory=True)
    val_data = PairLoader(args.DATA_PATH, 'test', 'valid',
                               (args.TRANSFROM_SCALES,args.TRANSFROM_SCALES))
    val_loader = Data.DataLoader(val_data, batch_size=args.BATCH_SIZE,
                                 shuffle=False, num_workers=4, pin_memory=True)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    sfm = wt_m()
    # sfm = SWTForward()
    # sfm = SWTForward()
    # net = UNet(block=resmspblock)
    # quant_modules.initialize()
    if args.weight_path is not None:
        print("Loading weights from {}".format(args.weight_path))
        import importlib
        import sys
        model_path = args.weight_path.rsplit("/", 1)[0]
        spec = importlib.util.spec_from_file_location("net", os.path.join(model_path, "net.py"))
        model = importlib.util.module_from_spec(spec)
        sys.modules["net"] = model
        spec.loader.exec_module(model)
        net = model.UNet_wavelet()
        net.load_state_dict(torch.load(args.weight_path)["state_dict"])
    else:
        # import net
        import importlib
        import sys
        model_path = "./"
        spec = importlib.util.spec_from_file_location("net", os.path.join(model_path, "net"+n+".py"))
        model = importlib.util.module_from_spec(spec)
        sys.modules["net"] = model
        spec.loader.exec_module(model)
        net = model.UNet_wavelet()

    model_name = net.__class__.__name__
    print(model_name)
    net.to(device)
    qat_quantizer = Int8DynActInt4WeightQATQuantizer()
    net = qat_quantizer.prepare(net)
    dataset_name = args.DATA_PATH.split('/')[-1]
    SAVE_PATH = os.path.join("./output", dataset_name+"_"+model_name)
    ab_test_dir = check_path(SAVE_PATH)
    # with torch.no_grad():
    #     calibrate_model(
    #         model=net,
    #         model_name="wavelet_Unet",
    #         data_loader=train_loader,
    #         num_calib_batch=args.BATCH_SIZE,
    #         calibrator="max",
    #         hist_percentile=[99.9, 99.99, 99.999, 99.9999],
    #         out_dir=ab_test_dir)

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(pytorch_total_params)
    net.to(device)
    sfm.to(device)

    criterion1 = torch.nn.L1Loss()
    criterion2 = nn.MSELoss()
    # Cw_SsimLoss_D = CW_SSIM(imgSize=(args.TRANSFROM_SCALES//2, args.TRANSFROM_SCALES//2), channels=9, level=4, ori=8).to(device)
    SsimLoss = pytorch_ssim.SSIM().to(device)


    # Optimizer
    # optimizer = torch.optim.SGD(net.parameters(), lr=args.INIT_LEARNING_RATE,momentum = args.MOMENTUM, weight_decay=args.DECAY)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.INIT_LEARNING_RATE)

    # Scheduler, For each 50 epoch, decay 0.1
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.MAX_EPOCHS//3, gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_ITER,
    #                                                        eta_min=INIT_LEARNING_RATE * 1e-2)

    print("train_loader", len(train_loader))
    print("val_loader", len(val_loader))

    loss_pre = 1000
    bestpsnr = 0
    max_id = 0
    train_loss = []
    train_psnr = []
    val_loss = []
    val_psnr = []
    val_ssmi = []

    for epoch in range(args.MAX_EPOCHS):
        print("epoch", epoch, ", learning rate: ", optimizer.param_groups[0]['lr'])
        # model.train()
        
        # Start to train
        # model_engine.train()
        total_loss = []
        total_psnr = []
        step_loss = []
        step_psnr = []
        net.train()
        optimizer.zero_grad()
        pbar = tqdm.tqdm(enumerate(train_loader))
        # if epoch != 0:
        #     net.reset_arg()
        net.to(device)
        net.train()
        for batch_idx, batch in pbar:
            optimizer.zero_grad()
            step = epoch * len(train_loader) + batch_idx
            img_idx, label_idx = batch["source"] , batch["target"]
            img = img_idx.to(device)
            label = label_idx.to(device)
            coeffs = sfm(label)
            ll_label = coeffs[:, [0, 4, 8]]
            # ll_scale1_label, ll_scale2_label = net.ll_scale(label)
            detail_label = coeffs[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
            # label = detail_label
            out = net(img)
            # out_label = net(label)
            output_map = out[0]
            out_ll,out_detail = out[1]
            # out_dec2 = out[2]
            # out_dec3 = out[3]
            # label_out_dec2 = out_label[2]
            # label_out_dec3 = out_label[3]
            # loss = criterion(output_map, label)
            # (output_l,enc1_l, enc2_l, enc3_l, enc4_l) = net(label)
            # loss = criterion(output_map, label)
            # loss = 0.5 * criterion(out_ll,ll_label) + 0.5 * criterion(out_detail,detail_label) + 0.5 * criterion(out_ll_scale1,ll_scale1_label) + 0.5 * criterion(out_ll_scale2,ll_scale2_label) + criterion(output_map,label)
            loss1 = criterion1(output_map, label) +  0.5 * criterion1(out_ll, ll_label) +  0.5 * criterion1(out_detail, detail_label)
            loss2 = criterion2(output_map, label) +  0.5 * criterion2(out_ll, ll_label) +  0.5 * criterion2(out_detail, detail_label)
            # loss3 = 0.1 * Cw_SsimLoss_D(out_detail, detail_label)
            # loss2 = 0.4 * criterion2(out_dec2,label_out_dec2) + 0.2  * criterion2(out_dec3,label_out_dec3)
            loss = loss1 + loss2
            # loss = criterion(output_map, label) + criterion(out_ll, ll_label) + 0.1 * Cw_SsimLoss_D(out_detail,detail_label) + criterion(out_detail, detail_label)
            psnr = 10 * torch.log10(1 / F.mse_loss(output_map, label)).item()
            # ssmi = SsimLoss(output_map, label)
            step_loss.append(loss.item())
            step_psnr.append(psnr)
            total_loss.append(loss.item())
            total_psnr.append(psnr)
            # total_ssmi.append(ssmi.item())


            # loss.backward()
            # optimizer.step()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # if (batch_idx) % 100 == 0:
            #     print(" Epoch [%d/%d] Loss: %.4f PSNR: %.4f lr: %f" % (epoch, batch_idx, np.mean(step_loss),np.mean(step_psnr),optimizer.param_groups[0]['lr']))
            #     step_loss = []
            #     step_psnr = []
        scheduler.step()
        # if len(train_loader) % final_batch != 0:
        #     scaler.step(optimizer)
        #     scaler.update()
        #     scheduler.step()
        #     optimizer.zero_grad()
        train_loss.append(np.mean(total_loss))
        train_psnr.append(np.mean(total_psnr))
        # scheduler.step()

        loss_total = []
        psnr = []
        ssmi = []
        net.eval()
        for batch_idx, batch in tqdm.tqdm(enumerate(val_loader)):
            with (torch.no_grad()):
                img_idx, label_idx = batch["source"], batch["target"]
                img = img_idx.to(device)
                label = label_idx.to(device)
                coeffs = sfm(label)
                ll_label = coeffs[:, [0, 4, 8]]
                # ll_scale1_label, ll_scale2_label = net.ll_scale(label)
                detail_label = coeffs[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]
                # label = detail_label
                out = net(img)
                output_map = out[0]
                out_ll, out_detail = out[1]
                # loss = criterion(output_map, label)
                # (output_l,enc1_l, enc2_l, enc3_l, enc4_l) = net(label)
                # loss = criterion(output_map, label)
                # loss = 0.5 * criterion(out_ll,ll_label) + 0.5 * criterion(out_detail,detail_label) + 0.5 * criterion(out_ll_scale1,ll_scale1_label) + 0.5 * criterion(out_ll_scale2,ll_scale2_label) + criterion(output_map,label)
                loss1 = criterion1(output_map, label) + 0.5 * criterion1(out_ll, ll_label) + 0.5 * criterion1(
                    out_detail, detail_label)
                loss2 = criterion2(output_map, label) + 0.5 * criterion2(out_ll, ll_label) + 0.5 * criterion2(
                    out_detail, detail_label)
                loss = loss1 + loss2
                # loss = criterion(output_map, label) + criterion(out_ll, ll_label) + 0.1 * Cw_SsimLoss_D(
                #     out_detail, detail_label) + criterion(out_detail, detail_label)
                ssmi.append(SsimLoss(output_map, label).item())
                psnr.append(10 * torch.log10(1 / F.mse_loss(output_map, label)).item())
                loss_total.append(loss.item())

                # miou.append(MIOU(output_map, gt, smooth=1e-10, n_classes=2))

        val_loss.append(np.mean(loss_total))
        val_psnr.append(np.mean(psnr))
        val_ssmi.append(np.mean(ssmi))
        print("############################")
        print("loss_avg ", np.mean(loss_total))
        print("PSNR ", np.mean(psnr))
        print("SSMI ", np.mean(ssmi))

        print('\n')
        convert_model = qat_quantizer.convert(net)
        if np.mean(loss_total) < loss_pre:
            loss_pre = np.mean(loss_total)
            # torch.save({'state_dict': model_engine.state_dict()}, os.path.join(ab_test_dir, 'model_best.pth'))
            max_id = epoch
            save_model(convert_model, ab_test_dir, 'model_best.pth')
        if epoch % 10 == 0:
            # torch.save({'state_dict': model_engine.state_dict()}, os.path.join(ab_test_dir, 'model' + str(epoch) + '.pth'))
            save_model(convert_model, ab_test_dir, 'model' + str(epoch) + '.pth')
        if np.nanmean(psnr) > bestpsnr:
            bestpsnr = np.nanmean(psnr)
    print("bestpsnr ", bestpsnr, " in {} epoch".format(max_id))
        

def opt_args():
    args = argparse.ArgumentParser()
    args.add_argument('--DATA_PATH', type=str, default="/mnt/d/Train Data/dz_data/RESIDE-6K",
                      help='Path to Dataset')
    args.add_argument('--weight_path', type=str,default="./output/RESIDE-6K_UNet_wavelet_113/model_best.pth", help='Path to model weight')
    args.add_argument('--DECAY', type=float, default=5e-5)
    args.add_argument('--MOMENTUM', type=float, default=0.90)
    args.add_argument('--OUTPUT_PATH', type=str, default="./output",
                      help='Output Path')
    args.add_argument('--BATCH_SIZE', type=int, default=32,
                        help='Batch size')
    args.add_argument('--TRANSFROM_SCALES', type=int, default=256,
                      help='train img size')
    args.add_argument('--INIT_LEARNING_RATE', type=float, default= 5e-4,
                      help='Init learning rate')
    args.add_argument('--MAX_EPOCHS', type=int, default=100,
                      help='train Epochs')
    return args.parse_args()

if __name__ == '__main__':
    opt = opt_args()
    # for n in ["1","2","3"]:
    #     main(opt,n=n)
    # for n in [2552,2553,2554]:
    #     main(opt,seed=n)
    main(opt)
import os
import sys
import csv
import time
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from utils import torchPSNR
from model import RSHazeNet
from torch.cuda.amp import autocast, GradScaler
from datasets import *
from options import Options
import torch.nn.functional as F
from pytorch_msssim import ssim


def calculate_sam(img1, img2):
    img1 = img1.float()
    img2 = img2.float()
    eps = 1e-8
    dot = (img1 * img2).sum(dim=1)
    norm1 = img1.pow(2).sum(dim=1).sqrt()
    norm2 = img2.pow(2).sum(dim=1).sqrt()
    cos = dot / (norm1 * norm2 + eps)
    cos = cos.clamp(-1, 1)
    sam = torch.acos(cos).mean()
    return torch.degrees(sam).item()


def calculate_ergas(img1, img2, ratio=1):
    img1 = img1.float()
    img2 = img2.float()
    mean_ref = img2.mean(dim=(2, 3), keepdim=True)
    rmse = (img1 - img2).pow(2).mean(dim=(2, 3), keepdim=True).sqrt()
    ergas = 100 / ratio * ((rmse / (mean_ref + 1e-8)).pow(2).mean(dim=1)).sqrt().mean()
    return ergas.item()


def expand2square(timg, factor=16.0):
    _, _, h, w = timg.size()
    X = int(max(h, w) / factor + 0.5) * factor
    img = torch.zeros(1, 3, X, X).type_as(timg)
    mask = torch.zeros(1, 1, X, X).type_as(timg)
    img[:, :, (X - h) // 2:(X - h) // 2 + h, (X - w) // 2:(X - w) // 2 + w] = timg
    mask[:, :, (X - h) // 2:(X - h) // 2 + h, (X - w) // 2:(X - w) // 2 + w].fill_(1)
    return img, mask


if __name__ == '__main__':

    opt = Options()
    cudnn.benchmark = True

    best_psnr = 0
    best_epoch = 0

    myNet = RSHazeNet()
    if opt.CUDA_USE:
        myNet = myNet.cuda()

    optimizer = optim.Adam(myNet.parameters(), lr=opt.Learning_Rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.Epoch, eta_min=1e-8)

    datasetTrain = MyTrainDataSet(opt.Input_Path_Train, opt.Target_Path_Train, patch_size=opt.Patch_Size_Train)
    trainLoader = DataLoader(dataset=datasetTrain, batch_size=opt.Batch_Size_Train, shuffle=True,
                             drop_last=True, num_workers=opt.Num_Works, pin_memory=True)

    datasetValue = MyValueDataSet(opt.Input_Path_Val, opt.Target_Path_Val, patch_size=opt.Patch_Size_Val)
    valueLoader = DataLoader(dataset=datasetValue, batch_size=opt.Batch_Size_Val, shuffle=False,
                             drop_last=False, num_workers=opt.Num_Works, pin_memory=True)

    csv_path = os.path.join(opt.MODEL_SAVE_PATH, 'training_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_psnr', 'val_ssim', 'val_sam', 'val_ergas'])

    print('-------------------------------------------------------------------------------------------------------')
    if opt.MODEL_PRE_PATH and os.path.exists(opt.MODEL_PRE_PATH):
        if opt.CUDA_USE:
            myNet.load_state_dict(torch.load(opt.MODEL_PRE_PATH))
        else:
            myNet.load_state_dict(torch.load(opt.MODEL_PRE_PATH, map_location=torch.device('cpu')))

    scaler = GradScaler()
    for epoch in range(opt.Epoch):
        myNet.train()
        iters = tqdm(trainLoader, file=sys.stdout)
        epochLoss = 0
        timeStart = time.time()
        for index, (x, y) in enumerate(iters, 0):

            myNet.zero_grad()
            optimizer.zero_grad()

            if opt.CUDA_USE:
                input_train, target = Variable(x).cuda(), Variable(y).cuda()
            else:
                input_train, target = Variable(x), Variable(y)

            with autocast(True):
                restored = myNet(input_train)
                loss = F.mse_loss(restored, target)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epochLoss += loss.item()
            iters.set_description('Training !!!  Epoch %d / %d,  Batch Loss %.6f' % (epoch+1, opt.Epoch, loss.item()))

        avg_loss = epochLoss / len(trainLoader)

        if (epoch + 1) % opt.Val_Freq == 0 or epoch == 0:
            myNet.eval()
            psnr_list, ssim_list, sam_list, ergas_list = [], [], [], []
            for index, (x, y) in enumerate(valueLoader, 0):
                input_, target_value = (x.cuda(), y.cuda()) if opt.CUDA_USE else (x, y)
                with torch.no_grad():
                    _, _, h, w = input_.shape
                    input_pad, mask = expand2square(input_, factor=128)
                    output_value = myNet(input_pad).clamp_(-1, 1)
                    output_value = output_value * 0.5 + 0.5
                    target_value = target_value * 0.5 + 0.5
                    output_value = torch.masked_select(output_value, mask.bool()).reshape(1, 3, h, w)

                    mse_val = F.mse_loss(output_value, target_value)
                    psnr_val = 10 * torch.log10(1 / mse_val).item()
                    psnr_list.append(psnr_val)

                    _, _, H, W = output_value.size()
                    down_ratio = max(1, round(min(H, W) / 256))
                    ssim_val = ssim(F.adaptive_avg_pool2d(output_value, (int(H / down_ratio), int(W / down_ratio))),
                                    F.adaptive_avg_pool2d(target_value, (int(H / down_ratio), int(W / down_ratio))),
                                    data_range=1, size_average=False).item()
                    ssim_list.append(ssim_val)

                    sam_val = calculate_sam(output_value, target_value)
                    sam_list.append(sam_val)

                    ergas_val = calculate_ergas(output_value, target_value)
                    ergas_list.append(ergas_val)

            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)
            avg_sam = np.mean(sam_list)
            avg_ergas = np.mean(ergas_list)

            print(f"\n[Val Epoch {epoch+1}] PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | SAM: {avg_sam:.4f} | ERGAS: {avg_ergas:.4f}")

            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, avg_loss, avg_psnr, avg_ssim, avg_sam, avg_ergas])

            if avg_psnr >= best_psnr:
                best_psnr = avg_psnr
                best_epoch = epoch
                torch.save(myNet.state_dict(), os.path.join(opt.MODEL_SAVE_PATH, 'model_best.pth'))

        if (epoch + 1) % opt.Save_Freq == 0:
            torch.save(myNet.state_dict(), os.path.join(opt.MODEL_SAVE_PATH, f'model_{epoch+1}.pth'))

        scheduler.step()
        timeEnd = time.time()
        print("------------------------------------------------------------")
        print("Epoch:  {}  Finished,  Time:  {:.4f} s,  Loss:  {:.6f}, best psnr:  {:.3f}.".format(
                epoch + 1, timeEnd - timeStart, avg_loss, best_psnr))
        print('-------------------------------------------------------------------------------------------------------')
    print("Training Process Finished ! Best Epoch : {} , Best PSNR : {:.2f}".format(best_epoch, best_psnr))

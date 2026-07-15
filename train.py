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
    return torch.rad2deg(sam).item()


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


def collate_train(batch):
    """Collate that returns captions (list[str]) and levels alongside tensors."""
    inputs = torch.stack([b[0] for b in batch], 0)
    targets = torch.stack([b[1] for b in batch], 0)
    captions = [b[2] for b in batch]
    levels = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return inputs, targets, captions, levels


def collate_eval(batch):
    inputs = torch.stack([b[0] for b in batch], 0)
    targets = torch.stack([b[1] for b in batch], 0)
    captions = [b[2] for b in batch]
    levels = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return inputs, targets, captions, levels


def load_full_checkpoint(net, optimizer, scheduler, path, use_cuda, device):
    """Load a full-state checkpoint (or a bare state_dict for backwards compat).

    Returns (start_epoch, best_psnr, best_epoch).
    """
    map_location = torch.device('cpu') if not use_cuda else None
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        net.load_state_dict(ckpt['model'], strict=False)
        if optimizer is not None and 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if scheduler is not None and 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0)
        best_psnr = ckpt.get('best_psnr', 0.0)
        best_epoch = ckpt.get('best_epoch', 0)
        print(f"[Resume] Loaded full checkpoint from epoch {start_epoch} | "
              f"best PSNR {best_psnr:.4f} (epoch {best_epoch}).")
        return start_epoch, best_psnr, best_epoch
    # Bare state_dict (older save or weights-only --pretrained)
    net.load_state_dict(ckpt, strict=False)
    print("[Resume] Loaded bare state_dict (no optimizer/scheduler/epoch).")
    return 0, 0.0, 0


def save_full_checkpoint(net, optimizer, scheduler, epoch, best_psnr, best_epoch, path):
    torch.save({
        'epoch': epoch,
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_psnr': best_psnr,
        'best_epoch': best_epoch,
        'arch': 'RSHazeNet-vlm',
    }, path)


if __name__ == '__main__':

    opt = Options()
    cudnn.benchmark = True

    best_psnr = 0
    best_epoch = 0
    start_epoch = 0

    myNet = RSHazeNet()
    if opt.CUDA_USE:
        myNet = myNet.cuda()

    optimizer = optim.Adam(myNet.parameters(), lr=opt.Learning_Rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.Epoch, eta_min=1e-8)

    # ---- Caption-based conditioning setup ---------------------------------
    text_encoder = None
    use_caption = opt.Use_Caption
    caption_map_train = {}
    caption_map_val = {}
    if use_caption:
        from text_encoder import CaptionTextEncoder
        from datasets import load_caption_map
        device = 'cuda' if opt.CUDA_USE else 'cpu'
        text_encoder = CaptionTextEncoder(device=device, model_name=opt.CLIP_Model)
        caption_map_train = load_caption_map(opt.Caption_Path_Train)
        caption_map_val = load_caption_map(opt.Caption_Path_Val)
        if not caption_map_train:
            print(f"[Caption] WARNING: no training captions found at {opt.Caption_Path_Train}.\n"
                  f"  Run `python caption.py --input_dir <train_hazy>` first, or pass --caption_train.")
            use_caption = False
        else:
            print(f"[Caption] Loaded {len(caption_map_train)} train captions, "
                  f"{len(caption_map_val)} val captions.")

    datasetTrain = MyTrainDataSet(opt.Input_Path_Train, opt.Target_Path_Train,
                                 patch_size=opt.Patch_Size_Train, caption_map=caption_map_train)
    trainLoader = DataLoader(dataset=datasetTrain, batch_size=opt.Batch_Size_Train, shuffle=True,
                             drop_last=True, num_workers=opt.Num_Works, pin_memory=True,
                             collate_fn=collate_train)

    datasetValue = MyValueDataSet(opt.Input_Path_Val, opt.Target_Path_Val,
                                 patch_size=opt.Patch_Size_Val, caption_map=caption_map_val)
    valueLoader = DataLoader(dataset=datasetValue, batch_size=opt.Batch_Size_Val, shuffle=False,
                             drop_last=False, num_workers=opt.Num_Works, pin_memory=True,
                             collate_fn=collate_eval)

    # ---- Checkpoint loading (resume takes precedence over pretrained) -----
    if opt.RESUME_PATH and os.path.exists(opt.RESUME_PATH):
        start_epoch, best_psnr, best_epoch = load_full_checkpoint(
            myNet, optimizer, scheduler, opt.RESUME_PATH, opt.CUDA_USE, 'cuda' if opt.CUDA_USE else 'cpu')
    elif opt.MODEL_PRE_PATH and os.path.exists(opt.MODEL_PRE_PATH):
        load_full_checkpoint(myNet, None, None, opt.MODEL_PRE_PATH, opt.CUDA_USE,
                            'cuda' if opt.CUDA_USE else 'cpu')

    # ---- Metrics CSV (append on resume) -----------------------------------
    csv_path = os.path.join(opt.MODEL_SAVE_PATH, 'training_metrics.csv')
    write_header = not (opt.RESUME_PATH and os.path.exists(opt.RESUME_PATH) and os.path.exists(csv_path))
    if write_header or not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_psnr', 'val_ssim', 'val_sam', 'val_ergas'])
    else:
        print(f"[Resume] Appending metrics to existing {csv_path}")

    print('-------------------------------------------------------------------------------------------------------')
    if use_caption:
        print(f"[Caption] Conditioning enabled (CLIP={opt.CLIP_Model}, haze_weight={opt.Haze_Weight}).")
    else:
        print("[Caption] Conditioning disabled (plain RSHazeNet).")

    scaler = GradScaler()
    for epoch in range(start_epoch, opt.Epoch):
        myNet.train()
        iters = tqdm(trainLoader, file=sys.stdout)
        epochLoss = 0
        timeStart = time.time()
        for index, (x, y, captions, levels) in enumerate(iters, 0):

            myNet.zero_grad()
            optimizer.zero_grad()

            if opt.CUDA_USE:
                input_train, target = Variable(x).cuda(), Variable(y).cuda()
                levels = levels.cuda()
            else:
                input_train, target = Variable(x), Variable(y)

            text_tokens = None
            if use_caption and text_encoder is not None:
                text_tokens = text_encoder.encode_captions(captions)  # (B, N, 512), no grad

            with autocast(True):
                restored = myNet(input_train, text_tokens)
                loss = F.mse_loss(restored, target)

                if opt.Haze_Weight > 0:
                    w = 1.0 + opt.Haze_Weight * (levels.float() / 3.0)
                    loss = (w.view(-1) * ((restored - target) ** 2).mean(dim=[1, 2, 3])).mean()

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
            for index, (x, y, captions, levels) in enumerate(valueLoader, 0):
                input_batch, target_batch = (x.cuda(), y.cuda()) if opt.CUDA_USE else (x, y)

                for i in range(input_batch.size(0)):
                    input_ = input_batch[i:i+1]
                    target_value = target_batch[i:i+1]

                    text_tokens = None
                    if use_caption and text_encoder is not None:
                        text_tokens = text_encoder.encode_captions([captions[i]])

                    with torch.no_grad():
                        _, _, h, w = input_.shape
                        input_pad, mask = expand2square(input_, factor=128)
                        output_value = myNet(input_pad, text_tokens).clamp_(-1, 1)
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

            avg_psnr = float(np.mean(psnr_list))
            avg_ssim = float(np.mean(ssim_list))
            avg_sam = float(np.mean(sam_list))
            avg_ergas = float(np.mean(ergas_list))

            print(f"\n[Val Epoch {epoch+1}] PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | SAM: {avg_sam:.4f} | ERGAS: {avg_ergas:.4f}")

            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, avg_loss, avg_psnr, avg_ssim, avg_sam, avg_ergas])

            if avg_psnr >= best_psnr:
                best_psnr = avg_psnr
                best_epoch = epoch
                save_full_checkpoint(myNet, optimizer, scheduler, epoch + 1, best_psnr, best_epoch,
                                     os.path.join(opt.MODEL_SAVE_PATH, 'model_best.pth'))

        if (epoch + 1) % opt.Save_Freq == 0:
            save_full_checkpoint(myNet, optimizer, scheduler, epoch + 1, best_psnr, best_epoch,
                                 os.path.join(opt.MODEL_SAVE_PATH, f'model_{epoch+1}.pth'))

        scheduler.step()
        timeEnd = time.time()
        print("------------------------------------------------------------")
        print("Epoch:  {}  Finished,  Time:  {:.4f} s,  Loss:  {:.6f}, best psnr:  {:.3f}.".format(
                epoch + 1, timeEnd - timeStart, avg_loss, best_psnr))
        print('-------------------------------------------------------------------------------------------------------')
    print("Training Process Finished ! Best Epoch : {} , Best PSNR : {:.2f}".format(best_epoch, best_psnr))
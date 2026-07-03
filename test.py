import sys
import time
import cv2
import math
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from pytorch_msssim import ssim
from model import RSHazeNet
from skimage import img_as_ubyte
from datasets import MyTestDataSet, load_caption_map
from options import Options


def expand2square(timg, factor=16.0):
    _, _, h, w = timg.size()
    X = int(math.ceil(max(h, w) / float(factor)) * factor)
    img = torch.zeros(1, 3, X, X).type_as(timg)
    mask = torch.zeros(1, 1, X, X).type_as(timg)
    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1)
    return img, mask


def collate_test(batch):
    inputs = torch.stack([b[0] for b in batch], 0)
    targets = torch.stack([b[1] for b in batch], 0)
    names = [b[2] for b in batch]
    captions = [b[3] for b in batch]
    levels = torch.tensor([b[4] for b in batch], dtype=torch.long)
    return inputs, targets, names, captions, levels


if __name__ == '__main__':

    opt = Options()

    myNet = RSHazeNet()
    if opt.CUDA_USE:
        myNet = myNet.cuda()

    text_encoder = None
    use_caption = opt.Use_Caption
    caption_map_test = {}
    if use_caption:
        from text_encoder import CaptionTextEncoder
        device = 'cuda' if opt.CUDA_USE else 'cpu'
        text_encoder = CaptionTextEncoder(device=device, model_name=opt.CLIP_Model)
        caption_map_test = load_caption_map(opt.Caption_Path_Test)
        if not caption_map_test:
            print(f"[Caption] WARNING: no test captions found at {opt.Caption_Path_Test}.")
            use_caption = False
        else:
            print(f"[Caption] Loaded {len(caption_map_test)} test captions.")

    datasetTest = MyTestDataSet(opt.Input_Path_Test, opt.Target_Path_Test, caption_map=caption_map_test)
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False,
                            num_workers=opt.Num_Works, pin_memory=True, collate_fn=collate_test)

    print('--------------------------------------------------------------')
    ckpt = torch.load(opt.MODEL_PRE_PATH,
                      map_location='cpu' if not opt.CUDA_USE else None)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    myNet.load_state_dict(state, strict=False)
    myNet.eval()

    PSNR = 0
    SSIM = 0
    MSE = 0
    L = 0

    with torch.no_grad():
        timeStart = time.time()
        for index, (x, y, names, captions, levels) in enumerate(tqdm(testLoader, desc='Testing !!! ', file=sys.stdout), 0):
            torch.cuda.empty_cache()

            input_test = x.cuda() if opt.CUDA_USE else x
            target = y.cuda() if opt.CUDA_USE else y

            text_tokens = None
            if use_caption and text_encoder is not None:
                text_tokens = text_encoder.encode_captions(captions)

            _, _, h, w = input_test.shape
            input_test_pad, mask = expand2square(input_test, factor=128)
            restored_ = myNet(input_test_pad, text_tokens).clamp_(-1, 1)

            restored = restored_ * 0.5 + 0.5
            target = target * 0.5 + 0.5

            restored = torch.masked_select(restored, mask.bool()).reshape(1, 3, h, w)

            mse_val = F.mse_loss(restored, target)
            psnr_val = 10 * torch.log10(1 / mse_val).item()

            _, _, H, W = restored.size()
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(F.adaptive_avg_pool2d(restored, (int(H / down_ratio), int(W / down_ratio))),
                            F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                            data_range=1, size_average=False).item()

            MSE += mse_val.item()
            PSNR += psnr_val
            SSIM += ssim_val

            L = index + 1

            restored_out = restored_.cpu().numpy().squeeze().transpose((1, 2, 0))
            cv2.imwrite(opt.Result_Path_Test + names[0], cv2.cvtColor(img_as_ubyte(restored_out), cv2.COLOR_RGB2BGR))

        timeEnd = time.time()
        print('---------------------------------------------------------')
        print(PSNR / L, SSIM / L, MSE / L)
        print('---------------------------------------------------------')
        print("Testing Process Finished !!! Time: {:.4f} s".format(timeEnd - timeStart))
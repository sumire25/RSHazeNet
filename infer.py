import os
import sys
import math
import cv2
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import RSHazeNet
from datasets import MyTestDataSet, load_caption_map
from skimage import img_as_ubyte


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_input', type=str, required=True)
    parser.add_argument('--test_target', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--result_path', type=str, default='./results/')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--caption', action='store_true',
                        help='Enable VLM haziness-caption conditioning.')
    parser.add_argument('--caption_test', type=str, default='',
                        help='Path to captions.json for the test hazy images. '
                             'Defaults to <test_input>/captions.json.')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    myNet = RSHazeNet()
    if use_cuda:
        myNet = myNet.cuda()

    ckpt = torch.load(args.weights, map_location='cpu' if not use_cuda else None)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    myNet.load_state_dict(state, strict=False)
    myNet.eval()

    text_encoder = None
    use_caption = args.caption
    caption_map_test = {}
    if use_caption:
        from text_encoder import CaptionTextEncoder
        device = 'cuda' if use_cuda else 'cpu'
        text_encoder = CaptionTextEncoder(device=device, model_name=args.clip_model)
        cap_path = args.caption_test or os.path.join(args.test_input, 'captions.json')
        caption_map_test = load_caption_map(cap_path)
        if not caption_map_test:
            print(f"[Caption] WARNING: no test captions found at {cap_path}.")
            use_caption = False
        else:
            print(f"[Caption] Loaded {len(caption_map_test)} test captions.")

    os.makedirs(args.result_path, exist_ok=True)

    datasetTest = MyTestDataSet(args.test_input, args.test_target, caption_map=caption_map_test)
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False, num_workers=2,
                           collate_fn=collate_test)

    with torch.no_grad():
        for index, (x, y, names, captions, levels) in enumerate(tqdm(testLoader, desc='Inference', file=sys.stdout)):
            torch.cuda.empty_cache()
            input_test = x.cuda() if use_cuda else x
            text_tokens = None
            if use_caption and text_encoder is not None:
                text_tokens = text_encoder.encode_captions(captions)
            _, _, h, w = input_test.shape
            input_test, mask = expand2square(input_test, factor=128)
            restored = myNet(input_test, text_tokens).clamp_(-1, 1)
            restored = torch.masked_select(restored, mask.bool()).reshape(1, 3, h, w)
            restored = restored.cpu().numpy().squeeze().transpose((1, 2, 0))
            cv2.imwrite(os.path.join(args.result_path, names[0]),
                        cv2.cvtColor(img_as_ubyte(restored), cv2.COLOR_RGB2BGR))
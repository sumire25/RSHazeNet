import os
import sys
import math
import cv2
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import RSHazeNet
from datasets import MyTestDataSet
from skimage import img_as_ubyte


def expand2square(timg, factor=16.0):
    _, _, h, w = timg.size()
    X = int(math.ceil(max(h, w) / float(factor)) * factor)
    img = torch.zeros(1, 3, X, X).type_as(timg)
    mask = torch.zeros(1, 1, X, X).type_as(timg)
    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1)
    return img, mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_input', type=str, required=True)
    parser.add_argument('--test_target', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--result_path', type=str, default='./results/')
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    myNet = RSHazeNet()
    if use_cuda:
        myNet = myNet.cuda()

    if use_cuda:
        myNet.load_state_dict(torch.load(args.weights))
    else:
        myNet.load_state_dict(torch.load(args.weights, map_location=torch.device('cpu')))
    myNet.eval()

    os.makedirs(args.result_path, exist_ok=True)

    datasetTest = MyTestDataSet(args.test_input, args.test_target)
    testLoader = DataLoader(dataset=datasetTest, batch_size=1, shuffle=False, drop_last=False, num_workers=2)

    with torch.no_grad():
        for index, (x, y, name) in enumerate(tqdm(testLoader, desc='Inference', file=sys.stdout)):
            torch.cuda.empty_cache()
            input_test = x.cuda() if use_cuda else x
            _, _, h, w = input_test.shape
            input_test, mask = expand2square(input_test, factor=128)
            restored = myNet(input_test).clamp_(-1, 1)
            restored = torch.masked_select(restored, mask.bool()).reshape(1, 3, h, w)
            restored = restored.cpu().numpy().squeeze().transpose((1, 2, 0))
            cv2.imwrite(os.path.join(args.result_path, name[0]), cv2.cvtColor(img_as_ubyte(restored), cv2.COLOR_RGB2BGR))

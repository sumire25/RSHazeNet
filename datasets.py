import os
import re
import random
import torch
import torchvision.transforms.functional as ttf
from torch.utils.data import Dataset
from PIL import Image


def _extract_id(filename):
    match = re.match(r'(\d+)', filename)
    return match.group(1) if match else filename


def _build_id_map(directory):
    return {_extract_id(f): f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))}


class MyTrainDataSet(Dataset):
    def __init__(self, inputPathTrain, targetPathTrain, patch_size=512):
        super(MyTrainDataSet, self).__init__()

        self.inputPath = inputPathTrain
        self.targetPath = targetPathTrain

        input_map = _build_id_map(inputPathTrain)
        target_map = _build_id_map(targetPathTrain)

        self.common_ids = sorted(set(input_map.keys()) & set(target_map.keys()), key=lambda x: int(x) if x.isdigit() else x)
        self.input_map = input_map
        self.target_map = target_map

        self.ps = patch_size

    def __len__(self):
        return len(self.common_ids)

    def __getitem__(self, index):

        ps = self.ps
        index = index % len(self.common_ids)

        img_id = self.common_ids[index]
        inputImagePath = os.path.join(self.inputPath, self.input_map[img_id])
        targetImagePath = os.path.join(self.targetPath, self.target_map[img_id])

        inputImage = Image.open(inputImagePath).convert('RGB')
        targetImage = Image.open(targetImagePath).convert('RGB')

        inputImage = ttf.to_tensor(inputImage)
        targetImage = ttf.to_tensor(targetImage)

        hh, ww = targetImage.shape[1], targetImage.shape[2]

        rr = random.randint(0, hh-ps)
        cc = random.randint(0, ww-ps)
        aug = random.randint(0, 8)

        input_ = inputImage[:, rr:rr+ps, cc:cc+ps]
        target = targetImage[:, rr:rr+ps, cc:cc+ps]

        if aug == 1:
            input_, target = input_.flip(1), target.flip(1)
        elif aug == 2:
            input_, target = input_.flip(2), target.flip(2)
        elif aug == 3:
            input_, target = torch.rot90(input_, dims=(1, 2)), torch.rot90(target, dims=(1, 2))
        elif aug == 4:
            input_, target = torch.rot90(input_, dims=(1, 2), k=2), torch.rot90(target, dims=(1, 2), k=2)
        elif aug == 5:
            input_, target = torch.rot90(input_, dims=(1, 2), k=3), torch.rot90(target, dims=(1, 2), k=3)
        elif aug == 6:
            input_, target = torch.rot90(input_.flip(1), dims=(1, 2)), torch.rot90(target.flip(1), dims=(1, 2))
        elif aug == 7:
            input_, target = torch.rot90(input_.flip(2), dims=(1, 2)), torch.rot90(target.flip(2), dims=(1, 2))

        return input_, target


class MyValueDataSet(Dataset):
    def __init__(self, inputPathTrain, targetPathTrain, patch_size=64):
        super(MyValueDataSet, self).__init__()

        self.inputPath = inputPathTrain
        self.targetPath = targetPathTrain

        input_map = _build_id_map(inputPathTrain)
        target_map = _build_id_map(targetPathTrain)

        self.common_ids = sorted(set(input_map.keys()) & set(target_map.keys()), key=lambda x: int(x) if x.isdigit() else x)
        self.input_map = input_map
        self.target_map = target_map

        self.ps = patch_size

    def __len__(self):
        return len(self.common_ids)

    def __getitem__(self, index):

        ps = self.ps
        index = index % len(self.common_ids)

        img_id = self.common_ids[index]
        inputImagePath = os.path.join(self.inputPath, self.input_map[img_id])
        targetImagePath = os.path.join(self.targetPath, self.target_map[img_id])

        inputImage = Image.open(inputImagePath).convert('RGB')
        targetImage = Image.open(targetImagePath).convert('RGB')

        inputImage = ttf.center_crop(inputImage, [ps, ps])
        targetImage = ttf.center_crop(targetImage, [ps, ps])

        input_ = ttf.to_tensor(inputImage)
        target = ttf.to_tensor(targetImage)

        return input_, target


class MyTestDataSet(Dataset):
    def __init__(self, inputPathTest, targetPathTest):
        super(MyTestDataSet, self).__init__()

        self.inputPath = inputPathTest
        self.targetPath = targetPathTest

        input_map = _build_id_map(inputPathTest)
        target_map = _build_id_map(targetPathTest)

        self.common_ids = sorted(set(input_map.keys()) & set(target_map.keys()), key=lambda x: int(x) if x.isdigit() else x)
        self.input_map = input_map
        self.target_map = target_map

    def __len__(self):
        return len(self.common_ids)

    def __getitem__(self, index):
        index = index % len(self.common_ids)

        img_id = self.common_ids[index]
        inputImagePath = os.path.join(self.inputPath, self.input_map[img_id])
        targetImagePath = os.path.join(self.targetPath, self.target_map[img_id])

        inputImage = Image.open(inputImagePath).convert('RGB')
        targetImage = Image.open(targetImagePath).convert('RGB')

        input_ = ttf.to_tensor(inputImage)
        target = ttf.to_tensor(targetImage)

        return input_, target, self.input_map[img_id]

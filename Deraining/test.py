"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
import torch.nn as nn
import torch

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from pathlib import Path
from skimage import img_as_ubyte
from tqdm import tqdm
from PIL import Image

import utils
from MPRNet import MPRNet

class RainData(Dataset):
    def __init__(self, dataset_dir):
        super().__init__()
        self.datasets = []
        p = Path(dataset_dir)
        for ext in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif']:
            self.datasets.extend(p.glob(f'*.{ext}'))

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        path = self.datasets[idx]
        img = Image.open(path)
        img = TF.to_tensor(img)
        return img, path.name

parser = argparse.ArgumentParser(description='Image Deraining using MPRNet')
parser.add_argument('--input_dir', default='datasets/Test100/input', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='results', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained/deraining.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = MPRNet()
utils.load_checkpoint(model_restoration, args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

rgb_dir_test = Path(args.input_dir)
test_loader = DataLoader(RainData(rgb_dir_test), pin_memory=True)
Path(args.result_dir).mkdir(exist_ok=True)

with torch.no_grad():
    for img, img_name in tqdm(test_loader):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        
        restored_ = model_restoration(img.cuda())
        restored_ = torch.clamp(restored_[0], 0, 1)
        restored_ = restored_.permute(0, 2, 3, 1).cpu().detach().numpy()

        for restored in restored_:
            restored = img_as_ubyte(restored)
            utils.save_img(Path(args.result_dir, img_name[0]).as_posix(), restored)
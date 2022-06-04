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

from torch.utils.data import DataLoader
from pathlib import Path
from skimage import img_as_ubyte
from tqdm import tqdm

import utils
from data_RGB import get_test_data
from MPRNet import MPRNet

parser = argparse.ArgumentParser(description='Image Deraining using MPRNet')

parser.add_argument('--input_dir', default='Datasets/Test100/input', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='results', type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained/deraining.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = MPRNet()

utils.load_checkpoint(model_restoration,args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

rgb_dir_test = os.path.join(args.input_dir, dataset, 'input')
rgb_dir_test = Path(args.input_dir)
print(rgb_dir_test)

# test_dataset = get_test_data(rgb_dir_test, img_options={})
# test_loader  = DataLoader(dataset=test_dataset, pin_memory=True)

# result_dir  = os.path.join(args.result_dir, dataset)
# utils.mkdir(result_dir)

# with torch.no_grad():
#     for data_test in tqdm(test_loader):
        
#         torch.cuda.ipc_collect()
#         torch.cuda.empty_cache()
        
#         input_    = data_test[0].cuda()
#         filenames = data_test[1]

#         restored = model_restoration(input_)
#         restored = torch.clamp(restored[0],0,1)

#         restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

#         for batch in range(len(restored)):
#             restored_img = img_as_ubyte(restored[batch])
#             utils.save_img((os.path.join(result_dir, filenames[batch]+'.png')), restored_img)

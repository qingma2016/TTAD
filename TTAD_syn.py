import numpy as np
import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
# Load test image from URL
import torchvision.transforms as transforms
import requests
from io import BytesIO
import os
from PIL import Image
import time
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as ssim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision.transforms.functional import to_pil_image
from arch_unet import UNet
import argparse
import math


torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = "cuda:0"

parser = argparse.ArgumentParser('Pixel2Pixel')
parser.add_argument('--data_path', default='/home/lpw/mq/data', type=str)
parser.add_argument('--dataset', default='kodak', type=str)
parser.add_argument('--save', default='./results', type=str)
parser.add_argument('--out_image', default='./results_image_pre', type=str)
parser.add_argument('--ws', default=34, type=int)
parser.add_argument('--ps', default=7, type=int)
parser.add_argument('--nn', default=16, type=int)
parser.add_argument('--nr', default=20, type=int)
parser.add_argument('--nt', default='gauss', type=str)
parser.add_argument('--nl', default=25.0, type=float)
parser.add_argument('--banknum', default=10, type=int)
parser.add_argument('--loss', default='L2', type=str)
args = parser.parse_args()


image_folder = os.path.join(args.data_path, args.dataset)
image_files = os.listdir(image_folder)
image_files = sorted(image_files)


sim_image_folder = os.path.join(args.save, '_'.join(str(i) for i in [args.dataset, args.nt,args.nl, args.ws, args.ps, args.nn,args.nr, args.loss]))
sim_image_files = [f for f in os.listdir(sim_image_folder) if f.endswith(('.png', '.jpg', '.jpeg','.PNG', '.npy'))]
sim_image_files = sorted(sim_image_files)

transform = transforms.Compose([transforms.ToTensor()])

if args.nt=='gauss' and args.nl==50:
    max_epoch=100
elif args.nt=='poiss' and args.nl==10:
    max_epoch=100
else:
    max_epoch = 10

lr = 0.0001


def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt, pred)

loss_l1 = nn.L1Loss()


def loss_func(img1, img2, loss_f=mse):
    pred1 = model(img1)
    loss = loss_f(img2, pred1)
    return loss

def train(model, optimizer, img_bank):
    model.train()
    # prepare a noise2noise pair
    index1 = torch.randint(0, N, size=(H * W, 1))
    img1 = torch.gather(img_bank, 0, index=index1.expand_as(img_bank))[0]
    img1 = img1.view(1, H, W, C).permute(0, 3, 1, 2)

    index2 = torch.randint(0, N, size=(H * W, 1))
    index2[index2==index1] = (index2[index2==index1] + 1) % N
    img2 = torch.gather(img_bank, 0, index=index2.expand_as(img_bank))[0]
    img2 = img2.view(1, H, W, C).permute(0, 3, 1, 2)

    img1 = img1.to(device)
    img2 = img2.to(device)

    if args.loss == 'L2':
        loss_fu = mse
    elif args.loss == 'L1':
        loss_fu = loss_l1

    loss = loss_func(img1, img2, loss_fu)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model, noisy_img, clean_img):
    model.eval()
    with torch.no_grad():
        pred = torch.clamp(model(noisy_img), 0, 1)
        MSE = mse(clean_img, pred).item()
        PSNR = 10 * np.log10(1 / MSE)
        clean_img = clean_img.squeeze().cpu()
        clean_img = clean_img.detach().numpy().transpose(1, 2, 0).astype(np.float32)
        pred1 = pred.squeeze().cpu()
        pred1 = pred1.detach().numpy().transpose(1, 2, 0).astype(np.float32)

    return PSNR,pred


def denoise(model, noisy_img):
    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img), 0, 1)

    return pred

avg_PSNR = 0
root = os.path.join(args.out_image, '_'.join(str(i) for i in [args.dataset,args.nt,args.nl, args.loss]))
os.makedirs(root, exist_ok=True)
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    clean_img = Image.open(image_path)
    clean_img1 = clean_img
    # Convert image to tensor and add an extra batch dimension
    clean_img = transform(clean_img).unsqueeze(0)

    sim_img_path = os.path.join(sim_image_folder,str(image_file).replace("png", "npy").replace("jpg", "npy").replace("PNG","npy").replace("JPG", "npy").replace("tif", "npy"))

    img_bank = np.load(sim_img_path).astype(np.float32).transpose((2, 0, 1, 3))  # NxHxWxC

    noisy_img = torch.from_numpy(img_bank[:1].transpose(0, 3, 1, 2))

    if args.nt=='gauss' and args.nl==10:
        args.banknum=2
    else:
        args.banknum = 10
    img_bank = img_bank[:args.banknum]

    N, H, W, C = img_bank.shape
    img_bank = torch.from_numpy(img_bank).view(img_bank.shape[0], -1, img_bank.shape[-1])

    clean_img = clean_img.to(device)
    noisy_img = noisy_img.to(device)
    model = UNet()
    checkpoint = './epoch_model_100.pth'
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[5], gamma=0.1)

    for epoch in range(max_epoch):
        train(model, optimizer, img_bank)
        scheduler.step()

    PSNR, out_img = test(model, noisy_img, clean_img)
    noisy_img = torch.clamp(noisy_img, 0, 1)
    noisy_img = to_pil_image(noisy_img.squeeze(0))
    noisy_img.save(os.path.join(root, os.path.splitext(image_file)[0] + '_noisy.png'))
    out_img = to_pil_image(out_img.squeeze(0))
    out_img.save(os.path.join(root, os.path.splitext(image_file)[0] + '.png'))
    print(f"PSNR for {image_file}: {PSNR}")
    avg_PSNR += PSNR
avg_PSNR = avg_PSNR / len(image_files)
print(f"PSNR for avg_PSNR: {avg_PSNR}")

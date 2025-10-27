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
parser.add_argument('--data_path', default='../data', type=str)
parser.add_argument('--dataset', default='polyu40', type=str)
parser.add_argument('--Noisy', default='Noisy', type=str)
parser.add_argument('--GT', default='GT', type=str)
parser.add_argument('--save', default='./results', type=str)
parser.add_argument('--out_image', default='./results_image_pre', type=str)
parser.add_argument('--ws', default=34, type=int)
parser.add_argument('--ps', default=7, type=int)
parser.add_argument('--nn', default=16, type=int)
parser.add_argument('--nr', default=20, type=int)
parser.add_argument('--banknum', default=10, type=int)
parser.add_argument('--loss', default='L2', type=str)
args = parser.parse_args()


image_folder = os.path.join(args.data_path, args.dataset, args.GT)
image_files = os.listdir(image_folder)
image_files = sorted(image_files)


sim_image_folder = os.path.join(args.save, '_'.join(str(i) for i in [args.dataset,  args.ws, args.ps, args.nn,args.nr, args.loss]))
sim_image_files = [f for f in os.listdir(sim_image_folder) if f.endswith(('.png', '.jpg', '.jpeg','.PNG', '.npy'))]
sim_image_files = sorted(sim_image_files)

transform = transforms.Compose([transforms.ToTensor()])

max_epoch = 100
lr = 0.00001


def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt, pred)

loss_l1 = nn.L1Loss()

def loss_func(img1, img2, loss_f=mse):
    pred1 = model(img1)
    loss = loss_f(img2, pred1)
    return loss

def calculate_psnr(img1, img2, border=0):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

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
        restored = pred
        restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        clean_img = torch.clamp(clean_img, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        PSNR = calculate_psnr(clean_img * 255.0, restored * 255.0)

    return PSNR,pred

def denoise(model, noisy_img):
    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img), 0, 1)

    return pred

avg_PSNR = 0
root = os.path.join(args.out_image, '_'.join(str(i) for i in [args.dataset, args.loss]))
os.makedirs(root, exist_ok=True)
for image_file in image_files:
    start_time = time.time()
    image_path = os.path.join(image_folder, image_file)
    clean_img = Image.open(image_path)
    clean_img = transform(clean_img).unsqueeze(0)
    sim_img_path = os.path.join(sim_image_folder,
                                str(image_file).replace("png", "npy").replace("jpg", "npy").replace("PNG","npy").replace("JPG", "npy").replace("tif", "npy"))

    img_bank = np.load(sim_img_path).astype(np.float32).transpose((2, 0, 1, 3))  # NxHxWxC

    noisy_img = torch.from_numpy(img_bank[:1].transpose(0, 3, 1, 2))
    img_bank = img_bank[0:args.banknum]

    N, H, W, C = img_bank.shape
    img_bank = torch.from_numpy(img_bank).view(img_bank.shape[0], -1, img_bank.shape[-1])

    clean_img = clean_img.to(device)
    noisy_img = noisy_img.to(device)
    model = UNet()
    checkpoint = './epoch_model_100.pth'
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device)
    print("The number of parameters of the network is: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = MultiStepLR(optimizer, milestones=[50,75], gamma=0.1)

    for epoch in range(max_epoch):
        train(model, optimizer, img_bank)
        scheduler.step()
    PSNR, out_img = test(model, noisy_img, clean_img)

    out_img = to_pil_image(out_img.squeeze(0))
    out_img.save(os.path.join(root, os.path.splitext(image_file)[0] + '.png'))
    print(f"PSNR for {image_file}: {PSNR}")
    end_time = time.time()
    total_time = end_time - start_time
    print("Total execution time: {:.2f} seconds".format(total_time))
    avg_PSNR += PSNR
avg_PSNR = avg_PSNR / len(image_files)
print(f"PSNR for avg_PSNR: {avg_PSNR}")

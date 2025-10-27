import glob
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing.pool import Pool
from tqdm import tqdm
from PIL import Image
from sklearn.feature_extraction.image import extract_patches_2d
import os
import torch
import argparse
import torchvision.transforms as transforms
import time
import torch.nn.functional as F
import torch.nn as nn
import einops

parser = argparse.ArgumentParser('Pixel2Pixel')

parser.add_argument('--data_path', default='../data', type=str)
parser.add_argument('--dataset', default='polyu40', type=str)
parser.add_argument('--GT', default='GT', type=str)
parser.add_argument('--Noisy', default='Noisy', type=str)
parser.add_argument('--save', default='./results', type=str)
parser.add_argument('--ws', default=34, type=int)
parser.add_argument('--ps', default=7, type=int)
parser.add_argument('--nn', default=16, type=int)
parser.add_argument('--nr', default=20, type=int)
parser.add_argument('--loss', default='L2', type=str)

args = parser.parse_args()

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

WINDOW_SIZE = args.ws
PATCH_SIZE = args.ps
NUM_NEIGHBORS = args.nn
NUM_ROWS = args.nr
loss_type = args.loss

transform = transforms.Compose([transforms.ToTensor()])

def search_patches(img, x, y):
    w, h = img.shape[0], img.shape[1]
    center_patch = img[x - (PATCH_SIZE - 1) // 2: x + (PATCH_SIZE + 1) // 2,
                   y - (PATCH_SIZE - 1) // 2: y + (PATCH_SIZE + 1) // 2]
    lx = max(0, x - WINDOW_SIZE // 2)
    rx = min(w - PATCH_SIZE + 1, x + WINDOW_SIZE // 2)
    ly = max(0, y - WINDOW_SIZE // 2)
    ry = min(h - PATCH_SIZE + 1, y + WINDOW_SIZE // 2)
    window = img[lx: rx, ly: ry]
    patches = extract_patches_2d(window, (PATCH_SIZE, PATCH_SIZE))
    if loss_type == 'L2':
        sorted_patches = sorted(patches, key=lambda x: np.sum((x - center_patch) ** 2))
    elif loss_type == 'L1':
        sorted_patches = sorted(patches, key=lambda x: np.sum(np.abs(x - center_patch)))
    return np.array(sorted_patches[: NUM_NEIGHBORS])


def search_pixel(img, x, y):
    patches = search_patches(img, x, y)
    mid = patches.shape[1] // 2
    return patches[:, mid, mid]


def pixel_shuffle(img, h, i):
    x = np.zeros((h, NUM_NEIGHBORS, 3))
    for j in range(h):
        x[j] = search_pixel(img, i + WINDOW_SIZE // 2, j + WINDOW_SIZE // 2)
    return x


if __name__ == "__main__":
    root = os.path.join(args.save, '_'.join(str(i) for i in [args.dataset, args.ws, args.ps, args.nn, args.nr, args.loss]))
    os.makedirs(root, exist_ok=True)
    image_folder = os.path.join(args.data_path, args.dataset, args.Noisy)
    image_files = [f for f in os.listdir(image_folder)]
    image_files = sorted(image_files)

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        start_time = time.time()

        img = Image.open(image_path)
        img = transform(img)
        img = img.half().cuda()
        img = img.cuda()
        img = img[None, ...]

        pad_sz = PATCH_SIZE // 2
        img_pad = F.pad(img, (pad_sz, pad_sz, pad_sz, pad_sz), mode='reflect')
        img_unfold = F.unfold(img_pad, kernel_size=PATCH_SIZE, padding=0, stride=1, )
        img_unfold = einops.rearrange(img_unfold, 'b c (h w) -> b c h w', h=img.shape[-2], w=img.shape[-1])
        img_unfold = F.pad(img_unfold, (WINDOW_SIZE // 2, WINDOW_SIZE // 2, WINDOW_SIZE // 2, WINDOW_SIZE // 2), mode='replicate')
        blk_sz = 64
        num_blk_w = img.shape[-1] // blk_sz
        num_blk_h = img.shape[-2] // blk_sz
        is_window_size_even = WINDOW_SIZE % 2 == 0
        topk_list = []
        topk_patches_list = []
        for blk_i in range(num_blk_w):
            for blk_j in range(num_blk_h):
                torch.cuda.empty_cache()
                sub_img_uf = img_unfold[..., blk_j * blk_sz: (WINDOW_SIZE // 2) * 2 + (blk_j + 1) * blk_sz,
                             blk_i * blk_sz: (WINDOW_SIZE // 2) * 2 + (blk_i + 1) * blk_sz]
                sub_img_shape = sub_img_uf.shape
                if is_window_size_even:
                    sub_img_uf_inp = sub_img_uf[..., :-1, :-1]
                else:
                    sub_img_uf_inp = sub_img_uf
                img_uf_uf = F.unfold(sub_img_uf_inp, kernel_size=WINDOW_SIZE, padding=0, stride=1, )
                img_uf_uf_reshape = einops.rearrange(img_uf_uf, 'b (c k1 k2 k3 k4) (h w) -> b (c k1 k2) (k3 k4) h w',
                                                     k1=PATCH_SIZE, k2=PATCH_SIZE, k3=WINDOW_SIZE, k4=WINDOW_SIZE,
                                                     h=blk_sz, w=blk_sz)
                img_uf_reshape = einops.rearrange(sub_img_uf, 'b (c k1 k2) h w -> b (c k1 k2) 1 h w', k1=PATCH_SIZE,
                                                  k2=PATCH_SIZE, h=sub_img_shape[-2], w=sub_img_shape[-1])
                img_uf_reshape = img_uf_reshape[..., WINDOW_SIZE // 2: WINDOW_SIZE // 2 + blk_sz,
                                 WINDOW_SIZE // 2: WINDOW_SIZE // 2 + blk_sz]
                l2_dis = torch.sum((img_uf_reshape - img_uf_uf_reshape) ** 2, dim=1)
                sort_indices = torch.argsort(l2_dis, dim=-3)[..., :NUM_NEIGHBORS, :, :]
                topk = torch.gather(img_uf_uf_reshape, dim=-3,
                                    index=sort_indices.unsqueeze(1).repeat(1, img_uf_uf_reshape.shape[1], 1, 1, 1))
                topk_list.append(topk)
        topk = torch.cat(topk_list, dim=0)
        topk = einops.rearrange(topk, '(w1 w2) (c k1k2) k h w -> k c k1k2 (w2 h) (w1 w)', w1=num_blk_w, w2=num_blk_h, c=3)
        topk_center = topk[:, :, topk.shape[2]//2:topk.shape[2]//2+1]
        l2_dis = torch.sum((topk - topk_center) ** 2, dim=[0,1])
        sort_indices = torch.argsort(l2_dis, dim=0)[:NUM_ROWS, :, :]
        ttopk = torch.gather(topk, dim=2, index=sort_indices[None, None, ...].repeat(topk.shape[0], topk.shape[1], 1, 1, 1))
        ttopk = ttopk[0]
        ttopk = einops.rearrange(ttopk, 'c nr h w -> h w nr c')
        end_time = time.time()
        total_time = end_time - start_time
        print("Total execution time: {:.2f} seconds".format(total_time))
        ttopk = ttopk.cpu()
        file_name_without_extension = os.path.splitext(image_file)[0]
        np.save(os.path.join(root, file_name_without_extension), ttopk)
    print('All subprocesses done.')

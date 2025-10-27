# TTAD
Official PyTorch implementation of "Robust Test-Time Adaptation for Single Image Denoising Using Deep Gaussian Prior" in ICCV 2025

## Abstract
Gaussian denoising often serves as the initiation of research in the field of image denoising, owing to its prevalence and intriguing properties. However, deep Gaussian denoiser typically generalizes poorly to other types of noises, such as Poisson noise and real-world noise. In this paper, we reveal that deep Gaussian denoisers have an underlying ability to handle other noises with only ten iterations of self-supervised learning, which is referred to as deep denoiser prior. Specifically, we first pre-train a Gaussian denoising model in a self-supervised manner. Then, for each test image, we construct a pixel bank based on the self-similarity and randomly sample pseudo-instance examples from it to perform test-time adaptation. Finally, we fine-tune the pre-trained Gaussian denoiser using the randomly sampled pseudo-instances. Extensive experiments demonstrate that our test-time adaptation method helps the pre-trained Gaussian denoiser rapidly improve performance in removing both in-distribution and out-of-distribution noise, achieving superior performance compared to existing single-image denoising methods while also significantly reducing computational time.

---

## Setup

### Requirements

Our experiments are done with:

- python = 3.8.13
- pytorch = 1.11.0+cu113
- numpy = 1.21.5
- scikit-image = 0.19.2

## Pre-training
We use the ["Neighbor2Neighbor"](https://github.com/TaoHuang2018/Neighbor2Neighbor) model as the pre-trained Gaussian denoiser. The model is trained on Gaussian noise with $σ = 25$.


## Building Noise2Noise pairs

* For synthetic noise, run ``build_pixel_bank_syn.py`` 

* For real-world images, run ``build_pixel_bank_real.py``

## Test Time Adaptation

* For synthetic noise, run ``TTAD_syn.py`` 

* For real-world images, run ``build_pixel_bankTTAD_real.py``

## Acknowledgment
This repository is based on the official implementation of ["Neighbor2Neighbor"](https://github.com/TaoHuang2018/Neighbor2Neighbor). We thank the authors for releasing the code.

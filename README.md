# Exploring Multilingual Text Dataset Distillation

The base version of this repo is a clone of [Soft-Label Dataset Distillation and Text Dataset Distillation](https://github.com/ilia10000/dataset-distillation). The experiments in our project can be reproduced by using the commands given in `run.sh`.

###Methods Implemented:
1. VanillaDistill
2. SkipLookupDistill
3. VocabDistill (Softmax)
4. VocabDistill (Gumbel)

## Prerequisites

### System requirements
- Python 3
- CPU or NVIDIA GPU + CUDA

### Dependencies
- ``faiss==1.7.3``
- ``matplotlib==3.7.1``
- ``numpy==1.24.3``
- ``pandas==2.0.2``
- ``Pillow==9.5.0``
- ``PyYAML==5.4.1``
- ``scikit_learn==1.2.2``
- ``six==1.16.0``
- ``skimage==0.0``
- ``torch==1.13.1``
- ``torchtext==0.6.0``
- ``torchvision==0.14.1``
- ``tqdm==4.65.0``
- ``transformers==4.29.2``

## Using this repo
The file `docs/advanced.md` by the original authours gives a detailed description of useful parameters.

References:
1. Soft-Label Dataset Distillation and Text Dataset Distillation [Paper](https://arxiv.org/abs/1910.02551)
2. Dataset Distillation [Dataset Distillation](https://ssnl.github.io/dataset_distillation):
The code in the original repo is written by [Tongzhou Wang](https://ssnl.github.io/),  [Jun-Yan Zhu](https://github.com/junyanz) and [Ilia Sucholutsky](https://ilia10000.github.io/).

## Prerequisites

### System requirements
- Python 3
- CPU or NVIDIA GPU + CUDA

### Dependencies
- ``torch >= 1.0.0``
- ``torchvision >= 0.2.1``
- ``numpy``
- ``matplotlib``
- ``pyyaml``
- ``tqdm``
- ``torchtext``

You may install `PyTorch` (`torch` package above) using any suggested method for your environment [here](https://pytorch.org/get-started/locally/).

## Using this repo

This fork provides the implementation of the two distillation algorithms described in the paper. Below we describe the basic distillation setting. For other settings and usages, please check out the [Advanced Usage](docs/advanced.md) as well as the [useful scripts](docs/scripts.txt). 



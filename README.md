# 😈 READ ME!
<!-- *This file is last updated by Ziyu on June, 2022.* -->

[![Python 3.8](https://img.shields.io/badge/python-3.8-blueviolet.svg)](https://www.python.org/downloads/release/python-380/) [![Pytorch](https://img.shields.io/badge/Pytorch-1.12.1-critical.svg)](https://github.com/pytorch/pytorch/releases/tag/v1.12.0) [![License](https://img.shields.io/badge/License-Apache%202.0-ff69b4.svg)](https://opensource.org/licenses/Apache-2.0) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-success.svg)](https://github.com/ZIYU-DEEP/Generalization-and-Memorization-in-Sparse-Training)
## 1. Intro
This repository is organized as follows:
```bash
sparse-training-via-information
|== loader
│   └── loader_cifar100.py
│   └── loader_cifar100_noisy.py
│   └── main.py
|== network
│   └── mlp.py
│   └── resnet.py
│   └── main.py
|== optim
│   └── trainer.py
│   └── model.py
|== helper
│   └── pruner.py
│   └── utils.py
|== scripts
│   └── cifar100_resnet.sh
|== run.py
|== experiment.py
```


## 2. Requirements
### Working with CPU/GPU
If you are using anaconda:
```bash
conda create --name sparse python=3.8
conda activate sparse
```

To install necessary pakacges, check the list in `./requirements.txt` or lazily run the following in the designated environment for the project:
```bash
python3 -m pip install -r requirements.txt
```



## 3. Example Commands
```shell
# Training the network
. scripts/train.sh

# Get the Hessian spectrum
. scripts/spectrum.sh
```
Be sure check the original file for more detailed instructions on datasets, network, and optimization options, and modify the bash scripts accordingly. I have written help strings for every argument.

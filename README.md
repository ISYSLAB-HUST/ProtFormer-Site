# ProtFormer-Site

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

ProtFormer-Site......

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/shuxiang111/ProtFormer-Site.git
cd ProtFormer-Site
```

### !!!Noted: 

### Please change prefix path into your own path in the last line of environment.yaml file.

### Please match your python version ,cuda version and torch version with the package.

### 2. Create and Activate Environment
```bash
conda env create -f environment.yml
conda activate ProtFormerSite
```

## Pre-trained models and parameters
The pre trained model and parameters are placed in the [weight folder](https://github.com/shuxiang111/ProtFormer-Site/weight). Please read the README.md file in the weight folder for further prediction

## Usage
```bash
python predict.py
```

## Issues
If you encounter any problems, please open an issue.

## License
...

### Code License
...

### Parameter Files License
...

## Acknowledgements
ProtFormer-Site with and/or references the following separate libraries and packages:
- [PyTorch](https://github.com/pytorch/pytorch)
- [biopython](https://github.com/biopython/biopython)
- [esm](https://github.com/facebookresearch/esm)
- [minLoRA](https://github.com/cccntu/minLoRA)

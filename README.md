# ProtFormer-Site

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Introduction

We propose a novel PPI site prediction framework, ProtFormer-Site, which utilizes large protein language model, an efficient parameter fine-tuning strategy, and the ProtFormer backbone. ProtFormer-Site demonstrated outstanding performance across all evaluation metrics on three benchmark datasets, with Matthews correlation coefficient (MCC) improvements ranging from 22.4% to 61.5% across different datasets. These results demonstrate that ProtFormer offers significant advantages in PPI site prediction, providing a more accurate and efficient solution.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ISYSLAB-HUST/ProtFormer-Site.git
cd ProtFormer-Site
```


### 2. Create and Activate Environment
```bash
conda env create -f environment.yml
conda activate ProtFormerSite
```

## Pre-trained models and Config parameters
The pre-trained models and parameters are placed in the [weight folder](https://github.com/ISYSLAB-HUST/ProtFormer-Site-final/tree/main/weight) and the [config folder](https://github.com/ISYSLAB-HUST/ProtFormer-Site-final/tree/main/config).

## Usage
We provide test script for users to evaluate the prediction result.
```bash
# example: run Single_DeepPPIS test
python predict.py --config ./config/Single_DeepPPISP.yaml
```

## Issues
If you encounter any problems, please open an issue.


## License
This project is licensed under the MIT License for the code and a custom license for the parameter files.

### Code License

The code in this project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

### Parameter Files License

The parameter files in this project are licensed under a custom license. Educational use is free of charge, while commercial use requires a commercial license. See the [PARAMETER_LICENSE](./PARAMETER_LICENSE) file for more details.

## Acknowledgements
ProtFormer-Site with and/or references the following separate libraries and packages:
- [PyTorch](https://github.com/pytorch/pytorch)
- [biopython](https://github.com/biopython/biopython)
- [esm](https://github.com/facebookresearch/esm)
- [minLoRA](https://github.com/cccntu/minLoRA)

## Citation
If you use this code or one of our pretrained models for your publication, please cite our paper:

```
@article{wang2024,
  title={Ultra-fast and Accurate Prediction of Protein-protein Interaction Sites Using Protein Language Models and ProtFormer}
}
```






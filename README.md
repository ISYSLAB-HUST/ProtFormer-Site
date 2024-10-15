# ProtFormer-Site

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

## Introduction

ProtFormer-Site is a novel protein-protein interaction site prediction tool that leverages the [ESM2](https://github.com/facebookresearch/esm) protein language model and our newly developed ProtFormer architecture. It surpasses most PPI prediction tools that rely on protein structure and co-evolution information, using only protein single sequence information.

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
The pre-trained model and parameters are placed in the [weight folder](https://github.com/ISYSLAB-HUST/ProtFormer-Site/tree/main/weight). Please read the [README.md](https://github.com/ISYSLAB-HUST/ProtFormer-Site/blob/main/config) file in the weight folder for further prediction.

## Usage
We provide test script for users to evaluate the prediction result.
```bash
# example: run DeepPPIS test
python predict.py --config ./config/DeepPPIS.yaml
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






# ProtFormer-Site
![License](https://img.shields.io/github/license/shuxiang111/ProtFormer-Site)
![Issues](https://img.shields.io/github/issues/shuxiang111/ProtFormer-Site)
![Stars](https://img.shields.io/github/stars/shuxiang111/ProtFormer-Site)

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

ProtSSSD is a powerful model that leverages a large protein language model (ESM) to predict protein structural properties, including disorder. This model outperforms other single-sequence methods, providing more accurate and reliable predictions for various protein characteristics.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ISYSLAB-HUST/ProtSSSD.git
cd ProtSSSD
```
### 2. Create and Activate Environment
```bash
conda env create -f environment.yml
conda activate protsssd
```
## Usage
### CAID-2 Disorder
```bash
python predict_disorder.py ./dataset/caid2/ ./weight/ProtSSSD.pth --output result_disorder.json
```

### CASP 15
```bash
python predict_ss.py ./dataset/casp15/ ./weight/ProtSSSD.pth
```

## Issues
If you encounter any problems, please open an issue.

## Pull Requests
1. Fork the repository
2. Create a new branch (git checkout -b feature/yourfeature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin feature/yourfeature)
5. Create a new Pull Request
## License
This project is licensed under the MIT License for the code and a custom license for the parameter files.

### Code License

The code in this project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

### Parameter Files License

The parameter files in this project are licensed under a custom license. Educational use is free of charge, while commercial use requires a commercial license. See the [PARAMETER_LICENSE](./PARAMETER_LICENSE) file for more details.

## Acknowledgements
ProtSSSD with and/or references the following separate libraries and packages:
- [PyTorch](https://github.com/pytorch/pytorch)
- [biopython](https://github.com/biopython/biopython)
- [einops](https://github.com/arogozhnikov/einops)
- [torchmetrics](https://github.com/Lightning-AI/torchmetrics)
- [esm](https://github.com/facebookresearch/esm)
- [minLoRA](https://github.com/cccntu/minLoRA)

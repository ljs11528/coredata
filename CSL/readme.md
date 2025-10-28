# CSL

## Getting Started

### Installation

```bash
cd CSL
conda create -n csl python=3.11.9
conda activate csl
pip install -r requirements.txt
pip install torch==2.4.0 torchvision==0.19.0 -f https://download.pytorch.org/whl/torch_stable.html

```

### Dataset

- Pascal: (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)
- Cityscapes: (https://www.cityscapes-dataset.com/)

Please modify your dataset path in configuration files.

```
├── [Your Pascal Path]
    ├── JPEGImages
    └── SegmentationClass
    
├── [Your Cityscapes Path]
    ├── leftImg8bit
    └── gtFine
```

## Usage

### csl

```bash
cd CSL
scripts/csl.sh
```

### Supervised Baseline

```bash
cd CSL
scripts/supervised.sh
```
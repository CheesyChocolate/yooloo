# yooloo

[![PyPI - Version](https://img.shields.io/pypi/v/yooloo.svg)](https://pypi.org/project/yooloo)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/yooloo.svg)](https://pypi.org/project/yooloo)

-----

## Table of Contents

- [Installation](#installation)
- [License](#license)

## Installation

> [!WARNING]
> Not released yet. This package is still in development and not available on PyPI.

```console
pip install yooloo
```

## Note about example dataset in the repository

The example dataset in the repository and releases is taken from
[Kaggle](https://www.kaggle.com/datasets/rajkumarl/people-clothing-segmentation).
This release file is provided as an example of how the directory structure
should look like.

## Dir structure of dataset formats

### Labelme format

```text
labelme_dataset/
├── img1.json
├── img1.jpg
├── img2.json
├── img2.png
├── subdirectory/
│   ├── img3.json
│   └── img3.jpg
└── ...
```

### COCO format

```text
coco_dataset/
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   ├── person_keypoints_train.json
│   ├── person_keypoints_val.json
│   ├── captions_train.json
│   └── captions_val.json
├── train/
│   ├── 000000000001.jpg
│   ├── 000000000002.jpg
│   └── ...
└── val/
    ├── 000000000501.jpg
    ├── 000000000502.jpg
    └── ...
```

### YOLO format

```text
yolov8_dataset/
├── data.yaml
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── ...
│   └── val/
│       ├── img101.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img001.txt
    │   └── ...
    └── val/
        ├── img101.txt
        └── ...
```

### Pascal VOC format

```text
VOCdevkit/
└── VOC/
    ├── Annotations/
    │   ├── 000001.xml
    │   ├── 000002.xml
    │   └── ...
    ├── ImageSets/
    │   ├── Action/
    │   │   ├── train.txt
    │   │   ├── trainval.txt
    │   │   ├── val.txt
    │   │   └── Action-specific files (jump_train.txt, etc)
    │   ├── Layout/
    │   │   ├── train.txt
    │   │   ├── trainval.txt
    │   │   └─── val.txt
    │   ├── Main/
    │   │   ├── train.txt
    │   │   ├── trainval.txt
    │   │   ├── val.txt
    │   │   └── class-specific files (cat_train.txt, etc)
    │   └── Segmentation/
    │       ├── train.txt
    │       ├── trainval.txt
    │       └── val.txt
    ├── JPEGImages/
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ...
    ├── SegmentationClass/
    │   ├── 000001.png
    │   ├── 000002.png
    │   └── ...
    └── SegmentationObject/
        ├── 000001.png
        ├── 000002.png
        └── ...
```

## License

`yooloo` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

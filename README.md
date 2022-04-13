# openpifpaf_furniture
A plugin of OpenPifPaf for furniture pose detection and classification

#### Abstract

> Real-time Multi-Object Furniture Pose Detection and Classification
>
> We present a multi-object pose detection and classification method of home furniture in cluttered and occluded indoor environments.
> We generalize OpenPifPaf, a field-based method that jointly detects and forms spatio-temporal keypoint associations of a specific object, with the capacity of jointly performing detection and classification of multiple objects in a bottom-up, box-free and real-time manner. We demonstrate that our proposed method outperforms state-of-the-art furniture key-
point detection methods on two publicly available datasets (Keypoint-5 and Pascal3D+).
> We further implement a synthetic dataset to evaluate the performance when target objects have occluded viewpoints or limited resolutions. Results also show that our synthetic dataset boosts the performance of detecting real-world instances. All source codes are shared.

![Example](docs/example.png)

### Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Interfaces](#interfaces)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project structure](#project-structure)

## Installation

We encourage to setup a virtual environment in your work space.
```
# Create a virtual environment in work_space.
mkdir work_space
cd ws_space
virtualenv -p /usr/bin/python3.6 venv
source venv/bin/activate

```

Clone this repository.
```
# To clone the repository using HTTPS
git clone https://github.com/Mikasatlx/openpifpaf_furniture.git
cd openpifpaf_furniture
```

All dependencies can be found in the `requirements.txt` file.
```
# To install dependencies
pip3 install -r requirements.txt
```

Build the cpp extension.
```
# To compile the cpp extension
pip3 install -e .
```

This project has been tested with Python 3.7.7, PyTorch 1.9.1, CUDA 10.2 and OpenPifPaf 0.13.0.


## Dataset

This project uses dataset [Keypoint5](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/) and [Pascal3D+](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/) for training and evaluation. 

Please refer to JAAD documentation to download the dataset.


## Interfaces

This project is implemented as an [OpenPifPaf](https://github.com/openpifpaf/openpifpaf) plugin module.
As such, it benefits from all the core capabilities offered by OpenPifPaf, and only implements the additional functions it needs.

All the commands can be run through OpenPifPaf's interface using subparsers.
Help can be obtained for any of them with option `--help`.
More information can be found in [OpenPifPaf documentation](https://openpifpaf.github.io/intro.html).


## Training

Training is done using subparser `openpifpaf.train`.

Training on JAAD with all attributes can be run with the command:
```
python3 -m openpifpaf.train \
  --output <path/to/model.pt> \
  --dataset jaad \
  --jaad-root-dir <path/to/jaad/folder/> \
  --jaad-subset default \
  --jaad-training-set train \
  --jaad-validation-set val \
  --log-interval 10 \
  --val-interval 1 \
  --epochs 5 \
  --batch-size 4 \
  --lr 0.0005 \
  --lr-warm-up-start-epoch -1 \
  --weight-decay 5e-4 \
  --momentum 0.95 \
  --basenet fn-resnet50 \
  --pifpaf-pretraining \
  --detection-bias-prior 0.01 \
  --jaad-head-upsample 2 \
  --jaad-pedestrian-attributes all \
  --fork-normalization-operation power \
  --fork-normalization-duplicates 35 \
  --lambdas 7.0 7.0 7.0 7.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 \
  --attribute-regression-loss l1 \
  --attribute-focal-gamma 2 \
  --auto-tune-mtl
```
Arguments should be modified appropriately if needed.

More information about the options can be obtained with the command:
```
python3 -m openpifpaf.train --help
```


## Evaluation

Evaluation of a checkpoint is done using subparser `openpifpaf.eval`.

Evaluation on JAAD with all attributes can be run with the command:
```
python3 -m openpifpaf.eval \
  --output <path/to/outputs> \
  --dataset jaad \
  --jaad-root-dir <path/to/jaad/folder/> \
  --jaad-subset default \
  --jaad-testing-set test \
  --checkpoint <path/to/checkpoint.pt> \
  --batch-size 1 \
  --jaad-head-upsample 2 \
  --jaad-pedestrian-attributes all \
  --head-consolidation filter_and_extend \
  --decoder instancedecoder:0 \
  --decoder-s-threshold 0.2 \
  --decoder-optics-min-cluster-size 10 \
  --decoder-optics-epsilon 5.0 \
  --decoder-optics-cluster-threshold 0.5
```
Arguments should be modified appropriately if needed.

Using option `--write-predictions`, a json file with predictions can be written as an additional output.

Using option `--show-final-image`, images with predictions displayed on them can be written in the folder given by option `--save-all <path/to/image/folder/>`.
To also display ground truth annotations, add option `--show-final-ground-truth`.

More information about the options can be obtained with the command:
```
python3 -m openpifpaf.eval --help
```


## Project structure

The code is organized as follows:
```
openpifpaf_detection_attributes/
├── datasets/
│   ├── jaad/
│   ├── (+ common files for datasets)
│   └── (add new datasets here)
└── models/
    ├── mtlfields/
    ├── (+ common files for models)
    └── (add new models here)
```


## License

This project is built upon [OpenPifPaf](https://openpifpaf.github.io/intro.html) and shares the AGPL Licence.

This software is also available for commercial licensing via the EPFL Technology Transfer
Office (https://tto.epfl.ch/, info.tto@epfl.ch).


## Citation

If you use this project in your research, please cite the corresponding paper:
```text
@article{mordan2021detecting,
  title={Detecting 32 Pedestrian Attributes for Autonomous Vehicles},
  author={Mordan, Taylor and Cord, Matthieu and P{\'e}rez, Patrick and Alahi, Alexandre},
  journal={IEEE Transactions on Intelligent Transportation Systems (T-ITS)},
  year={2021},
  doi={10.1109/TITS.2021.3107587}
}
```


## Acknowledgements

We would like to thank Valeo for funding our work, and Sven Kreiss for the OpenPifPaf Plugin architecture.

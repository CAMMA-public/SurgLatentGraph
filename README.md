<div align="center">
<a href="http://camma.u-strasbg.fr/">
<img src="logo.png" width="30%">
</a>
</div>

# **Latent Graph Representations for Surgical Scene Understanding**
This repository contains the code corresponding to our Transactions on Medical Imaging paper _Latent Graph Representations for Critical View of Safety Assessment_ and our MICCAI 2023 paper _Encoding Surgical Videos as Latent Spatiotemporal Graphs for Object- and Anatomy-Driven Reasoning_.

[1] **Latent Graph Representations for Critical View of Safety Assessment**. _Aditya Murali, Deepak Alapatt, Pietro Mascagni, Armine Vardazaryan, Alain Garcia, Nariaki Okamoto, Didier Mutter, Nicolas Padoy. **IEEE Transactions on Medical Imaging 2023**_


[![arXiv](https://img.shields.io/badge/arXiv%20-%202212.04155%20-%20red)](https://arxiv.org/abs/2212.04155)
[![Paper](https://img.shields.io/badge/Paper%20-%20darkblue)](https://ieeexplore.ieee.org/document/10319763)

[2] **Encoding Surgical Videos as Latent Spatiotemporal Graphs for Object- and Anatomy-Driven Reasoning**. _Aditya Murali, Deepak Alapatt, Pietro Mascagni, Armine Vardazaryan, Alain Garcia, Nariaki Okamoto, Didier Mutter, Nicolas Padoy. **MICCAI 2023**_

[![arXiv](https://img.shields.io/badge/arXiv%20-%202212.04155%20-%20red)](https://arxiv.org/abs/2212.04155)
[![Paper](https://img.shields.io/badge/Paper%20-%20darkblue)](https://link.springer.com/chapter/10.1007/978-3-031-43996-4_62)

## News

#### In this repo we provide:
- Implementations of 3 different object detectors (Faster-RCNN, Cascade-RCNN, Deformable-DETR) and 3 different instance segmentation models (Mask-RCNN, Cascade-Mask-RCNN, Mask2Former) using the **_mmdetection_** framework.
- Implementations of 4 different object-centric models for CVS prediction introduced in [1]: LatentGraph-CVS (LG-CVS), DeepCVS, LayoutCVS, and ResNet50-DetInit, each of which can be run using any of the aforementioned object detection/segmentation models.
- Implementation of a simple classifier using **_mmengine + mmdetection_**.
- Implementations of 2 different spatiotemporal object-centric models introduced in [2]: **S**urgical **V**ideos as **L**atent **S**patio**T**emporal **G**raphs (SV2LSTG), and DeepCVS-Temporal (DC-Temp).
- Config files and instructions to train/evaluate object detectors/segmentation models on Endoscapes [3] and CholecSeg8k [4].
- Config files and instructions to train/evaluate the 5 single frame and 2 spatiotemporal methods on three tasks/datasets: CVS Prediction (Endoscapes), Phase Recognition (Cholec80), and Action Triplet Recognition (CholecT50)
- Trained model checkpoints for all tasks (coming soon).

# Get Started

## Installation
This project uses Pytorch 2.1.0 + CUDA 11.8, DGL 1.1.1, torch-scatter, mmdetection 3.2.0, and mmengine 0.7.4. Please note that you may encounter issues if you diverge from these versions. If you must diverge, please ensure that the DGL and torch-scatter versions match your versions of pytorch, and make sure to use mmengine<=0.7.4.

```sh
> cd $LATENTGRAPH
> conda create -n latentgraph python=3.8 && conda activate latentgraph
# install dependencies 
(latentgraph) > conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
(latentgraph) > conda install -c dglteam/label/cu113 dgl
(latentgraph) > pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
(latentgraph) > pip install -U openmim
(latentgraph) > mim install mmdet
(latentgraph) > mim install mmengine==0.7.4
(latentgraph) > pip install torchmetrics
(latentgraph) > pip install scikit-learn
(latentgraph) > pip install prettytable
(latentgraph) > pip install imagesize
(latentgraph) > pip install networkx
(latentgraph) > pip install opencv-python
(latentgraph) > pip install yapf==0.40.1
```

## Dataset Setup

Each dataset needs to be setup in the appropriate format. We adopt the COCO format and modify the annotation files to contain image-level annotations and video ids as tags. We retain the image folder structure, and for all datasets, frames are extracted at 1 fps with the naming format `${VIDEO}_${FRAME}.jpg`. 

- The directory structure should look as follows.
```shell
data/mmdet_datasets
└── endoscapes/
    ├── train/
        └── 10947_14050.jpg
        ...
        └── 9762_40750.jpg
        └── annotation_coco.json
        └── annotation_ds_coco.json
        └── annotation_coco_vid.json
    ├── val/
        └── 11057_23575.jpg
        ...
        └── 9916_39400.jpg
        └── annotation_coco.json
        └── annotation_ds_coco.json
        └── annotation_coco_vid.json
    ├── test/
        └── 10983_1225.jpg
        ...
        └── 9823_55250.jpg
        └── annotation_coco.json
        └── annotation_ds_coco.json
        └── annotation_coco_vid.json
    ├── train_seg/
        └── 11088_10825.jpg
        ...
        └── 15736_60875.jpg
        └── annotation_coco.json
    ├── val_seg/
        └── 11088_10825.jpg
        ...
        └── 15736_60875.jpg
        └── annotation_coco.json
    ├── test_seg/
        └── 11104_22925.jpg
        ...
        └── 13513_34875.jpg
        └── annotation_coco.json
    ├── train_vids.txt
    ├── val_vids.txt
    ├── test_vids.txt
    ├── train_seg_vids.txt
    ├── val_seg_vids.txt
    ├── test_seg_vids.txt
└── cholec80/
    ├── train_phase/
        └── 1_0.jpg
        └── 1_25.jpg
        └── 1_50.jpg
        ...
        └── 30_0.jpg
        ...
        └── 40_55525.jpg
        └── annotation_coco.json
        └── annotation_ds_coco.json
        └── annotation_coco_vid.json
    ├── val_phase/
        └── 41_0.jpg
        ...
        └── 48_45825.jpg
        └── annotation_coco.json
        └── annotation_ds_coco.json
        └── annotation_coco_vid.json
    ├── test_phase/
        └── 49_0.jpg
        ...
        └── 80_43075.jpg
        └── annotation_coco.json
        └── annotation_ds_coco.json
        └── annotation_coco_vid.json
    ├── train_vids.txt
    ├── val_vids.txt
    ├── test_vids.txt
└── cholecT50/
    ├── train/
        └── 1_0.jpg
        ...
        └── 42_92775.jpg
        └── annotation_ds_coco.json
        └── annotation_coco_vid.json
    ├── val/
        └── 5_0.jpg
        ...
        └── 74_40825.jpg
        └── annotation_ds_coco.json
        └── annotation_coco_vid.json
    ├── test/
        └── 92_0.jpg
        ...
        └── 111_53625.jpg
        └── annotation_ds_coco.json
        └── annotation_coco_vid.json
```
### Dataset/Annotation Downloads
[![Endoscapes](https://img.shields.io/badge/Endoscapes%20-red)](https://github.com/CAMMA-public/Endoscapes)
[![CholecT50](https://img.shields.io/badge/CholecT50%20-green)](https://github.com/CAMMA-public/cholect50)
[![Cholec80](https://img.shields.io/badge/Cholec80%20-purple)](https://docs.google.com/forms/d/1GwZFM3-GhEduBs1d5QzbfFksKmS1OqXZAz8keYi-wKI)
[![COCO-Style Annotations](https://img.shields.io/badge/COCOs%20-teal)](https://github.com/CAMMA-public/Endoscapes)

## Config Structure

## Training and Testing

### Object Detectors

### Single-Frame Models

### Temporal Models

## Model Weights

Coming Soon!

## Citation

Please cite the appropriate papers if you make use of this repository.

```bibtex
@article{murali2023latent,
  author={Murali, Aditya and Alapatt, Deepak and Mascagni, Pietro and Vardazaryan, Armine and Garcia, Alain and Okamoto, Nariaki and Mutter, Didier and Padoy, Nicolas},
  journal={IEEE Transactions on Medical Imaging},
  title={Latent Graph Representations for Critical View of Safety Assessment}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMI.2023.3333034}
}

@inproceedings{murali2023encoding,
  title={Encoding Surgical Videos as Latent Spatiotemporal Graphs for Object and Anatomy-Driven Reasoning},
  author={Murali, Aditya and Alapatt, Deepak and Mascagni, Pietro and Vardazaryan, Armine and Garcia, Alain and Okamoto, Nariaki and Mutter, Didier and Padoy, Nicolas},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={647--657},
  year={2023},
  organization={Springer}
}
```

<div align="center">
<a href="http://camma.u-strasbg.fr/">
<img src="logo.png" width="30%">
</a>
</div>

# **Latent Graph Representations for Surgical Scene Understanding**
This repository contains the code corresponding to our Transactions on Medical Imaging paper _Latent Graph Representations for Critical View of Safety Assessment_ and our MICCAI 2023 paper _Encoding Surgical Videos as Latent Spatiotemporal Graphs for Object- and Anatomy-Driven Reasoning_.

<div align="center">
<img src="LG_overall.png" width="90%">
</div>

[1] **Latent Graph Representations for Critical View of Safety Assessment**. _Aditya Murali, Deepak Alapatt, Pietro Mascagni, Armine Vardazaryan, Alain Garcia, Nariaki Okamoto, Didier Mutter, Nicolas Padoy. **IEEE Transactions on Medical Imaging 2023**_


[![arXiv](https://img.shields.io/badge/arXiv%20-%202212.04155%20-%20red)](https://arxiv.org/abs/2212.04155)
[![Paper](https://img.shields.io/badge/Paper%20-%20darkblue)](https://ieeexplore.ieee.org/document/10319763)

<div align="center">
<img src="stgraph_arch.png" width="90%">
</div>

[2] **Encoding Surgical Videos as Latent Spatiotemporal Graphs for Object and Anatomy-Driven Reasoning**. _Aditya Murali, Deepak Alapatt, Pietro Mascagni, Armine Vardazaryan, Alain Garcia, Nariaki Okamoto, Didier Mutter, Nicolas Padoy. **MICCAI 2023**_

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
    └── train/
        └── 10947_14050.jpg
        ...
        └── 9762_40750.jpg
        └── annotation_coco.json
        └── annotation_ds_coco.json
        └── annotation_coco_vid.json
    └── val/
        └── 11057_23575.jpg
        ...
        └── 9916_39400.jpg
        └── annotation_coco.json
        └── annotation_ds_coco.json
        └── annotation_coco_vid.json
    └── test/
        └── 10983_1225.jpg
        ...
        └── 9823_55250.jpg
        └── annotation_coco.json
        └── annotation_ds_coco.json
        └── annotation_coco_vid.json
    └── train_seg/
        └── 11088_10825.jpg
        ...
        └── 15736_60875.jpg
        └── annotation_coco.json
    └── val_seg/
        └── 11088_10825.jpg
        ...
        └── 15736_60875.jpg
        └── annotation_coco.json
    └── test_seg/
        └── 11104_22925.jpg
        ...
        └── 13513_34875.jpg
        └── annotation_coco.json
    └── train_vids.txt
    └── val_vids.txt
    └── test_vids.txt
    └── train_seg_vids.txt
    └── val_seg_vids.txt
    └── test_seg_vids.txt
└── cholec80/
    └── train_phase/
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
    └── val_phase/
        └── 41_0.jpg
        ...
        └── 48_45825.jpg
        └── annotation_coco.json
        └── annotation_ds_coco.json
        └── annotation_coco_vid.json
    └── test_phase/
        └── 49_0.jpg
        ...
        └── 80_43075.jpg
        └── annotation_coco.json
        └── annotation_ds_coco.json
        └── annotation_coco_vid.json
    └── train_vids.txt
    └── val_vids.txt
    └── test_vids.txt
└── cholecT50/
    └── train/
        └── 1_0.jpg
        ...
        └── 42_92775.jpg
        └── annotation_ds_coco.json
        └── annotation_coco_vid.json
    └── val/
        └── 5_0.jpg
        ...
        └── 74_40825.jpg
        └── annotation_ds_coco.json
        └── annotation_coco_vid.json
    └── test/
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
Each `dataset | detector | downstream_method` combination has its own configuration. We summarize the config structure below.
```shell
configs/
└── models/
    └── endoscapes/
        └── lg_base_box.py
        └── lg_base_seg.py
        └── lg_ds_base.py
        └── lg_save_base.py
        └── deepcvs_base.py
        └── simple_classifier.py
        └── simple_classifier_with_recon.py
        └── ssl
            └── simple_classifier_${INIT}.py
            ... # ResNet50 with different backbone initializations
    └── c80_phase/
        └── lg_base_box.py
        ... # same files as endoscapes
    └── cholecT50/
        └── lg_base_box.py
        ... # same files as endoscapes
    └── ${DETECTOR}/ # e.g. faster_rcnn
        └── lg_${DETECTOR}.py
        └── lg_ds_${DETECTOR}.py
        └── lg_ds_${DETECTOR}_no_recon.py
        └── lg_save_${DETECTOR}.py
        └── lg_ft_save_${DETECTOR}.py
        └── layout_${DETECTOR}.py
        └── layout_${DETECTOR}_no_recon.py
        └── dc_${DETECTOR}.py
        └── dc_${DETECTOR}_no_recon.py
        ... # some ablations
    ... # one folder for each detector
    └── select_dataset.sh
└── temporal_models/
    └── endoscapes/
        └── sv2lstg_model_base.py
        └── sv2lstg_5_base.py # sv2lstg
        └── sv2lstg_10_base.py
        └── sv2lstg_15_base.py
        └── sv2lstg_load_graphs_5_base.py # sv2lstg, skip image -> graph encoding and load saved graph
        └── sv2lstg_load_graphs_10_base.py
        └── sv2lstg_load_graphs_15_base.py
        └── dc_temp_model_base.py
        └── dc_temp_5_base.py # deepcvs-temporal
        └── dc_temp_10_base.py
        └── dc_temp_15_base.py
    └── cholecT50/
        └── ... # same as endoscapes
    └── c80_phase/
        └── ... # same as endoscapes
        └── sv2lstg_load_graphs_all.py # load all graphs in video -> temporal decoding to predict phase
    └── ${DETECTOR}/ # e.g. faster_rcnn
        └── sv2lstg_${DETECTOR}_5.py # use ${DETECTOR} to construct each latent graph, clips of 5 frames
        └── sv2lstg_${DETECTOR}_10.py
        └── sv2lstg_${DETECTOR}_15.py
        └── sv2lstg_lin_probe_${DETECTOR}_5.py # load latent graphs constructed with ${DETECTOR}, linear probing with clips of 5 frames
        └── sv2lstg_lin_probe_${DETECTOR}_10.py
        └── sv2lstg_lin_probe_${DETECTOR}_15.py
    ... # one folder for each detector
    └── select_dataset.sh
└── datasets/
    └── endoscapes/
        └── endoscapes_instance.py # dataset cfg to load a frame and any associated annotations
        └── endoscapes_vid_instance.py # dataset cfg to load a clip and any associated annotations
        └── endoscapes_vid_instance_load_graphs.py # dataset cfg to load a clip, precomputed latent graphs for each frame, and any associated annotations
    ... # same structure for each dataset
```

## Models
```shell
└── model
    └── lg.py # LG-CVS
    └── deepcvs.py # DeepCVS
    └── simple_classifier.py # R50 & R50-DetInit
    └── sv2lstg.py # SV2LSTG
    └── deepcvs_temporal.py # DC-Temporal
    └── predictor_heads
        └── graph.py # graph head in LG-CVS
        └── reconstruction.py # reconstruction head for all
        └── ds.py # downstream classification head in LG-CVS
        ... # additional model components
    ... # additional model components
```

## Training and Testing
We provide instructions to train each of model on dataset `${DATASET}` using underlying object detector `${DETECTOR}` and clips of length `${CLIP_SIZE}`.

### Select Dataset
Before training any object detector/downstream classification model, the dataset needs to be selected.
```shell
cd configs/models
./select_dataset.sh ${DATASET}
cd ../..
```

### Object Detector

To train the downstream models (with the exception of the simple classifier), an object detector must first be trained.
We provide example commands for training and testing diffent object detectors below.

**Train**
```shell
mim train mmdet configs/models/${DETECTOR}/lg_${DETECTOR}.py
```

**Test**
```shell
mim test mmdet configs/models/${DETECTOR}/lg_${DETECTOR}.py work_dirs/lg_${DETECTOR}/best_${DATASET}_{bbox/segm}_mAP_epoch_${BEST_VAL_EPOCH}.pth
```

### Single-Frame Models

Here, we provide example commands for training/testing each of the single-frame downstream classification methods (LG-CVS, DeepCVS, LayoutCVS, ResNet50-DetInit, ResNet50).

**LG**
```shell
mim train mmdet configs/models/${DETECTOR}/lg_ds_${DETECTOR}.py
mim test mmdet configs/models/${DETECTOR}/lg_ds_${DETECTOR}.py work_dirs/lg_ds_${DETECTOR}/best_${DATASET}_ds_${SELECTION_METRIC}_epoch_${EPOCH}.pth
```
OR 
```shell
# no reconstruction objective
mim train mmdet configs/models/${DETECTOR}/lg_ds_${DETECTOR}_no_recon.py
mim test mmdet configs/models/${DETECTOR}/lg_ds_${DETECTOR}_no_recon.py work_dirs/lg_ds_${DETECTOR}_no_recon/best_${DATASET}_ds_${SELECTION_METRIC}_epoch_${EPOCH}.pth
```

**DeepCVS**
```shell
mim train mmdet configs/models/${DETECTOR}/dc_${DETECTOR}.py
mim test mmdet configs/models/${DETECTOR}/dc_${DETECTOR}.py work_dirs/dc_${DETECTOR}/best_${DATASET}_ds_${SELECTION_METRIC}_epoch_${EPOCH}.pth
```
OR 
```shell
# no reconstruction objective
mim train mmdet configs/models/${DETECTOR}/dc_${DETECTOR}_no_recon.py
mim test mmdet configs/models/${DETECTOR}/dc_${DETECTOR}_no_recon.py work_dirs/dc_${DETECTOR}_no_recon/best_${DATASET}_ds_${SELECTION_METRIC}_epoch_${EPOCH}.pth
```

**LayoutCVS**
```shell
mim train mmdet configs/models/${DETECTOR}/layout_${DETECTOR}.py
mim test mmdet configs/models/${DETECTOR}/layout_${DETECTOR}.py work_dirs/layout_${DETECTOR}/best_${DATASET}_ds_${SELECTION_METRIC}_epoch_${EPOCH}.pth
```
OR
```shell
# no reconstruction objective
mim train mmdet configs/models/${DETECTOR}/layout_${DETECTOR}_no_recon.py
mim test mmdet configs/models/${DETECTOR}/layout_${DETECTOR}_no_recon.py work_dirs/layout_${DETECTOR}_no_recon/best_${DATASET}_ds_${SELECTION_METRIC}_epoch_${EPOCH}.pth
```

**ResNet50-DetInit**
```shell
mim train mmdet configs/models/simple_classifier_with_recon.py --cfg-options load_from=weights/${DATASET}/lg_{$DETECTOR}.pth --work-dir work_dirs/R50_DI_${DETECTOR}
mim test mmdet configs/models/simple_classifier_with_recon.py work_dirs/R50_DI_${DETECTOR}/best_${DATASET}_ds_${SELECTION_METRIC}_epoch_${EPOCH}.pth
```
OR 
```shell
# no reconstruction objective
mim train mmdet configs/models/simple_classifier.py --cfg-options load_from=weights/${DATASET}/lg_{$DETECTOR}.pth --work-dir work_dirs/R50_DI_${DETECTOR}_no_recon
mim test mmdet configs/models/simple_classifier.py work_dirs/R50_DI_${DETECTOR}_no_recon/best_${DATASET}_ds_${SELECTION_METRIC}_epoch_${EPOCH}.pth
```

**ResNet50**
```shell
mim train mmdet configs/models/simple_classifier.py
mim test mmdet configs/models/simple_classifier.py work_dirs/simple_classifier/best_${DATASET}_ds_${SELECTION_METRIC}_epoch_${EPOCH}.pth
```
OR 
```shell
# WITH reconstruction objective
mim train mmdet configs/models/simple_classifier_with_recon.py
mim test mmdet configs/models/simple_classifier_with_recon.py work_dirs/simple_classifier_with_recon/best_${DATASET}_ds_${SELECTION_METRIC}_epoch_${EPOCH}.pth
```

### Temporal Models
Here, we provide example commands for training/testing each of the spatiotemporal downstream classification methods (SV2LSTG, DC-Temporal). Note that we do not use a reconstruction objective for these methods.

**Encoding *S*urgical *V*ideos as *L*atent *S*patio*T*emporal *G*raphs (SV2LSTG)**
```shell
mim train mmdet configs/models/${DETECTOR}/sv2lstg_${DETECTOR}_${CLIP_SIZE}.py
mim test mmdet configs/models/${DETECTOR}/sv2lstg_${DETECTOR}_${CLIP_SIZE}.py work_dirs/sv2lstg_${DETECTOR}_${CLIP_SIZE}/best_${DATASET}_ds_${SELECTION_METRIC}_iter_${ITERATION}.pth
```
OR
```shell
# Linear Probing (No Finetuning Backbone)
mim train mmdet configs/models/${DETECTOR}/sv2lstg_lin_probe_${DETECTOR}_${CLIP_SIZE}.py
mim test mmdet configs/models/${DETECTOR}/sv2lstg_lin_probe_${DETECTOR}_${CLIP_SIZE}.py work_dirs/sv2lstg_lin_probe_${DETECTOR}_${CLIP_SIZE}/best_${DATASET}_ds_${SELECTION_METRIC}_epoch_${EPOCH}.pth
```

**DeepCVS-Temporal (DC-Temp)**
```shell
mim train mmdet configs/models/${DETECTOR}/dc_temp_${DETECTOR}_${CLIP_SIZE}.py
mim test mmdet configs/models/${DETECTOR}/dc_temp_${DETECTOR}_${CLIP_SIZE}.py work_dirs/dc_temp_${DETECTOR}_${CLIP_SIZE}/best_${DATASET}_ds_${SELECTION_METRIC}_epoch_${EPOCH}.pth
```

## Pretrained Model Weights

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

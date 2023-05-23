import os
import copy

# modify base for different detectors
_base_ = '../lg_ds_faster_rcnn_no_recon.py'

# modify load_from
load_from = _base_.load_from.replace('no_recon', 'moco_lap')

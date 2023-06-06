import os
import copy

# modify base for different detectors
_base_ = '../lg_ds_faster_rcnn.py'

# modify load_from
load_from = _base_.load_from.replace('no_recon', 'ssl')

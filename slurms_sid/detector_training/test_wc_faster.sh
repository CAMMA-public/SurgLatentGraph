#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH -p gpu_p13
#SBATCH -J test_wc_faster
#SBATCH --error test_wc_faster_error.log
#SBATCH --output test_wc_faster.log
#SBATCH -A lbw@v100
#SBATCH -C v100-32g

module purge
module load anaconda-py3/2019.03
module load gcc/9.3.0
module load cuda/10.2
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/lib64
export MMDETECTION=${WORK}/mmdet_files
export PYTHONPATH=${PYTHONPATH}:/gpfsscratch/rech/lbw/uou65jw/sid/latentgraph


cd $SCRATCH/sid/latentgraph
source $(conda info --base)/bin/activate
conda activate camma

cd ./configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py configs/models/faster_rcnn/lg_faster_rcnn.py wc/best_wc_bbox_mAP_epoch_20.pth  --work-dir lg_faster_rcnn/wc && \


cd ./configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py configs/models/faster_rcnn/lg_faster_rcnn.py wc/best_wc_bbox_mAP_epoch_20.pth --work-dir lg_faster_rcnn/endoscapes &
wait

#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH -p gpu_p13
#SBATCH -J detector_wc_faster
#SBATCH --error detector_wc_faster_error.log
#SBATCH --output detector_wc_faster.log
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
        python ${MMDETECTION}/tools/train.py configs/models/faster_rcnn/lg_faster_rcnn.py --work-dir wc/lg_faster_rcnn &
wait

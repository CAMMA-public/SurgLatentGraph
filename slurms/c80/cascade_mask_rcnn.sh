#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --qos=qos_gpu-t4
#SBATCH --gres=gpu:2
#SBATCH --time=20:00:00
#SBATCH -p gpu_p13
#SBATCH -J latentgraph_cascade_mask_rcnn
#SBATCH --error latentgraph_cascade_mask_rcnn_error.log
#SBATCH --output latentgraph_cascade_mask_rcnn.log
#SBATCH -A lbw@v100
#SBATCH -C v100-32g

module purge
module load anaconda-py3/2019.03
module load gcc/9.3.0
module load cuda/10.2
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib64
export MMDETECTION=$WORK/mmdet_files
export PYTHONPATH=${PYTHONPATH}:/gpfswork/rech/lbw/uou65jw/latentgraph

cd $WORK/latentgraph
source $(conda info --base)/bin/activate
conda activate camma

./slurms/run_all_mgpu.sh cascade_mask_rcnn c80

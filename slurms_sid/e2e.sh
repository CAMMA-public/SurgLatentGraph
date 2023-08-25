#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH -p gpu_p13
#SBATCH -J driver
#SBATCH --error driver_error.log
#SBATCH --output driver.log
#SBATCH -A lbw@v100

module purge
module load anaconda-py3/2019.03
module load gcc/9.3.0
module load cuda/10.2
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/lib64
export MMDETECTION=${WORK}/mmdet_files
export PYTHONPATH=${PYTHONPATH}:/gpfsscratch/rech/lbw/uou65jw/sid/latentgraph_new/


cd $SCRATCH/sid/latentgraph_new
source $(conda info --base)/bin/activate
conda activate camma

./run_all_mgpu.sh mask_rcnn endoscapes endoscapes

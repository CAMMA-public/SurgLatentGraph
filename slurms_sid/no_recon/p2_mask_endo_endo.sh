#!/bin/bash
#SBATCH -N 4
#SBATCH --cpus-per-task 10
#SBATCH --gres=gpu:4
#SBATCH --time=15:00:00
#SBATCH -p gpu_p2
#SBATCH -J mask_endo_endo
#SBATCH --error mask_endo_endo_error.log
#SBATCH --output mask_endo_endo.log
#SBATCH -A lbw@v100


module purge
module load anaconda-py3/2019.03
module load gcc/9.3.0
module load cuda/10.2
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/lib64
export MMDETECTION=${WORK}/mmdet_files
export PYTHONPATH=${PYTHONPATH}:/gpfsscratch/rech/lbw/uou65jw/sid/latentgraph/


cd $SCRATCH/sid/latentgraph
source $(conda info --base)/bin/activate
conda activate camma

./slurms_sid/run_all_mgpu.sh mask_rcnn endoscapes endoscapes
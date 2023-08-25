#!/bin/bash
#SBATCH -N 4
#SBATCH --cpus-per-task 10
#SBATCH --gres=gpu:4
#SBATCH --time=15:00:00
#SBATCH -p gpu_p13
#SBATCH -J faster_endo_wc
#SBATCH --error faster_endo_wc_error.log
#SBATCH --output faster_endo_wc.log
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

./slurms_sid/run_all_mgpu.sh faster_rcnn endoscapes wc

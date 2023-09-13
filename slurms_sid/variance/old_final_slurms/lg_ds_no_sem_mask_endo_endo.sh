#!/bin/bash
#SBATCH -N 6
#SBATCH --cpus-per-task 10
#SBATCH --gres=gpu:6
#SBATCH --time=10:00:00
#SBATCH -p gpu_p13
#SBATCH -J lg_ds_no_sem_mask_endo_endo
#SBATCH --error lg_ds_no_sem_mask_endo_endo_error.log
#SBATCH --output lg_ds_no_sem_mask_endo_endo.log
#SBATCH -A lbw@v100
#SBATCH -C v100-32g

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

./slurms_sid/variance/run_test.sh lg_ds_no_sem mask_rcnn endoscapes endoscapes

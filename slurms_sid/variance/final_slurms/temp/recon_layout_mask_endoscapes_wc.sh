#!/bin/bash 
#SBATCH -N 3 
#SBATCH --cpus-per-task 10 
#SBATCH --gres=gpu:3 
#SBATCH --time=10:00:00 
#SBATCH -p gpu_p13 
#SBATCH -J recon_layout_mask_endoscapes_wc 
#SBATCH --error recon_layout_mask_endoscapes_wc_error.log 
#SBATCH --output recon_layout_mask_endoscapes_wc.log 
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

./slurms_sid/variance/run_test_recon.sh layout mask_rcnn endoscapes wc 

#!/bin/bash
#SBATCH -N 1
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH -p gpu_p13
#SBATCH -J test3
#SBATCH --error test3_error.log
#SBATCH --output test3.log
#SBATCH -A lbw@v100
#SBATCH -C v100-32g


module purge
module load anaconda-py3/2019.03
module load gcc/9.3.0
module load cuda/10.2
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/lib64
export MMDETECTION=${WORK}/mmdet_files
export PYTHONPATH=${PYTHONPATH}:/gpfsscratch/rech/lbw/uou65jw/sid/latentgraph/


cd $SCRATCH/sid/latentgraph/
source $(conda info --base)/bin/activate
conda activate camma


cd configs/models/ && \
./select_dataset.sh wc && \
cd ../.. && \
python ${MMDETECTION}/tools/test.py $SCRATCH/sid/latentgraph/configs/models/mask_rcnn/layout_mask_rcnn_no_recon.py $SCRATCH/sid/work_dirs_domain_adaptation_run_1/work_dirs/layout_mask_rcnn_endoscapes_endoscapes/best_endoscapes_ds_average_precision_epoch_15.pth --work-dir work_dirs/test3 && \
        # test on wc

wait

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


cd ../configs/models/ && \
./select_dataset.sh wc && \
cd ../.. && \
python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_mask_no_recon.py weights/wc/lg_mask_rcnn_no_recon --work-dir work_dirs/test && \
        # test on wc

wait

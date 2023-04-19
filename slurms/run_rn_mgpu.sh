detector=$1
base_cfg_dir=configs/models
dataset='endoscapes'

# RUN MT (WITH/WITHOUT RECON IN PARALLEL)

# train
export CUDA_VISIBLE_DEVICES=0 && python $MMDETECTION/tools/train.py ${base_cfg_dir}/simple_cvs_classifier.py &
export CUDA_VISIBLE_DEVICES=1 && python $MMDETECTION/tools/train.py ${base_cfg_dir}/simple_cvs_classifier_with_recon.py &
wait

# test
export CUDA_VISIBLE_DEVICES=0 && python $MMDETECTION/tools/test.py $(ls work_dirs/mt_${detector}/best_${dataset}* | tail -1) &
export CUDA_VISIBLE_DEVICES=1 && python $MMDETECTION/tools/test.py $(ls work_dirs/mt_${detector}_with_recon/best_${dataset}* | tail -1) &
wait

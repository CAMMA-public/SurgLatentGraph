detector=$1
cfg_dir=temporal_cfgs

# 5 frames
num_temp_frames=5
export CUDA_VISIBLE_DEVICES=0 && python ${MMDETECTION}/tools/train.py ${cfg_dir}/sv2lstg_${detector}_${num_temp_frames}.py && \
    python ${MMDETECTION}/tools/test.py ${cfg_dir}/sv2lstg_${detector}_${num_temp_frames}.py $(ls work_dirs/sv2lstg_${detector}_${num_temp_frames}/best_${dataset}* | tail -1) &

# 10 frames
num_temp_frames=10
export CUDA_VISIBLE_DEVICES=1 && python ${MMDETECTION}/tools/train.py ${cfg_dir}/sv2lstg_${detector}_${num_temp_frames}.py && \
    python ${MMDETECTION}/tools/test.py ${cfg_dir}/sv2lstg_${detector}_${num_temp_frames}.py $(ls work_dirs/sv2lstg_${detector}_${num_temp_frames}/best_${dataset}* | tail -1) &

# 15 frames
num_temp_frames=15
export CUDA_VISIBLE_DEVICES=2 && python ${MMDETECTION}/tools/train.py ${cfg_dir}/sv2lstg_${detector}_${num_temp_frames}.py && \
    python ${MMDETECTION}/tools/test.py ${cfg_dir}/sv2lstg_${detector}_${num_temp_frames}.py $(ls work_dirs/sv2lstg_${detector}_${num_temp_frames}/best_${dataset}* | tail -1) &

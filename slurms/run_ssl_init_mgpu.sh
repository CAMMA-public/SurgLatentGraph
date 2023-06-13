detector=$1
dataset=$2
base_cfg_dir=configs/models
cfg_dir=${base_cfg_dir}/${detector}/ssl_init

# RUN LG PRETRAINING (TRAINING OF DETECTOR)
export CUDA_VISIBLE_DEVICES=0 && python ${MMDETECTION}/tools/train.py ${cfg_dir}/lg_${detector}_ssl.py
best_model_path=$(ls work_dirs/lg_${detector}_ssl/best_${dataset}* | tail -1)
echo $best_model_path

# copy checkpoint to weights
mkdir -p weights/${dataset}
cp $best_model_path weights/${dataset}/lg_${detector}_ssl.pth

# test detector
export CUDA_VISIBLE_DEVICES=0 && python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_${detector}_ssl.py weights/${dataset}/lg_${detector}_ssl.pth

# extract backbone weights from lg detector weights
python weights/extract_bb_weights.py weights/${dataset}/lg_${detector}_ssl.pth

# RUN LG DS (using ssl weights, ssl det weights in parallel)

# train
export CUDA_VISIBLE_DEVICES=0 && python ${MMDETECTION}/tools/train.py ${cfg_dir}/lg_ds_${detector}_ssl.py &
export CUDA_VISIBLE_DEVICES=1 && python ${MMDETECTION}/tools/train.py ${cfg_dir}/lg_ds_${detector}_ssl_detector.py &
wait

# test
export CUDA_VISIBLE_DEVICES=0 && python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_ssl.py $(ls work_dirs/lg_ds_${detector}_ssl/best_${dataset}* | tail -1) &
export CUDA_VISIBLE_DEVICES=1 && python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_ssl_detector.py $(ls work_dirs/lg_ds_${detector}_ssl_detector/best_${dataset}* | tail -1) &
wait

# RUN MT (using ssl det weights)

# train
export CUDA_VISIBLE_DEVICES=1 && python ${MMDETECTION}/tools/train.py ${base_cfg_dir}/simple_cvs_classifier_with_recon.py --work-dir ./work_dirs/mt_${detector}_with_recon_ssl --cfg-options load_from=weights/${dataset}/lg_${detector}_ssl_bb.pth &
wait

# test
export CUDA_VISIBLE_DEVICES=1 && python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier_with_recon.py $(ls work_dirs/mt_${detector}_with_recon_ssl/best_${dataset}* | tail -1) --work-dir ./work_dirs/mt_${detector}_with_recon_ssl &
wait

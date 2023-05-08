detector=$1
dataset=$2
base_cfg_dir=configs/models
cfg_dir=${base_cfg_dir}/${detector}

# RUN LG PRETRAINING (TRAINING OF DETECTOR)
export CUDA_VISIBLE_DEVICES=0 && python ${MMDETECTION}/tools/train.py ${cfg_dir}/lg_${detector}.py
best_model_path=$(ls work_dirs/lg_${detector}/best_${dataset}* | tail -1)
echo $best_model_path

# copy checkpoint to weights
mkdir -p weights/${dataset}
cp $best_model_path weights/${dataset}/lg_${detector}_no_recon.pth

export CUDA_VISIBLE_DEVICES=0 && python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_${detector}.py weights/${dataset}/lg_${detector}_no_recon.pth


# extract backbone weights from lg detector weights
python weights/extract_bb_weights.py weights/${dataset}/lg_${detector}_no_recon.pth

# RUN LG DS (WITH/WITHOUT RECON IN PARALLEL)

# train
export CUDA_VISIBLE_DEVICES=0 && python ${MMDETECTION}/tools/train.py ${cfg_dir}/lg_ds_${detector}.py &
export CUDA_VISIBLE_DEVICES=1 && python ${MMDETECTION}/tools/train.py ${cfg_dir}/lg_ds_${detector}_no_recon.py &
wait

# test
export CUDA_VISIBLE_DEVICES=0 && python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}.py $(ls work_dirs/lg_ds_${detector}/best_${dataset}* | tail -1) &
export CUDA_VISIBLE_DEVICES=1 && python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_no_recon.py $(ls work_dirs/lg_ds_${detector}_no_recon/best_${dataset}* | tail -1) &
wait

# RUN LAYOUTCVS (WITH/WITHOUT RECON IN PARALLEL)
# train
export CUDA_VISIBLE_DEVICES=0 && python ${MMDETECTION}/tools/train.py ${cfg_dir}/layout_${detector}.py &
export CUDA_VISIBLE_DEVICES=1 && python ${MMDETECTION}/tools/train.py ${cfg_dir}/layout_${detector}_no_recon.py &
wait

# test
export CUDA_VISIBLE_DEVICES=0 && python ${MMDETECTION}/tools/test.py ${cfg_dir}/layout_${detector}.py $(ls work_dirs/layout_${detector}/best_${dataset}* | tail -1) &
export CUDA_VISIBLE_DEVICES=1 && python ${MMDETECTION}/tools/test.py ${cfg_dir}/layout_${detector}_no_recon.py $(ls work_dirs/layout_${detector}_no_recon/best_${dataset}* | tail -1) &
wait

# RUN DEEPCVS (WITH/WITHOUT RECON IN PARALLEL)

# train
export CUDA_VISIBLE_DEVICES=0 && python ${MMDETECTION}/tools/train.py ${cfg_dir}/dc_${detector}.py &
export CUDA_VISIBLE_DEVICES=1 && python ${MMDETECTION}/tools/train.py ${cfg_dir}/dc_${detector}_no_recon.py &
wait

# test
export CUDA_VISIBLE_DEVICES=0 && python ${MMDETECTION}/tools/test.py ${cfg_dir}/dc_${detector}.py $(ls work_dirs/dc_${detector}/best_${dataset}* | tail -1) &
export CUDA_VISIBLE_DEVICES=1 && python ${MMDETECTION}/tools/test.py ${cfg_dir}/dc_${detector}_no_recon.py $(ls work_dirs/dc_${detector}_no_recon/best_${dataset}* | tail -1) &
wait

# RUN MT (WITH/WITHOUT RECON IN PARALLEL)

# train
export CUDA_VISIBLE_DEVICES=0 && python ${MMDETECTION}/tools/train.py ${base_cfg_dir}/simple_cvs_classifier.py --work-dir ./work_dirs/mt_${detector} --cfg-options load_from=weights/${dataset}/lg_${detector}_bb.pth &
export CUDA_VISIBLE_DEVICES=1 && python ${MMDETECTION}/tools/train.py ${base_cfg_dir}/simple_cvs_classifier_with_recon.py --work-dir ./work_dirs/mt_${detector}_with_recon --cfg-options load_from=weights/${dataset}/lg_${detector}_bb.pth &
wait

# test
export CUDA_VISIBLE_DEVICES=0 && python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls work_dirs/mt_${detector}/best_${dataset}* | tail -1) --work-dir ./work_dirs/mt_${detector} &
export CUDA_VISIBLE_DEVICES=1 && python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier_with_recon.py $(ls work_dirs/mt_${detector}_with_recon/best_${dataset}* | tail -1) --work-dir ./work_dirs/mt_${detector}_with_recon &
wait

model=$1
detector=$2
dataset=$3
dataset2=$4
base_cfg_dir=configs/models
cfg_dir=${base_cfg_dir}/${detector}


##Make working directory
cd $SCRATCH/sid/latentgraph/work_dirs
mkdir ${model}_${detector}_${dataset}_${dataset2}_italy
cd ..












################################################################################################################################################################################################################################################################
if [[ "$model" = "mt" ]]; then
## RUN MT (WITH/WITHOUT RECON IN PARALLEL)

cd configs/models/ && \
./select_dataset.sh italy && \
cd ../.. && \


##RUN1
export CUDA_VISIBLE_DEVICES=0 && \
        ## test on italy
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls ../variance_work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}_italy/run1 && \

##RUN2
export CUDA_VISIBLE_DEVICES=0 && \
        ## test on italy
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls ../variance_work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}_italy/run2 && \



##RUN1
export CUDA_VISIBLE_DEVICES=0 && \
        ## test on italy
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls ../variance_work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}_italy/run3 &








################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################


else

################################################################################################################################################################################################################################################################
## RUN ALL OTHER MODELS WITHOUT RECON


cd configs/models/
./select_dataset.sh italy
cd ../..
wait


##RUN 1
export CUDA_VISIBLE_DEVICES=0 && \
        # test on italy
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_recon.py $(ls ../variance_work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}_italy/run1/ && \


##RUN 2
export CUDA_VISIBLE_DEVICES=0 && \
        # test on italy
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_recon.py $(ls ../variance_work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}_italy/run2/ && \


##RUN 1
export CUDA_VISIBLE_DEVICES=0 && \
        # test on italy
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_recon.py $(ls ../variance_work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}_italy/run3/ &


################################################################################################################################################################################################################################################################
fi
################################################################################################################################################################################################################################################################


wait


model=$1
detector=$2
dataset=$3
dataset2=$4
base_cfg_dir=configs/models
cfg_dir=${base_cfg_dir}/${detector}


##Make working directory
cd $SCRATCH/sid/latentgraph/work_dirs
mkdir recon_${model}_${detector}_${dataset}_${dataset2}_italy
cd ..






################################################################################################################################################################################################################################################################
if [[ "$model" = "mt" ]]; then
## RUN MT WITH RECON
cd configs/models/
./select_dataset.sh italy
cd ../..
wait

#RUN 1
export CUDA_VISIBLE_DEVICES=0 && \
        ## test on italy
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier_with_recon.py $(ls ../reconstruction_objective_work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}_italy/run1/ && \



#RUN 2
export CUDA_VISIBLE_DEVICES=0 && \
        ## test on italy
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier_with_recon.py $(ls ../reconstruction_objective_work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}_italy/run2/ && \



#RUN 3
export CUDA_VISIBLE_DEVICES=0 && \
        ## test on italy
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier_with_recon.py $(ls ../reconstruction_objective_work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}_italy/run3/ &
################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################



else

################################################################################################################################################################################################################################################################
## RUN ALL OTHER MODELS WITH RECON
##RUN 1
cd configs/models/
./select_dataset.sh italy
cd ../..
wait


#RUN 1
export CUDA_VISIBLE_DEVICES=0 && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}.py $(ls ../reconstruction_objective_work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}_italy/run1/ && \




#RUN 2
export CUDA_VISIBLE_DEVICES=0 && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}.py $(ls ../reconstruction_objective_work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}_italy/run2/ && \


#RUN 1
export CUDA_VISIBLE_DEVICES=0 && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}.py $(ls ../reconstruction_objective_work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}_italy/run3/ &






################################################################################################################################################################################################################################################################
fi
################################################################################################################################################################################################################################################################


wait


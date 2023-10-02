model=$1
detector=$2
dataset=$3
dataset2=$4
base_cfg_dir=configs/models
cfg_dir=${base_cfg_dir}/${detector}

#Copy latentgraph directory to jobscratch
cd $JOBSCRATCH
echo $JOBSCRATCH
cp -r $SCRATCH/sid/latentgraph  $JOBSCRATCH
cd latentgraph



## RUN LG PRETRAINING (TRAINING OF DETECTOR)
#cd ./configs/models/
#./select_dataset.sh ${dataset}
#cd ../..

#export CUDA_VISIBLE_DEVICES=0 && python ${MMDETECTION}/tools/train.py ${cfg_dir}/lg_${detector}.py --work-dir lg_${detector}_${dataset}
#best_model_path=$(ls work_dirs/lg_${detector}_${dataset}/best_${dataset}* | tail -1)
#echo $best_model_path
#
## copy checkpoint to weights
#mkdir -p weights/${dataset}
#cp $best_model_path weights/${dataset}/lg_${detector}_no_recon.pth
#
#export CUDA_VISIBLE_DEVICES=0 && python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_${detector}.py weights/${dataset}/lg_${detector}_no_recon.pth
#
#
# extract backbone weights from lg detector weights
python weights/extract_bb_weights.py weights/${dataset}/lg_${detector}_no_recon.pth








################################################################################################################################################################################################################################################################
if ["$model" = "mt"]; then
## RUN MT (WITH/WITHOUT RECON IN PARALLEL)
#
cd configs/models/
./select_dataset.sh ${dataset2}
cd ../..
##RUN1
export CUDA_VISIBLE_DEVICES=0 && \
        python ${MMDETECTION}/tools/train.py ${base_cfg_dir}/simple_cvs_classifier.py --work-dir work_dirs/mt_${detector}_${dataset}_${dataset2}_run1 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon_bb.pth && \
        ## test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls work_dirs/mt_${detector}_${dataset}_${dataset2}_run1/best_${dataset2}* | tail -1) --work-dir work_dirs/mt_${detector}_${dataset}_${dataset2}_run1/endoscapes && \
        ## test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls work_dirs/mt_${detector}_${dataset}_${dataset2}_run1/best_${dataset2}* | tail -1) --work-dir work_dirs/mt_${detector}_${dataset}_${dataset2}_run1/wc && \
       #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/mt_${detector}_${dataset}_${dataset2}_run1 $SCRATCH/sid/latentgraph/work_dirs/mt_${detector}_${dataset}_${dataset2}_run1 &

##RUN2
export CUDA_VISIBLE_DEVICES=1 && \
        python ${MMDETECTION}/tools/train.py ${base_cfg_dir}/simple_cvs_classifier.py --work-dir work_dirs/mt_${detector}_${dataset}_${dataset2}_run2 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon_bb.pth && \
        ## test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls work_dirs/mt_${detector}_${dataset}_${dataset2}_run2/best_${dataset2}* | tail -1) --work-dir work_dirs/mt_${detector}_${dataset}_${dataset2}_run2/endoscapes && \
        ## test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls work_dirs/mt_${detector}_${dataset}_${dataset2}_run2/best_${dataset2}* | tail -1) --work-dir work_dirs/mt_${detector}_${dataset}_${dataset2}_run2/wc && \
       #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/mt_${detector}_${dataset}_${dataset2}_run2 $SCRATCH/sid/latentgraph/work_dirs/mt_${detector}_${dataset}_${dataset2}_run2 &

#RUN3
export CUDA_VISIBLE_DEVICES=2 && \
        python ${MMDETECTION}/tools/train.py ${base_cfg_dir}/simple_cvs_classifier.py --work-dir work_dirs/mt_${detector}_${dataset}_${dataset2}_run3 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon_bb.pth && \
        ## test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls work_dirs/mt_${detector}_${dataset}_${dataset2}_run3/best_${dataset2}* | tail -1) --work-dir work_dirs/mt_${detector}_${dataset}_${dataset2}_run3/endoscapes && \
        ## test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls work_dirs/mt_${detector}_${dataset}_${dataset2}_run3/best_${dataset2}* | tail -1) --work-dir work_dirs/mt_${detector}_${dataset}_${dataset2}_run3/wc && \
       #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/mt_${detector}_${dataset}_${dataset2}_run3 $SCRATCH/sid/latentgraph/work_dirs/mt_${detector}_${dataset}_${dataset2}_run3 &

################################################################################################################################################################################################################################################################
## RUN MT WITH RECON

#RUN 1
export CUDA_VISIBLE_DEVICES=3 && \
        python ${MMDETECTION}/tools/train.py ${base_cfg_dir}/simple_cvs_classifier_with_recon.py --work-dir work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run1 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon_bb.pth && \
        ## test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier_with_recon.py $(ls work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run1/endoscapes && \
        ## test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier_with_recon.py $(ls work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run1/wc && \
       #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run1 $SCRATCH/sid/latentgraph/work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run1 &


#RUN 2
export CUDA_VISIBLE_DEVICES=4 && \
        python ${MMDETECTION}/tools/train.py ${base_cfg_dir}/simple_cvs_classifier_with_recon.py --work-dir work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run2 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon_bb.pth && \
        ## test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier_with_recon.py $(ls work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run2/endoscapes && \
        ## test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier_with_recon.py $(ls work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run2/wc && \
       #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run2 $SCRATCH/sid/latentgraph/work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run2 &



#RUN 3
export CUDA_VISIBLE_DEVICES=5 && \
        python ${MMDETECTION}/tools/train.py ${base_cfg_dir}/simple_cvs_classifier_with_recon.py --work-dir work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run3 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon_bb.pth && \
        ## test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier_with_recon.py $(ls work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run3/endoscapes && \
        ## test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier_with_recon.py $(ls work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run3/wc && \
       #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run3 $SCRATCH/sid/latentgraph/work_dirs/recon_mt_${detector}_${dataset}_${dataset2}/run3 &















################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################
else

################################################################################################################################################################################################################################################################
## RUN ALL OTHER MODELS WITHOUT RECON
##RUN 1

cd configs/models/
./select_dataset.sh ${dataset2}
cd ../..
wait
export CUDA_VISIBLE_DEVICES=0 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/${model}_${detector}_no_recon.py --work-dir  work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_recon.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1/endoscapes && \
        # test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_recon.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1/wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1 $SCRATCH/sid/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1 &

##RUN 2
export CUDA_VISIBLE_DEVICES=1 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/${model}_${detector}_no_recon.py --work-dir  work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_recon.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2/endoscapes && \
        # test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_recon.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2/wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2 $SCRATCH/sid/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2 &

##RUN 3
export CUDA_VISIBLE_DEVICES=2 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/${model}_${detector}_no_recon.py --work-dir  work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_recon.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3/endoscapes && \
        # test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_recon.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3/wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3 $SCRATCH/sid/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3 &
################################################################################################################################################################################################################################################################
## RUN ALL OTHER MODELS WITH RECON
##RUN 1
export CUDA_VISIBLE_DEVICES=3 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/${model}_${detector}.py --work-dir  work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run1 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}.py $(ls work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run1/endoscapes && \
        # test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}.py $(ls work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run1/wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run1 $SCRATCH/sid/latentgraph/work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run1 &

##RUN 2
export CUDA_VISIBLE_DEVICES=4 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/${model}_${detector}.py --work-dir  work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run2 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}.py $(ls work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run2/endoscapes && \
        # test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}.py $(ls work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run2/wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run2 $SCRATCH/sid/latentgraph/work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run2 &


##RUN 3
export CUDA_VISIBLE_DEVICES=5 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/${model}_${detector}.py --work-dir  work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run3 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}.py $(ls work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run3/endoscapes && \
        # test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}.py $(ls work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run3/wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run3 $SCRATCH/sid/latentgraph/work_dirs/recon_${model}_${detector}_${dataset}_${dataset2}/run3 &


################################################################################################################################################################################################################################################################
fi
################################################################################################################################################################################################################################################################







if ["$model" = "lg_ds"]; then







################################################################################################################################################################################################################################################################
##NO_SEM_NO_RECON

cd configs/models/
./select_dataset.sh ${dataset2}
cd ../..
wait


##RUN 1
export CUDA_VISIBLE_DEVICES=6 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/lg_ds_${detector}_no_sem_no_recon.py --work-dir   work_dirs/no_sem_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_no_sem_no_recon.py $(ls  work_dirs/no_sem_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir  work_dirs/no_sem_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1/endoscapes && \
        # test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_no_sem_no_recon.py $(ls  work_dirs/no_sem_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir  work_dirs/no_sem_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1/wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/no_sem_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1 $SCRATCH/sid/latentgraph/work_dirs/no_sem_no_recon_without_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1 &

##RUN 2
export CUDA_VISIBLE_DEVICES=7 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/lg_ds_${detector}_no_sem_no_recon.py --work-dir   work_dirs/no_sem_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_no_sem_no_recon.py $(ls  work_dirs/no_sem_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir  work_dirs/no_sem_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2/endoscapes && \
        # test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_no_sem_no_recon.py $(ls  work_dirs/no_sem_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir  work_dirs/no_sem_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2/wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/no_sem_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2 $SCRATCH/sid/latentgraph/work_dirs/no_sem_no_recon_without_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2 &


##RUN 3
export CUDA_VISIBLE_DEVICES=8 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/lg_ds_${detector}_no_sem_no_recon.py --work-dir   work_dirs/no_sem_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_no_sem_no_recon.py $(ls  work_dirs/no_sem_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir  work_dirs/no_sem_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3/endoscapes && \
        # test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_no_sem_no_recon.py $(ls  work_dirs/no_sem_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir  work_dirs/no_sem_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3/wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/no_sem_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3 $SCRATCH/sid/latentgraph/work_dirs/no_sem_no_recon_without_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3 &





################################################################################################################################################################################################################################################################
##NO_SEM_YES_RECON
cd configs/models/
./select_dataset.sh ${dataset2}
cd ../..
wait

#RUN1
export CUDA_VISIBLE_DEVICES=9 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/${model}_${detector}_no_sem.py --work-dir  work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run1 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_sem.py $(ls work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run1/endoscapes && \
        # test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_sem.py $(ls work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run1/wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run1 $SCRATCH/sid/latentgraph/work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run1 &

#RUN2
export CUDA_VISIBLE_DEVICES=10 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/${model}_${detector}_no_sem.py --work-dir  work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run2 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_sem.py $(ls work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run2/endoscapes && \
        # test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_sem.py $(ls work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run2/wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run2 $SCRATCH/sid/latentgraph/work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run2 &


#RUN3
export CUDA_VISIBLE_DEVICES=11 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/${model}_${detector}_no_sem.py --work-dir  work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run3 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_sem.py $(ls work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run3/endoscapes && \
        # test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_sem.py $(ls work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run3/wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run3 $SCRATCH/sid/latentgraph/work_dirs/no_sem_yes_recon_${model}_${detector}_${dataset}_${dataset2}/run3 &


################################################################################################################################################################################################################################################################
##NO_VIZ_NO_RECON
cd configs/models/
./select_dataset.sh ${dataset2}
cd ../..
wait

##RUN 1
export CUDA_VISIBLE_DEVICES=12 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/lg_ds_${detector}_no_viz_no_recon.py --work-dir  work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_no_viz_no_recon.py $(ls work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1/endoscapes && \
        # test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_no_viz_no_recon.py $(ls work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1/wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1 $SCRATCH/sid/latentgraph/work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1 &

##RUN 2
export CUDA_VISIBLE_DEVICES=13 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/lg_ds_${detector}_no_viz_no_recon.py --work-dir  work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_no_viz_no_recon.py $(ls work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2/endoscapes && \
        # test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_no_viz_no_recon.py $(ls work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2/wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2 $SCRATCH/sid/latentgraph/work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2 &



export CUDA_VISIBLE_DEVICES=14 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/lg_ds_${detector}_no_viz_no_recon.py --work-dir  work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_no_viz_no_recon.py $(ls work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3/endoscapes && \
        # test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_no_viz_no_recon.py $(ls work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3/wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3 $SCRATCH/sid/latentgraph/work_dirs/no_viz_no_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3 &

################################################################################################################################################################################################################################################################
##NO_VIZ_YES_RECON
cd configs/models/
./select_dataset.sh ${dataset2}
cd ../..
wait

##RUN1
export CUDA_VISIBLE_DEVICES=15 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/lg_ds_${detector}_no_viz.py --work-dir  work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_no_viz.py $(ls work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1/endoscapes && \
        # test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_no_viz.py $(ls work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1/wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1 $SCRATCH/sid/latentgraph/work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run1 &


##RUN2
export CUDA_VISIBLE_DEVICES=16 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/lg_ds_${detector}_no_viz.py --work-dir  work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_no_viz.py $(ls work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2/endoscapes && \
        # test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_no_viz.py $(ls work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2/wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2 $SCRATCH/sid/latentgraph/work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run2 &



##RUN3
export CUDA_VISIBLE_DEVICES=17 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/lg_ds_${detector}_no_viz.py --work-dir  work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_no_viz.py $(ls work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3/endoscapes && \
        # test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/lg_ds_${detector}_no_viz.py $(ls work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3/wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3 $SCRATCH/sid/latentgraph/work_dirs/no_viz_yes_recon_lg_ds_${detector}_${dataset}_${dataset2}/run3 &

################################################################################################################################################################################################################################################################
else
fi
################################################################################################################################################################################################################################################################
wait


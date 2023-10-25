model=$1
detector=$2
dataset=$3
dataset2=$4
base_cfg_dir=configs/models
cfg_dir=${base_cfg_dir}/${detector}


##Make working directory
cd $SCRATCH/sid/latentgraph/work_dirs
mkdir ${model}_${detector}_${dataset}_${dataset2}



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
if [[ "$model" = "mt" ]]; then
## RUN MT (WITH/WITHOUT RECON IN PARALLEL)
#
cd configs/models/
./select_dataset.sh ${dataset2}
cd ../..
##RUN1
export CUDA_VISIBLE_DEVICES=0 && \
        python ${MMDETECTION}/tools/train.py ${base_cfg_dir}/simple_cvs_classifier.py --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon_bb.pth && \
        ## test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1/endoscapes && \
        ## test on small_wc
        cd configs/models/ && \
        ./select_dataset.sh small_wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1/ && \
       #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1 $SCRATCH/sid/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1 && \

##RUN2
export CUDA_VISIBLE_DEVICES=0 && \
        python ${MMDETECTION}/tools/train.py ${base_cfg_dir}/simple_cvs_classifier.py --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon_bb.pth && \
        ## test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2/endoscapes && \
        ## test on small_wc
        cd configs/models/ && \
        ./select_dataset.sh small_wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2/small_wc && \
       #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2 $SCRATCH/sid/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2 && \

#RUN3
export CUDA_VISIBLE_DEVICES=0 && \
        python ${MMDETECTION}/tools/train.py ${base_cfg_dir}/simple_cvs_classifier.py --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon_bb.pth && \
        ## test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3/endoscapes && \
        ## test on small_wc
        cd configs/models/ && \
        ./select_dataset.sh small_wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3/small_wc && \
       #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3 $SCRATCH/sid/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3 &

################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################

elif [[ "$model" = "simple" ]]; then
## RUN SIMPLECVS (WITH/WITHOUT RECON IN PARALLEL)
#
cd configs/models/
./select_dataset.sh ${dataset2}
cd ../..
##RUN1
export CUDA_VISIBLE_DEVICES=0 && \
        python ${MMDETECTION}/tools/train.py ${base_cfg_dir}/simple_cvs_classifier.py --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1 && \
        ## test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1/endoscapes && \
        ## test on small_wc
        cd configs/models/ && \
        ./select_dataset.sh small_wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1/small_wc && \
       #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1 $SCRATCH/sid/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1 && \

##RUN2
export CUDA_VISIBLE_DEVICES=0 && \
        python ${MMDETECTION}/tools/train.py ${base_cfg_dir}/simple_cvs_classifier.py --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2 && \
        ## test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2/endoscapes && \
        ## test on small_wc
        cd configs/models/ && \
        ./select_dataset.sh small_wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2/small_wc && \
       #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2 $SCRATCH/sid/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2 && \

#RUN3
export CUDA_VISIBLE_DEVICES=0 && \
        python ${MMDETECTION}/tools/train.py ${base_cfg_dir}/simple_cvs_classifier.py --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3 && \
        ## test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3/endoscapes && \
        ## test on small_wc
        cd configs/models/ && \
        ./select_dataset.sh small_wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${base_cfg_dir}/simple_cvs_classifier.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3/small_wc && \
       #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3 $SCRATCH/sid/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3 &

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
        # test on small_wc
        cd configs/models/ && \
        ./select_dataset.sh small_wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_recon.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1/small_wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1 $SCRATCH/sid/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1 && \

##RUN 2
export CUDA_VISIBLE_DEVICES=0 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/${model}_${detector}_no_recon.py --work-dir  work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_recon.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2/endoscapes && \
        # test on small_wc
        cd configs/models/ && \
        ./select_dataset.sh small_wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_recon.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2/small_wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2 $SCRATCH/sid/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run2 && \

##RUN 3
export CUDA_VISIBLE_DEVICES=0 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/${model}_${detector}_no_recon.py --work-dir  work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3 --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_recon.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3/endoscapes && \
        # test on small_wc
        cd configs/models/ && \
        ./select_dataset.sh small_wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_recon.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3/small_wc && \
        #copy everything back
        cp -r $JOBSCRATCH/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3 $SCRATCH/sid/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run3 &
################################################################################################################################################################################################################################################################
fi
################################################################################################################################################################################################################################################################


wait


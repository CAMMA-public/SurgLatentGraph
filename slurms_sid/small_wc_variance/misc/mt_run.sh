detector=$1
dataset=$2
dataset2=$3
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
## RUN MT (WITH/WITHOUT RECON IN PARALLEL)
#
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
	

wait


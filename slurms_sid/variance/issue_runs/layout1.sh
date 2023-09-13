model=$1
detector=$2
dataset=$3
dataset2=$4
base_cfg_dir=configs/models
cfg_dir=${base_cfg_dir}/${detector}


##Make working directory



#Copy latentgraph directory to jobscratch
cd $SCRATCH/sid/latentgraph




cd configs/models/
./select_dataset.sh ${dataset2}
cd ../..
wait
##RUN
        # test on wc
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}_no_recon.py $(ls work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1/best_${dataset2}* | tail -1) --work-dir work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1/wc && \
        #copy everything back
        #cp -r $JOBSCRATCH/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1 $SCRATCH/sid/latentgraph/work_dirs/${model}_${detector}_${dataset}_${dataset2}/run1 &

################################################################################################################################################################################################################################################################


wait


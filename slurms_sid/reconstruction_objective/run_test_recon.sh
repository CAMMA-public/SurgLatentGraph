model=$1
detector=$2
dataset=$3
dataset2=$4
base_cfg_dir=configs/models
cfg_dir=${base_cfg_dir}/${detector}




##RUN 1
cd $SCRATCH/sid/latentgraph
cd configs/models/
./select_dataset.sh ${dataset2}
cd ../..
wait



export CUDA_VISIBLE_DEVICES=0 && \
        python ${MMDETECTION}/tools/train.py ${cfg_dir}/${model}_${detector}.py --work-dir  work_dirs/reconstruction_objective_test --cfg-options load_from=weights/${dataset}/lg_${detector}_no_recon.pth  && \
        # test on endoscapes
        cd configs/models/ && \
        ./select_dataset.sh endoscapes && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}.py $(ls work_dirs/reconstruction_objective_test/best_${dataset2}* | tail -1) --work-dir work_dirs/reconstruction_objective_test/endoscapes && \
        # test on wc
        cd configs/models/ && \
        ./select_dataset.sh wc && \
        cd ../.. && \
        python ${MMDETECTION}/tools/test.py ${cfg_dir}/${model}_${detector}.py $(ls work_dirs/reconstruction_objective_test/best_${dataset2}* | tail -1) --work-dir work_dirs/reconstruction_objective_test/wc && \

################################################################################################################################################################################################################################################################


wait


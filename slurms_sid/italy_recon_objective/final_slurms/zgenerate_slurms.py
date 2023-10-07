# f = open("demofile2.sh", "w")

detectors = ["mask", "faster"]
models = ["lg_ds", "lg_ds_no_sem", "lg_ds_no_viz", "mt"]
datasets = ["endoscapes"]

# detector = detectors[1]
# model = models[1]
# dataset1 = datasets[1]
# dataset2 = datasets[1]

for detector in detectors:
    for model in models:
        for dataset1 in datasets:
            for dataset2 in datasets:
                
                
                #WITH RECON
                fname1 = "recon_"+model+"_"+detector+"_"+dataset1+"_"+dataset2+"_italy.sh"
                f1 = open(fname1, "w")
                slurm1 = "#!/bin/bash \n#SBATCH -N 1 \n#SBATCH --cpus-per-task 10 \n#SBATCH --gres=gpu:1 \n#SBATCH --time=10:00:00 \n#SBATCH -p gpu_p2 \n#SBATCH -J recon_"+model+ "_"+detector+"_"+dataset1+"_"+dataset2+"_italy \n#SBATCH --error recon_"+model+ "_"+detector+"_"+dataset1+"_"+dataset2+"_italy_error.log \n#SBATCH --output recon_" + model+ "_"+detector+"_"+dataset1+"_"+dataset2+"_italy.log \n#SBATCH -A lbw@v100 \n\n\n\nmodule purge \nmodule load anaconda-py3/2019.03 \nmodule load gcc/9.3.0 \nmodule load cuda/10.2 \nexport LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/lib64 \nexport MMDETECTION=${WORK}/mmdet_files \nexport PYTHONPATH=${PYTHONPATH}:/gpfsscratch/rech/lbw/uou65jw/sid/latentgraph/ \n\n\ncd $SCRATCH/sid/latentgraph \nsource $(conda info --base)/bin/activate \nconda activate camma \n\n./slurms_sid/italy_recon_objective/run_test_recon.sh "+model+" "+detector+"_rcnn "+dataset1+" "+dataset2+" \n"
                f1.write(slurm1)
                f1.close()
                




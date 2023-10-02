# f = open("demofile2.sh", "w")

detectors = ["mask", "faster"]
models = ["lg_ds", "lg_ds_no_sem", "lg_ds_no_viz", "dc", "layout", "mt"]
datasets = ["endoscapes", "small_wc"]

# detector = detectors[1]
# model = models[1]
# dataset1 = datasets[1]
# dataset2 = datasets[1]

for detector in detectors:
    for model in models:
        for dataset1 in datasets:
            for dataset2 in datasets:
                
                #WITHOUT RECON
                fname = model+"_"+detector+"_"+dataset1+"_"+dataset2+".sh"
                f = open(fname, "w")
                slurm = "#!/bin/bash \n#SBATCH -N 3 \n#SBATCH --cpus-per-task 10 \n#SBATCH --gres=gpu:3 \n#SBATCH --time=10:00:00 \n#SBATCH -p gpu_p2 \n#SBATCH -J "+model+ "_"+detector+"_"+dataset1+"_"+dataset2+" \n#SBATCH --error "+model+ "_"+detector+"_"+dataset1+"_"+dataset2+"_error.log \n#SBATCH --output " + model+ "_"+detector+"_"+dataset1+"_"+dataset2+".log \n#SBATCH -A lbw@v100 \n\n\nmodule purge \nmodule load anaconda-py3/2019.03 \nmodule load gcc/9.3.0 \nmodule load cuda/10.2 \nexport LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/lib64 \nexport MMDETECTION=${WORK}/mmdet_files \nexport PYTHONPATH=${PYTHONPATH}:/gpfsscratch/rech/lbw/uou65jw/sid/latentgraph/ \n\n\ncd $SCRATCH/sid/latentgraph \nsource $(conda info --base)/bin/activate \nconda activate camma \n\n./slurms_sid/small_wc_variance/run_test.sh "+model+" "+detector+"_rcnn "+dataset1+" "+dataset2+" \n"
                f.write(slurm)
                f.close()
                
                #WITH RECON
                fname1 = "recon_"+model+"_"+detector+"_"+dataset1+"_"+dataset2+".sh"
                f1 = open(fname1, "w")
                slurm1 = "#!/bin/bash \n#SBATCH -N 3 \n#SBATCH --cpus-per-task 10 \n#SBATCH --gres=gpu:3 \n#SBATCH --time=10:00:00 \n#SBATCH -p gpu_p2 \n#SBATCH -J recon_"+model+ "_"+detector+"_"+dataset1+"_"+dataset2+" \n#SBATCH --error recon_"+model+ "_"+detector+"_"+dataset1+"_"+dataset2+"_error.log \n#SBATCH --output recon_" + model+ "_"+detector+"_"+dataset1+"_"+dataset2+".log  \n#SBATCH -A lbw@v100 \n\n\nmodule purge \nmodule load anaconda-py3/2019.03 \nmodule load gcc/9.3.0 \nmodule load cuda/10.2 \nexport LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/lib64 \nexport MMDETECTION=${WORK}/mmdet_files \nexport PYTHONPATH=${PYTHONPATH}:/gpfsscratch/rech/lbw/uou65jw/sid/latentgraph/ \n\n\ncd $SCRATCH/sid/latentgraph \nsource $(conda info --base)/bin/activate \nconda activate camma \n\n./slurms_sid/small_wc_variance/run_test_recon.sh "+model+" "+detector+"_rcnn "+dataset1+" "+dataset2+" \n"
                f1.write(slurm1)
                f1.close()
                



slurm

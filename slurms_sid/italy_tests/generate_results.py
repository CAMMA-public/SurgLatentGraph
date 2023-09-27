import re
import os
import json
import statistics







def calc(model, detector, dataset1, dataset2, dataset3, recon):
    
    if recon:
        root =os.path.expandvars("$SCRATCH/sid/latentgraph/work_dirs/recon_"+model+"_"+detector+"_rcnn_"+dataset1+"_"+dataset2+"_italy")
    else:
        root = os.path.expandvars("$SCRATCH/sid/latentgraph/work_dirs/"+model+"_"+detector+"_rcnn_"+dataset1+"_"+dataset2+"_italy")

    
    
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    #####################################################################################################################
    ## WITHOUT RECON

    ## RUN 1
    parent_directory = rf'{root}/run1'
    # List all subdirectories in the parent directory
    subdirectories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
    # Filter subdirectories that start with "2"
    filtered_directories = [d for d in subdirectories if d.startswith('2')]
    time_stamp = filtered_directories[0]
    run1 = rf"{root}/run1/{time_stamp}/{time_stamp}.json"
    f_run1 = open(run1,"r")
    data_run1 = json.load(f_run1)
    data_run1


    ## RUN 2
    parent_directory = rf'{root}/run2/'
    # List all subdirectories in the parent directory
    subdirectories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
    # Filter subdirectories that start with "2"
    filtered_directories = [d for d in subdirectories if d.startswith('2')]
    time_stamp = filtered_directories[0]
    run2 = rf"{root}/run2/{time_stamp}/{time_stamp}.json"
    f_run2 = open(run2,"r")
    data_run2 = json.load(f_run2)
    data_run2




    ## RUN 3
    parent_directory = rf'{root}/run3'
    # List all subdirectories in the parent directory
    subdirectories = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d))]
    # Filter subdirectories that start with "2"
    filtered_directories = [d for d in subdirectories if d.startswith('2')]
    time_stamp = filtered_directories[0]
    run3 = rf"{root}/run3/{time_stamp}/{time_stamp}.json"
    f_run3 = open(run3,"r")
    data_run3 = json.load(f_run3)
    data_run3



    print(model+" "+detector+" "+dataset1+" "+dataset2+" "+dataset3) 
    avg_precision = [data_run1[rf'{dataset3}/ds_average_precision'],data_run2[rf'{dataset3}/ds_average_precision'],data_run3[rf'{dataset3}/ds_average_precision']]
#     print(avg_precision)

    mean = statistics.mean(avg_precision)
    std = statistics.stdev(avg_precision)

#     print(str(mean)+"±"+str(std))

    return avg_precision, mean, std














if __name__ == "__main__":
    detectors = ["mask", "faster"]
    models = ["lg_ds", "lg_ds_no_sem", "lg_ds_no_viz", "dc", "layout", "mt"]
    datasets = ["endoscapes", "wc"]

    file_path = 'results_no_recon.txt'
    dataset3 = "italy"
    # Open the file in write mode ('w')
    file = open(file_path, 'w')



    for dataset1 in datasets:
        for detector in detectors:
            for model in models:
                for dataset2 in datasets:             

                    avg_precision, mean, std = calc(model, detector, dataset1, dataset2, "italy", False)
                    file.write(model+"_"+detector+"_rcnn_"+dataset1+"_"+dataset2+"_"+dataset3+" = "+ str(mean)+"±"+str(std)+"\n")
    
    file.close()

    
    file_path = 'results_yes_recon.txt'
    file = open(file_path, 'w')


    for dataset1 in datasets:
        for detector in detectors:
            for model in models:
                for dataset2 in datasets:

                    avg_precision, mean, std = calc(model, detector, dataset1, dataset2, "italy", True)
                    file.write("recon_"+model+"_"+detector+"_rcnn_"+dataset1+"_"+dataset2+"_"+dataset3+" = "+ str(mean)+"±"+str(std)+"\n")


    file.close()


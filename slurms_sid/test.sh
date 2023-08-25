#!/bin/bash
#SBATCH -N 1
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
#SBATCH -p gpu_p13
#SBATCH -J test
#SBATCH --error test_error.log
#SBATCH --output test.log
#SBATCH -A lbw@v100
#SBATCH -C v100-32g

echo "1"
cd $JOBSCRATCH
echo $JOBSCRATCH

echo "2"
mkdir test1

echo "3"
#Copy latentgraph directory to jobscratch
cp -r $SCRATCH/sid/latentgraph  $JOBSCRATCH/test1/

echo "4"
cd test1
echo "5"

cd latentgraph
mkdir sid123
echo "6"

cd ..
cp -r latentgraph $SCRATCH/sid/latentgraph/weights
echo "7"


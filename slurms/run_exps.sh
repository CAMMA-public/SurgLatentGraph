scripts=$(find $1 -name '*.sh' ! -name 'run_*')
for s in $scripts
do
    sbatch $s
    echo $s
done

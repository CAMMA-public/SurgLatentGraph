scripts=$(find -maxdepth 1 $1 -name '*.sh' ! -name 'run_*' ! -name 'select_*')
for s in $scripts
do
    sbatch $s
    echo $s
done

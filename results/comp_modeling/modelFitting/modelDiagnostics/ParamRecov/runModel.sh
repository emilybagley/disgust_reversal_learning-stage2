#!/bin/bash
#SBATCH --job-name paramRecov
#SBATCH --output /OUTdir/slurm-%x.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=1  ## number of tasks (analyses) to run
#SBATCH --cpus-per-task=1  ## the number of threads allocated to each task
#SBATCH --mem-per-cpu=200GB

#model params
modelName="1lr_stick1_blk3_allparamsep"
stanFile="../../STANfiles/1lr_stick1_blk3_allparamsep.stan"
dataFile="stan_data.rds" 
chains=4
parallel_chains=4
#iter_warmup=2000
#iter_sampling=2000
#thin=1
iter_warmup=10000
iter_sampling=10000
thin=10
save_warmup=FALSE
output_dir="modelOutput/csvs"

module load R/4.3.1_v2
export TMPDIR=/TEMPdir
echo "runModel.R"
echo "$modelName"
srun Rscript runModel.R $modelName $stanFile $dataFile $chains $parallel_chains $iter_warmup $iter_sampling $thin $save_warmup $output_dir

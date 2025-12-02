#!/bin/bash
#SBATCH --output /OUTdir/slurm-%x.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=1  ## number of tasks (analyses) to run
#SBATCH --cpus-per-task=1  ## the number of threads allocated to each task
#SBATCH --mem-per-cpu=200GB

#model params
modelName="$1"
stanFile="$2"
dataFile="$3"
chains="$4"
parallel_chains="$5"
iter_warmup="$6"
iter_sampling="$7"
thin="$8"
save_warmup="$9"

stan_dir=./STANfiles
mkdir "modelOutputs/""$modelName"
mkdir "modelOutputs/""$modelName""/csvs"
output_dir="modelOutputs/""$modelName""/csvs"
module load R/4.3.1_v2
export TMPDIR= /TEMPdir
echo "runModel.R"
echo "$modelName"
srun Rscript runModel.R $modelName $stanFile $dataFile $chains $parallel_chains $iter_warmup $iter_sampling $thin $save_warmup $output_dir

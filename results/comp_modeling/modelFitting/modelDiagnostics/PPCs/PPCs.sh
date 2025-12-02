#!/bin/bash
#SBATCH --job-name PPCs
#SBATCH --output /group/nord/DisgustReversalLearningModeling/finalModelComp/modelCompAndDiagnostics/OUTdir/slurm-%x.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=1  ## number of tasks (analyses) to run
#SBATCH --cpus-per-task=1  ## the number of threads allocated to each task
#SBATCH --mem-per-cpu=200GB

#model params
modelName="1lr_stick1_blk3_allparamsep"
dataFile="stanData_excluded_df.rds"

module load R/4.3.1_v2
echo "PCCs.R"
echo "$modelName"
srun Rscript PPCs.R $modelName $dataFile

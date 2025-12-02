#!/bin/bash
#SBATCH --job-name paramRecov
#SBATCH --output /group/nord/DisgustReversalLearningModeling/finalModelComp/modelCompAndDiagnostics/OUTdir/slurm-%x.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=1  ## number of tasks (analyses) to run
#SBATCH --cpus-per-task=1  ## the number of threads allocated to each task
#SBATCH --mem-per-cpu=200GB

#cd "/group/nord/DisgustReversalLearningModeling/finalModelComp/modelCompAndDiagnostics/ParamRecov"

echo "simulate data"
#sbatch --job-name=$jobName srun Rscript simData.R $modelName
module load R/4.3.1_v2
srun Rscript simData.R 
#!/bin/bash
dataFile="../csvs/stanData_excluded_df.rds"
chains=4
parallel_chains=4
iter_warmup=10000
iter_sampling=10000
thin=10
save_warmup=FALSE

stan_dir=./STANfiles
echo "run model"
for stanFile in "$stan_dir"/*.stan
do
    modelName=$(basename "$stanFile" .stan)
    if [[ "$modelName" == "random_blk3" ]]; then
        echo "Special submitting: $modelName"
        sbatch --job-name=$modelName --dependency=singleton runRandomModel.sh $modelName $stanFile $dataFile $chains $parallel_chains $iter_warmup $iter_sampling $thin $save_warmup
        continue
    else
        echo "Submitting: $modelName"
        sbatch --job-name=$modelName runModel.sh $modelName $stanFile $dataFile $chains $parallel_chains $iter_warmup $iter_sampling $thin $save_warmup
    fi
done

stan_dir=./STANfiles
echo "check model and extract params"
for stanFile in "$stan_dir"/*.stan
do
    modelName=$(basename "$stanFile" .stan)
    if [[ "$modelName" == "random_blk3" ]]; then
        echo "Not submitting: $modelName"
        continue
    else
        echo "Submitting: $modelName"
        sbatch --job-name=$modelName --dependency=singleton checkModel_extractParams.sh $modelName $stanFile $dataFile $chains $parallel_chains $iter_warmup $iter_sampling $thin $save_warmup
    fi
done

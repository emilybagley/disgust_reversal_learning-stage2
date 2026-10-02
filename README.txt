Contains all scripts for all analyses detailed in the stage 2 registered report. 
(NB therefore, there is some duplication with the stage 1 report)

Note, many of the results are in markdown form using quarto notebooks (necessary to the combination of R and Python in these analyses) which can more clearly be read on GitHub. Link to repo: https://github.com/emilybagley/disgust_reversal_learning-stage2 

Alternatively, this folder can be downloaded in its entirety - markdown files can be viewed and quarto notebooks can be re-run. results-stage2RR_reproducable folder can be used to do this - as this folder does not contain output files. 

SETUP INSTRUCTIONS:
1. Install system dependencies
**Linux (Debian/Ubuntu)**:
  sudo apt install cmake default-jdk gsfonts libcurl4-openssl-dev \
    libfontconfig1-dev libfreetype6-dev libglpk-dev libmagick++-dev \
    libssl-dev libx11-dev libxml2-dev libzmq3-dev pandoc \
    libharfbuzz-dev libfribidi-dev libnlopt-dev

**macOS**:
  - Xcode Command Line Tools: `xcode-select --install`
  - A Java Development Kit matching your Mac's architecture (e.g. via 
    https://adoptium.net — select "aarch64" for Apple Silicon)


2. Install quarto (https://quarto.org/docs/get-started/)

3. Download folder in its entirety


4. Create and activate a python environment within the folder using Python3.12 and the requirements.txt file:
python3.12 -m venv disgust_env
source disgust_env/bin/activate
pip install -r requirements.txt

5. Create a .renvignore file in the project root to exclude non-R directories from dependency scanning:
cat > .renvignore << 'EOF'
disgust_env
data/
csvs/
results/
EOF

6. Restore the renv environment: Rscript -e 'renv::restore()' and install reticulate package (Rscript -e 'install.packages("reticulate")') for quarto rendering

7. Render quarto files (e.g., quarto render results/video_ratings/VideoRatings.qmd)
[add order]


Trouble shooting:
If Quarto cannot find R packages when rendering, confirm it is using 
the correct R installation by running `quarto check` and comparing 
the R path shown against `R.home()` run inside R. If they differ, set:
export QUARTO_R=$(Rscript -e 'cat(R.home())'). 

If Quarto reports that a package is unavailable, first run Rscript -e 'renv::status()' to check whether the installed R library is consistent with renv.lock. If packages are out of sync, run Rscript -e 'renv::restore()' before attempting to install packages manually.

NB model fitting/comparisons within the computational modeling section (comp_modeling/modelFitting) were run using a HPC rather than locally rendered quarto files

INSTRUCTIONS FOR REPRODUCIBILITY:
Scripts should be run in the order in which they were created. Steps with asterisk require re-running computational models (high compute power and memory requirement). These steps can be safely missed out because the output files are included in this folder. 
1. power_analysis folder (scripts should be run in order and are labelled A-D) - creates power analysis described in supplement

2. data_cleaning (scripts should be run in order and  labelled 0-1) - creates cleaned datafiles (stored in the ../csvs folder

3. results/videoratings/VideoRatings.qmd - carries out video ratings analyses

4. results/model_agnostic folder (quarto files should be rendered in order and are labelled 1-4)

5. results/comp_modeling/modelFitting (quarto files should be rendered in order)*
makeStandata.R (converts raw data into suitable format)
submitcluster.sh (calls runRandomModel.sh/.r (requires STANfiles); runModel.sh/.r (requires STANfiles); checkModel_extractParams.sh/.r)

6. results/comp_modeling/modelFitting/modelDiagnostics/testingrhatneff.ipynb (quarto files should be rendered in order)*

7. results/comp_modeling/modelFitting/modelDiagnostics/ParamRecov folder* (quarto files should be rendered in order)*
simData.sh (calls simData.r)
runModel.sh (calls runModel.r)

8. results/comp_modeling/modelFitting/modelDiagnostics/PPCs/PPCs.sh (calls PPCs.r)*

9. results/comp_modeling folder (quarto files should be rendered in order are are labeled 1-4) (NB 1_modelCompAndChecks* requires outputs of computational models)


RESULTS IN MANUSCRIPT:
can  be found in their corresponding quarto files (within video_ratings, model_agnostic and comp_modeling folders). E.g., all perseverative error analyses are found in the 1_perseverativeErrors.qmd quarto file within the model_agnostic folder, all stickiness analyses are in 4_stickiness.qmd in the comp_modeling folder. 


LIST OF FOLDERS AND FILES:
Csvs folder:
Contains raw blinded csv files (blinded_data_full.csv and blinded_demography.csv)
Contains the csvs created in the data-cleaning scripts 
(e.g., a full blinded dataframe, a demography dataframe, a dataframe with video-ratings, task dataframes, exclusion details)

Data folder:
Did contain original data (raw jatos files) and unblinding document [not shared due to PID]
data_to_blinded_csv.ipynb - converts data to blinded csv file 


data_cleaning folder:
00_data_checks.ipynb: was used to make rolling exclusions. Look at data from one participant and determine if they passed the exclusion criteria.
0_data_cleaning.ipynb: Takes blinded csv file and makes exclusions using pre-registered exclusion criteria
dataclean_func.py: contains functions for data cleaning
checking_exclusions.ipynb: checks the exclusions did not impact representativeness of the sample (compares actual numbers in each demographic category to target numbers)


power_analysis folder: 
A_power_analysis_script: carries out both liberal and conservative power analyses, saving the outputs into power tables
B_power_analysis_plots: uses the power tables to identify the minimum required sample and plot power curves
C_power_analysis_withinsubjcorr: carries out additional checks for the main power analysis. 
    namely, assessing the effects of a lower within-subject correlation on power.
D_power_analysis_maximalmodels:  carries out additional checks for the main power analysis. 
    namely, assessing the effects of a more maximal model on power.


renv folder:
necessary contents for activating R environment  (allows you to replicate the renv (Rscript -e 'renv::restore()')
Then should install reticulate package (Rscript -e 'install.packages("reticulate")') for quarto rendering)


results folder:
/video_ratings folder
//pvals: contains p-values from key results for plotting
//figures: contains code to create figures and the figure files themselves 
VideoRatings.md: contains all the video-ratings analyses (models A-H specified by the analysis plan)
VideoRatings.qmd: contains quartofile that creates video ratings markdown file

/model_agnostic folder: all the model_agnostic analyses
//figures: contains codes to create figures and figure files themselves
//pvals: contains p-values from key results for plotting
perseverativeErrors.md: all analyses for perseverative error outcome
perseverativeErrors.qmd: quartofile to create perseverativeErrors.md
regressiveErrors.md: all analyses for regressive error outcome
regressiveErrors.qmd: quartofile to create markdown file
winStay.md: all analyses for win-stay outcome
winStay.qmd: quartofile to make markdown file
loseShift.md: all analyses for lose-shift outcome
loseShift.qmd: quartofile to make markdown file


/comp_modeling folder: all the computational modeling analyses
//figures: contains code to create figures and figure files themselves
//pvals: contains p-values from key results for plotting
//csvs: contains csvs (and one rds file) for computational modeling analyses
complete_task_excluded.csv: raw task data (same as used in model agnostic analysis)
1lr_stick1_blk3_allparamsep_params.csv: parameters from winning model
winningModelOutput.csv: parameters from the winning model combined with key task outcomes
sensitivity_winningModelOutput.csv: parameters from the winning model once outliers have been excluded

modelCompAndChecks.md: contains model comparison and checks on the winning model
modelCompAndChecks.qmd: quartofile to create markdown
learningRate.md: contains analyses on learning rate parameter
learningRate.qmd: quartofile to create markdown
inverseTemp.md: contains analyses on inverse temperature parameter
inverseTemp.qmd: quartofile to create markdown
stickiness.md: contains analyses on stickiness parameter
stickiness.qmd:quartofile to create markdown


//modelFitting (within comp_modeling): contains all files necessary to fit data to all models in the model space. NB many files within this folder were run on a high performance computing cluster and cannot easily be re-run locally. 
makeStanData.R: converts task-data to the structure necessary to fit to the stan model
submitcluster.sh: submits all models to the cluster (specifying number of iterations etc.). Calls runModel.sh/.r, runRandomModel.sh/.r, and checkModel_extractParams files
runModel.sh/.r: runs model and saves csv files (not included in this repository due to size)
runRandomModel.sh/.r: same as above but for random model (also extracts log-likelihood)
///STANfiles: contains the stanfile for each model in the model space (all model are adapted from hBayesDM model - prl_rp (Ouden et al. ,2013 ;https://github.com/CCS-Lab/hBayesDM)
///modelOutputs: contains outputs for each model (model fit files and csvs not included here due to memory constraints but other outputs are included)
///checkModel_extractParams: contains a file for each model to extract everything needed from the model (parameters, draws matrix, effective sample size, r-hat, y-pred, log likelihood) 

///modelDiagnostics: contains code for parameter recovery, posterior predictive checks, r-hat checks and effective sample size checks (all run on winning model)
testingrhatneff.ipynb: checks that effective sample size and rhat for each model are good
rhat_neff_df.csv: a table which each model and its minimal effective sample size ratio and maximum r-hat value. 

////ParamRecov (within modelDiagnostics): contains code for parameter recovery in 1lr_stick1_allparamsep model
simData.sh/.r: simulates datapoints for 100 participants using the 1lr_stick1_allparamsep model equations (saves out simData.rds, simParams.rds and stan_data.rds)
runModel.sh/.r: runs model (as before) but using simulated data
/////modelOutputs (within ParamRecov): would contain output for this model (almost empty due to memory constraints)
modelPars_1lr_stick1_blk3_allparamsep.rds (within modelOutputs): contains recovered parameters from parameter recovery

////PPCs (within modelDiagnostics): contains code for posterior predictive checks
PPC.sh/.r: extracts y-pred matrix and combines with actual data
postpred_alltrials_1lr_stick1_blk3_allparamsep.rds: a combination of y-pred medians and IQR, and actual data
y_pred_1lr_stick1_blk3_allparamsep.rds: the y-pred matrix extracted from the winning model [not included due to memory constraints]




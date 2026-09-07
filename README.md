# disgust_reversal_learning-final
<p> Analysis scripts for stage 2 registered report

<p> Contains all scripts for all analyses detailed in the stage 2 registered report. 
<p> (NB therefore, there is some duplication with the stage 1 report)

Note, data is not shared in this repo but can be found on the corresponding OSF page: https://osf.io/f5rmt/files/osfstorage

<br>
<b>Instructions for reproducibility:</b>
<p>1. Download folder in its entirety
<p>2. Install quarto 
<p>3. Create a python environment using Python3.12 and the requirements.txt file
<p>4. Replicate the renv (Rscript -e 'renv::restore()') and install reticulate package (Rscript -e 'install.packages("reticulate")') for quarto rendering
<p>5. Render quarto files (e.g., quarto render VideoRatings.qmd)
<p>NB model fitting/comparisons within the computational modeling section (comp_modeling) are too computationally expensive to be run this way and full details of this process is found within the comp_modeling/modelFitting folder

<br>
<h3>Csvs folder:</h3>
<p> Contained the csvs created in the data-cleaning scripts 
<p> (e.g., a full blinded dataframe, a demography dataframe, a dataframe with video-ratings, task dataframes, exclusion details)

<br>
<h3>Data folder:</h3>
<p> Contained original data (raw jatos files) and unblinding document 
<p> data_to_blinded_csv.ipynb - converts data to blinded csv file 

<br>
<h3>data_cleaning folder:</h3>
<p> 00_data_checks.ipynb: was used to make rolling exclusions. Looks at data from one participant and determine if they passed the exclusion criteria.
<p> 0_data_cleaning.ipynb: Takes blinded csv files and makes exclusions using pre-registered exclusion criteria
<p>dataclean_func.py: contains functions for data cleaning
<p> checking_exclusions.ipynb: checks the exclusions did not affect representativeness of the sample (compares actual numbers in each demographic category to target numbers)


<br>
<h3>power_analysis folder:</h3>
<p> A_power_analysis_script: carries out both liberal and conservative power analyses, saving the outputs into power tables
<p> B_power_analysis_plots: uses the power tables to identify the minimum required sample and plot power curves
<p> C_power_analysis_withinsubjcorr: carries out additional checks for the main power analysis. 
    namely, assessing the effects of a lower within-subject correlation on power.
<p> D_power_analysis_maximalmodels:  carries out additional checks for the main power analysis. 
    namely, assessing the effects of a more maximal model on power.

<br>

<h3>renv folder:</h3>
necessary contents for activating R environment  (allows you to replicate the renv (Rscript -e 'renv::restore()'))
<p>Then should install reticulate package (Rscript -e 'install.packages("reticulate")') for quarto rendering)

<br>
<h3><b>results folder:</b></h3>
<h4>/video_ratings folder</h4>
<p> //pvals: contains p-values from key results for plotting
 <br>
  <br>
<p> //figures: contains code to create figures and the figure files themselves 
  <br>
  <br>
<p> VideoRatings.md: contains all the video-ratings analyses (models A-H specified by the analysis plan)
<p> VideoRatings.qmd: contains quartofile that creates videoratings markdownfile

<br>
<br>
<h4>/model_agnostic folder</h4>
<p> //figures: contains codes to create figures and figure files themselves
  <br>
  <br>
<p> //pvals: contains p-values from key results for plotting
  <br>
  <br>
<p> perseverativeErrors.md: all analyses for perseverative error outcome
<p> perseverativeErrors.qmd: quartofile to create perseverativeErrors.md
<p> regressiveErrors.md: all analyses for regressive error outcome
<p> regressiveErrors.qmd: quartofile to create markdown file
<p> winStay.md: all analyses for win-stay outcome
<p> winStay.qmd: quartofile to make markdown file
<p> loseShift.md: all analyses for lose-shift outcome
<p> loseShift.qmd: quartofile to make markdown file

<br>
<br>
<h4>/comp_modeling folder</h4>
<p> //figures: contains code to create figures and figure files themselves
  <br>
  <br>
<p> //pvals: contains p-values from key results for plotting
  <br>
  <br>
<p> modelCompAndChecks.md: contains model comparison and checks on the winning model
<p> modelCompAndChecks.qmd: quartofile to create markdown
<p> inverseTemp.md: contains analyses on inverse temperature parameter
<p> inverseTemp.qmd: quartofile to create markdown
<p> learningRate.md: contains analyses on learning rate parameter
<p> learningRate.qmd: quartofile to create markdown
<p> stickiness.md: contains analyses on stickiness parameter
<p> stickiness.qmd:quartofile to create markdown

<br>
 <br>
<p> //csvs: contains csvs (and one rds file) for computational modeling analyses
<p> complete_task_excluded.csv: raw task data (same as used in task agnostic analysis)
<p>1lr_stick1_blk3_allparamsep_params.csv: parameters from winning model
<p> winningModelOutput.csv: parameters from the winning model combined with key task outcomes
<p> sensitivity_winningModelOutput.csv: parameters from the winning model once outliers have been excluded

<br>
 <br>
<p> //modelFitting: contains all files necessary to fit data to all models in the model space. NB many files within this folder were run on a high performance computing cluster and cannot easily be re-run locally. 
<p> makeStanData.R: converts task-data to the structure necessary to fit to the stan model
<p> submitcluster.sh: submits all models to the cluster (specifying number of iterations etc.). Calls runModel.sh/.r, runRandomModel.sh/.r, and checkModel_extractParams files
<p> runModel.sh/.r: runs model and saves csv files (not included in this repository due to size)
<p> runRandomModel.sh/.r: same as above but for random model (also extracts log-likelihood)
  <br>
  <br>
<p> ///STANfiles: contains the stanfile for each model in the model space (all model are adapted from hBayesDM model - prl_rp (Ouden et al. ,2013 ;https://github.com/CCS-Lab/hBayesDM)
  <br>
  <br>
<p> ///modelOutputs: contains outputs for each model (fit files and csvs not included here due to memory constraints)
  <br>
  <br>
<p> ///checkModel_extractParams: contains a file for each model to extract everything needed from the model (parameters, draws matrix, effective sample size, r-hat, y-pred, log likelihood) 
 <br>

<br>
<p> ///modelDiagnostics: contains code for parameter recovery, posterior predictive checks, r-hat checks and effective sample size checks (all run on winning model)
<p> testingrhatneff.ipynb: checks that effective sample size and rhat for each model are good
<p> rhat_neff_df.csv: a table which each model and its minimal effective sample size ratio and maximum r-hat value. 

<br>
 <br>
<p> ////ParamRecov: contains code for parameter recovery in 1lr_stick1_allparamsep model
<p> simData.sh/.r: simulates datapoints for 100 participants using the 1lr_stick1_allparamsep model equations (saves out simData.rds, simParams.rds and stan_data.rds)
<p> runModel.sh/.r: runs model (as before) but using simulated data
  <br>
  <br>
<p> /////modelOutputs: would contain output for this model (almost empty due to memory constraints)
<p> modelPars_1lr_stick1_blk3_allparamsep.rds: contains recovered parameters from parameter recovery

<br>
 <br>
<p> ////PPCs: contains code for posterior predictive checks
<p> PPC.sh/.r: extracts y-pred matrix and combines with actual data
<p> postpred_alltrials_1lr_stick1_blk3_allparamsep.rds: a combination of y-pred medians and IQR, and actual data
<p> y_pred_1lr_stick1_blk3_allparamsep.rds: the y-pred matrix extracted from the winning model [not included due to memory constraints]


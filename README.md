# disgust_reversal_learning-final
 Analysis scripts for stage 2 registered report

 Contains all analysis scripts for model-agnostic analyses in the Stage 2 Registered Report, as well as the power analysis (which is unchanged from the stage 1 Registered Report).
 All changes from the stage 1 versions are minimal (debugging, adding real data, creating markdown versions of scripts etc.) and can be seen in the version history.

<br>
<b><h3>The clearest way to see all analysis is in the markdown scripts found in the Full_results_markdown folder:</h3></b>

These files contain all analyses reported in the paper (which are duplicated in the original notebooks found in the other folders of this repository) in a clearer more intelligible way (e.g., with markdown explanations and confusing extra steps)
<br>
<br>

VideoRatings.md: contains models A-H as specified in the analysis plan (all the video rating checks)

dataVisualisation.md: contains all the graphs for the model-agnostic analysis

exploratory_analyses.md: contains all the exploratory model-agnostic analyses run (alongside explanations of them in markdown)

model1_perseverativeErrors.md: contains all analyses regarding the perseverative error hypothesis (original model, model with video ratings and a sensitivity analysis)

model2_regressive_errors.md: contains all analyses regarding the regressive error hypothesis (original model, model with video ratings and a sensitivity analysis)

model3_winStay.md: contains all analyses regarding the win-stay hypothesis (original model, model with video ratings and a sensitivity analysis)

model4_loseShift.md: contains all analyses regarding the lose-shift hypothesis (original model, model with video ratings and a sensitivity analysis)

/figures contains high quality jpegs for each figure created in dataVisualisation.md

(NB all notebooks have .qmd counterparts which are the quarto files used to create these markdown files)



<br>
<br>
<h3>Original notebooks (i.e., those written for the stage 1 registered report) are also found in this repository, duplicating the results shown in the markdown files</h3>
<br>
 <b>Data checks/cleaning - all found within the data_cleaning folder:</b>
 
 00_data_checks.ipynb: used to make rolling exclusions. Look at data from one participant and determine if they passed the exclusion criteria (1 version for 1 example participant is included here).
 
 0_data_cleaning.ipynb: Used to create useable dataframes for subsequent analysis. Excludes using pre-registered exclusion criteria and calculates key task-outcomes.
 



<br>
<br>
 <b>Planned model-agnostic analyses - all found within the model_agnostic_analyses folder:</b>
 
 1_video_ratings_analyses.ipynb: Completes models A-H as specified by the analysis plan. Full analysis of the video-rating task.
 
 2_task_model_agnostic_analysis.ipynb: Completes models 1-6. Full model-agnostic analysis of the reversal learning task
 
 2_task_model_agnostic_analysis_nooutliers.ipynb: exactly the same as above, but excludes all outliers (>1.5IQR outside of IQR)
 
 /generalized_mixed_effects_models: contains notebooks with the model selection and results for each analysis that required a generalized mixed effects model (these were run in R, hence required separate notebooks). Contains both models for main analysis and sensitivity analysis.


   
<br>
 3_exploratory_analyses.ipynb: contains unplanned exploratory analyses regarding the model-agnostic outcomes

 3_exploratory_analyses_nooutiers.ipynb: exactly the same as above, but excludes all outliers (>1.5IQR outside of IQR)

 /exploratory_generalized_mixed_effects_models: contains notebooks with model selection and results for each exploratory analysis that required a generalized mixed effects model (these were run in R, hence required separate notebooks). Contains both models for main analysis and sensitivity analysis.




<br>
<br>
<b>Power analysis (unchanged from stage 1 Registered Report) - all found witin the power_analysis folder:</b>

A_power_analysis_script: carries out both liberal and conservative power analyses, saving the outputs into power tables

B_power_analysis_plots: uses the power tables to identify the minimum required sample and plot power curves

C_power_analysis_withinsubjcorr: carries out additional checks for the main power analysis. 
    namely, assessing the effects of a lower within-subject correlation on power.

D_power_analysis_maximalmodels:  carries out additional checks for the main power analysis. 
    namely, assessing the effects of a more maximal model on power.

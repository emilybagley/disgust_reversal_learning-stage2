# disgust_reversal_learning-final
 Analysis scripts for stage 2 registered report

 Contains all analysis scripts for model-agnostic analyses in the Stage 2 Registered Report, as well as the power analysis (which is unchanged from the stage 1 Registered Report).
 All changes from the stage 1 versions are minimal (debugging, adding real data etc.) and can be seen in the version history.



 Data checks/cleaning:
 
 00_data_checks: used to make rolling exclusions. Look at data from one participant and determine if they passed the exclusion criteria (1 version for 1 example participant is included here).
 
 0_data_clearning: Used to create useable dataframes for subsequent analysis. Excludes using pre-registered exclusion criteria and calculates key task-outcomes.
 checking_exclusions: checks that participant exclusions in data_cleaning notebook has not hurt the representativeness of the final sample.


 
 Planned model-agnostic analyses
 
 1_video_ratings_analyses: Completes models A-H as specified by the analysis plan. Full analysis of the video-rating task.
 
 2_task_model_agnostic_analysis: Completes models 1-6. Full model-agnostic analysis of the reversal learning task
 
 2_task_model_agnostic_analysis_nooutliers: exactly the same as above, but excludes all outliers (>1.5IQR outside of IQR)



Power analysis (unchanged from stage 1 Registered Report)

A_power_analysis_script: carries out both liberal and conservative power analyses, saving the outputs into power tables

B_power_analysis_plots: uses the power tables to identify the minimum required sample and plot power curves

C_power_analysis_withinsubjcorr: carries out additional checks for the main power analysis. 
    namely, assessing the effects of a lower within-subject correlation on power.

D_power_analysis_maximalmodels:  carries out additional checks for the main power analysis. 
    namely, assessing the effects of a more maximal model on power.

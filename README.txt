Contains example analysis scripts for all planned model-agnostic analyses in the Registered Report, as well as the power analysis (included in the stage 1 Registered Report).

Data checks/cleaning:
00_data_checks: will be used to make rolling exclusions. Look at data from one participant and determine if they passed the exclusion criteria.
0_data_clearning: Will be used to transform files downloaded from jatos (used to host the online experiment) to useable dataframes for use in subsequent analyses


Planned model-agnostic analyses
1_video_ratings_analyses: Completes models A-H as specified by the analysis plan. Full analysis of the video-rating task.
2_task_model_agnostic_analysis: Completes models 1-6. Full model-agnostic analysis of the reversal learning task
2_task_model_agnostic_analysis_nooutliers: exactly the same as above, but excludes all outliers (>1.5IQR outside of IQR)

Computational modelling analyses (specified in the analysis plan) will be completed using the hBayesDM package which is a freely available software package for hierarchical Bayesian modelling (https://github.com/CCS-Lab/hBayesDM). On completion of the stage 2 manuscript, full modelling code will also be added to this OSF folder (in a way that is fully understandable without accessing the hBayesDM GitHub repository). 


Power analysis
A_power_analysis_script: carries out both liberal and conservative power analyses, saving the outputs into power tables
B_power_analysis_plots: uses the power tables to identify the minimum required sample and plot power curves
C_power_analysis_withinsubjcorr: carries out additional checks for the main power analysis. 
    namely, assessing the effects of a lower within-subject correlation on power.
D_power_analysis_maximalmodels:  carries out additional checks for the main power analysis. 
    namely, assessing the effects of a more maximal model on power.


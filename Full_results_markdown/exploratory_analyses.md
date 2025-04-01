# Exploratory analyses


<p>

This file contains all of the exploratory analyses run - these are
analyses not specified in the stage 1 registered report, but are run to
help understand our model-agnostic results better
</p>

<br>
<p>

This includes assessing whether:
</p>

<p>

- Overall task performance (index by percentage of correct trials)
  differs across the three blocks/feedback types
- The difference between self-reported points and disgust ratings
  explains the difference between points and disgust learning
- there is anything noteable about the outliers on the perseverative
  error outcome - explaining why they drive effects
- Video ratings for *all* videos (not just the ones used in the reversal
  learning task) show similar patterns to those selected for use in the
  reversal learning task.

</p>

<h3>

Firstly, we load in the relevant packages for both R and python:
</h3>

<details class="code-fold">
<summary>Code</summary>

``` r
library(tidyverse, quietly=TRUE)
library(lme4)
library(emmeans)
library(DHARMa)
library('readxl')
library('xlsx')

task_summary <- read.csv("U:/Documents/Disgust learning project/github/disgust_reversal_learning-final/csvs/dem_vids_task_excluded.csv")
chosen_stim_df <- read.csv('U:/Documents/Disgust learning project/github/disgust_reversal_learning-final/csvs/chosen_stim_excluded.csv')
sensitivity_df <- read.csv('U:/Documents/Disgust learning project/github/disgust_reversal_learning-final/csvs/sensitivity_df.csv')
```

</details>

<details class="code-fold">
<summary>Code</summary>

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import jsonlines
from functools import reduce
import statistics
import scipy.stats
import seaborn as sns
import math
import os
import json
import ast
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pingouin as pg
import warnings
from scipy.stats import ttest_rel
#from statannotations.Annotator import Annotator
from scipy.stats import skew
from statsmodels.stats.diagnostic import het_white
from sklearn.preprocessing import PowerTransformer
import statannot
from scipy.stats import ttest_ind
import itertools

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.copy_on_write = True

task_summary=pd.read_csv("U:/Documents/Disgust learning project/github/disgust_reversal_learning-final/csvs/dem_vids_task_excluded.csv")
chosen_stim_df=pd.read_csv('U:/Documents/Disgust learning project/github/disgust_reversal_learning-final/csvs/chosen_stim_excluded.csv')
sensitivity_df = pd.read_csv('U:/Documents/Disgust learning project/github/disgust_reversal_learning-final/csvs/sensitivity_df.csv')
vid_ratings_df=pd.read_csv('U:/Documents/Disgust learning project/github/disgust_reversal_learning-final/csvs/ratings_df.csv')

def bayes_factor(df, dependent_var, condition_1_name, condition_2_name):
    df=df[(df.block_type==condition_1_name)| (df.block_type==condition_2_name)][[dependent_var, 'block_type', 'participant_no']]
    df.dropna(inplace=True)
    df=df.pivot(index='participant_no', columns='block_type', values=dependent_var).reset_index()
    ttest=pg.ttest(df[condition_1_name], df[condition_2_name], paired=True)
    bf_null=1/float(ttest.BF10)
    return bf_null
```

</details>

<h3>

<b>Exploratory analysis 1: </b>
</h3>

<p>

Hypothesis testing analyses showed a difference in perseverative error
rate and lose-shift probability between fear and points learning. To
better understand this change, we tested whether this difference is
mirrored by a difference in overall task performance (indexed by
percentage of trials where participants were correct)
</p>

<br>
<p>

Firstly, check for skewness of the variable
</p>

``` python
sns.histplot(data=task_summary, x="percentage_correct") 
print('Percentage correct: '+str(skew(task_summary.percentage_correct)))
```

    Percentage correct: -0.49987154559115393

![](exploratory_analyses_files/figure-commonmark/Skewness%20percentage%20correct-1.jpeg)

<br>
<p>

And then run a mixed effects model to assess whether it differs by
block-type
</p>

<p>

The regular mixed effects model did not converge, so a generalized mixed
effects model is run in its place
</p>

<p>

The winning model (as indexed by BIC) had:
<p>

- a gamma probability function
- an identity link function
- no covariates
- random intercepts for feedback videos

``` r
data<-task_summary

generalized_model <- glmer(percentage_correct ~ block_type +(1|participant_no) + (1|feedback_details), data=data, family=Gamma(link="identity"))
summary(generalized_model)
```

    Generalized linear mixed model fit by maximum likelihood (Laplace
      Approximation) [glmerMod]
     Family: Gamma  ( identity )
    Formula: percentage_correct ~ block_type + (1 | participant_no) + (1 |  
        feedback_details)
       Data: data

         AIC      BIC   logLik deviance df.resid 
     -2612.5  -2583.0   1312.3  -2624.5     1014 

    Scaled residuals: 
        Min      1Q  Median      3Q     Max 
    -3.2556 -0.5020  0.0390  0.5401  2.7870 

    Random effects:
     Groups           Name        Variance  Std.Dev.
     participant_no   (Intercept) 2.580e-03 0.050795
     feedback_details (Intercept) 9.027e-06 0.003005
     Residual                     9.488e-03 0.097408
    Number of obs: 1020, groups:  participant_no, 340; feedback_details, 11

    Fixed effects:
                     Estimate Std. Error t value Pr(>|z|)    
    (Intercept)      0.658173   0.006157 106.905   <2e-16 ***
    block_typeFear   0.005246   0.005721   0.917    0.359    
    block_typePoints 0.010104   0.009967   1.014    0.311    
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Correlation of Fixed Effects:
                (Intr) blck_F
    block_typFr -0.316       
    blck_typPnt -0.077  0.591

<p>

The results of the outlier-free sensitivity analysis model is similar
(although it is closer to significance)
</p>

``` r
data<-sensitivity_df

generalized_model <- glmer(percentage_correct ~ block_type +(1|participant_no), data=data, family=Gamma(link="identity"))
summary(generalized_model)
```

    Generalized linear mixed model fit by maximum likelihood (Laplace
      Approximation) [glmerMod]
     Family: Gamma  ( identity )
    Formula: percentage_correct ~ block_type + (1 | participant_no)
       Data: data

         AIC      BIC   logLik deviance df.resid 
     -2655.1  -2630.5   1332.5  -2665.1     1006 

    Scaled residuals: 
        Min      1Q  Median      3Q     Max 
    -3.0023 -0.5192  0.0357  0.5614  2.8656 

    Random effects:
     Groups         Name        Variance Std.Dev.
     participant_no (Intercept) 0.002378 0.04876 
     Residual                   0.008815 0.09389 
    Number of obs: 1011, groups:  participant_no, 340

    Fixed effects:
                     Estimate Std. Error t value Pr(>|z|)    
    (Intercept)      0.659653   0.005420 121.698   <2e-16 ***
    block_typeFear   0.002027   0.004166   0.486   0.6266    
    block_typePoints 0.008153   0.004198   1.942   0.0521 .  
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Correlation of Fixed Effects:
                (Intr) blck_F
    block_typFr -0.387       
    blck_typPnt -0.383  0.497

<p>

Bayes factors also show support for the null hypothesis (no difference
between points and disgust):
</p>

<b>With outliers:</b>

``` python
print(bayes_factor(task_summary, 'percentage_correct', 'Disgust', 'Points'))
```

    3.546099290780142

<p>

This means that there is <b>moderate</b> evidence for the null
</p>

<br> <b>Without outliers</b>:

``` python
print(bayes_factor(sensitivity_df, 'percentage_correct', 'Disgust', 'Points'))
```

    4.424778761061947

<p>

This means that there is <b>moderate</b> evidence for the null
</p>

<br> <br>
<h3>

<b>Exploratory analysis 2: </b> assessing the effect of video ratings on
lose-shift results
</h3>

<p>

In the planned analysis, we assessed whether all hypothesis testing
models were affected by differences between <b>fear and disgust</b>
detected in the video rating analyses
</p>

<p>

These analyses were planned in order to assess whether any differences
between fear and disgust were driven by any differences in valence and
arousal
</p>

<p>

However, since no hypothesis testing model found a difference between
fear and disgust, instead finding a difference bewteen <b>points and
disgust</b> - arguably these video ratings were not the most relevant
ones to include.
</p>

<p>

Instead, it makes more sense to test whether the difference between
<b>points and disgust</b> is driven by the differences between <b>points
and disgust</b> found in the video rating analyses.
</p>

<p>

Namely, whether the difference between disgust and points learning
(indexed by lose-shift probability) is driven by:
<p>

- the difference in valence found between points and disgust feedback
- the difference in fear rating found between points and disgust
- the difference in disgust rating found between points and disgust

</p>

<p>

NB lose-shift is chosen for this analysis as it is the most robust of
our findings (relative to the perseverative error finding) so allows
more firm conclusions

<b>Due to an error in the task code, we don’t have points-ratings values
for some participants. Given that these analyses pertain to
points-ratings, we will first exclude all these participants.</b>
</p>

``` python
participants_to_remove=list(set(chosen_stim_df[chosen_stim_df.unpleasant_1.isna()].participant_no))
chosen_stim_df_short=chosen_stim_df[~chosen_stim_df['participant_no'].isin(participants_to_remove)]
task_summary_short=task_summary[~task_summary['participant_no'].isin(participants_to_remove)]
```

<p>

NB we ran a sanity check to show that the original ‘lose-shift’ result
remains in this slightly smaller sample
</p>

``` python
formula = 'lose_shift ~ block_type + prolific_age'
data=task_summary_short
results=smf.mixedlm(formula, data=data, groups=data['participant_no'], missing='drop').fit(reml=False)
print(results.summary())
```

                Mixed Linear Model Regression Results
    ==============================================================
    Model:              MixedLM   Dependent Variable:   lose_shift
    No. Observations:   774       Method:               ML        
    No. Groups:         258       Scale:                0.0086    
    Min. group size:    3         Log-Likelihood:       506.1864  
    Max. group size:    3         Converged:            Yes       
    Mean group size:    3.0                                       
    --------------------------------------------------------------
                         Coef. Std.Err.   z    P>|z| [0.025 0.975]
    --------------------------------------------------------------
    Intercept            0.584    0.027 21.616 0.000  0.531  0.637
    block_type[T.Fear]   0.008    0.008  0.991 0.322 -0.008  0.024
    block_type[T.Points] 0.024    0.008  2.935 0.003  0.008  0.040
    prolific_age         0.001    0.001  2.273 0.023  0.000  0.002
    Group Var            0.015    0.021                           
    ==============================================================

<p>

Also create a data-frame with points vs disgust ratings difference
scores (as had originally done for disgust vs fear)
</p>

``` python
stim_ratings_covariates=pd.DataFrame()
block_feedback=pd.DataFrame()
for participant_no in set(chosen_stim_df_short.participant_no):
    participant_df=chosen_stim_df_short[chosen_stim_df_short.participant_no==participant_no]
    disgust=participant_df[participant_df.trial_type=="disgust"]
    points=participant_df[participant_df.trial_type=="points"]
    valence_diff=int(points.unpleasant_1)-int(disgust.unpleasant_1)
    disgust_diff=int(points.disgusting_1)-int(disgust.disgusting_1)
    fear_diff=int(points.frightening_1)-int(disgust.frightening_1)

    
    row=pd.DataFrame({
        'participant_no': [participant_no],
        'points_valence_diff': [valence_diff],
        'points_disgust_diff': [disgust_diff],
        'points_fear_diff': [fear_diff]
    })
    stim_ratings_covariates=pd.concat([stim_ratings_covariates, row])
data=pd.merge(task_summary_short, stim_ratings_covariates, on='participant_no', how='outer')
```

<p>

In a series of mixed effects models we showed that the lose-shift result
was <b>not</b> moderated by differences in video ratings between disgust
and points
</p>

<b>Valence difference between points and disgust</b>

``` python
formula = 'lose_shift ~ block_type + prolific_age + points_valence_diff'
results=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
print(results.summary())
```

                Mixed Linear Model Regression Results
    ==============================================================
    Model:              MixedLM   Dependent Variable:   lose_shift
    No. Observations:   774       Method:               ML        
    No. Groups:         258       Scale:                0.0086    
    Min. group size:    3         Log-Likelihood:       506.5363  
    Max. group size:    3         Converged:            Yes       
    Mean group size:    3.0                                       
    --------------------------------------------------------------
                         Coef. Std.Err.   z    P>|z| [0.025 0.975]
    --------------------------------------------------------------
    Intercept            0.584    0.027 21.652 0.000  0.531  0.637
    block_type[T.Fear]   0.008    0.008  0.991 0.322 -0.008  0.024
    block_type[T.Points] 0.024    0.008  2.935 0.003  0.008  0.040
    prolific_age         0.001    0.001  2.354 0.019  0.000  0.002
    points_valence_diff  0.003    0.003  0.837 0.402 -0.004  0.009
    Group Var            0.015    0.021                           
    ==============================================================

<b>Disgust difference between points and disgust</b>

``` python
formula = 'lose_shift ~ block_type + prolific_age + points_disgust_diff'
results=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
print(results.summary())
```

                 Mixed Linear Model Regression Results
    ===============================================================
    Model:               MixedLM   Dependent Variable:   lose_shift
    No. Observations:    774       Method:               ML        
    No. Groups:          258       Scale:                0.0086    
    Min. group size:     3         Log-Likelihood:       506.2876  
    Max. group size:     3         Converged:            Yes       
    Mean group size:     3.0                                       
    ---------------------------------------------------------------
                         Coef.  Std.Err.   z    P>|z| [0.025 0.975]
    ---------------------------------------------------------------
    Intercept             0.579    0.029 19.954 0.000  0.522  0.636
    block_type[T.Fear]    0.008    0.008  0.991 0.322 -0.008  0.024
    block_type[T.Points]  0.024    0.008  2.935 0.003  0.008  0.040
    prolific_age          0.001    0.001  2.182 0.029  0.000  0.002
    points_disgust_diff  -0.001    0.003 -0.450 0.653 -0.008  0.005
    Group Var             0.015    0.021                           
    ===============================================================

<b>Fear difference between points and disgust</b>

``` python
formula = 'lose_shift ~ block_type + prolific_age + points_fear_diff'
results=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
print(results.summary())
```

                Mixed Linear Model Regression Results
    ==============================================================
    Model:              MixedLM   Dependent Variable:   lose_shift
    No. Observations:   774       Method:               ML        
    No. Groups:         258       Scale:                0.0086    
    Min. group size:    3         Log-Likelihood:       506.2015  
    Max. group size:    3         Converged:            Yes       
    Mean group size:    3.0                                       
    --------------------------------------------------------------
                         Coef. Std.Err.   z    P>|z| [0.025 0.975]
    --------------------------------------------------------------
    Intercept            0.583    0.028 20.893 0.000  0.528  0.638
    block_type[T.Fear]   0.008    0.008  0.991 0.322 -0.008  0.024
    block_type[T.Points] 0.024    0.008  2.935 0.003  0.008  0.040
    prolific_age         0.001    0.001  2.258 0.024  0.000  0.002
    points_fear_diff     0.001    0.004  0.174 0.862 -0.007  0.008
    Group Var            0.015    0.021                           
    ==============================================================

<br>
<h3>

<b>Exploratory analysis 3: </b> assessing the nature of outliers in the
perseverative error outcome
</h3>

<p>

The perseverative error outcome in the hypothesis testing model seemed
to be quite dependent on outliers. Therefore, here we assess the nature
of those outliers:
<p>

a- to determine whether they are ‘true’ outliers (i.e., due to
inattention etc.) resulting in alterations in task performance across
all metrics
</p>

<p>

b- to determine whether these outliers performed differently on the
video rating task (e.g., were they more disgusted, leading to their
altered task performance?)
<p>

c- to determine whether these outliers differed from the general sample
in terms of psychiatric diagnosis <br>
<p>

Points a and b were assessed using a series of histograms which can be
seen in the datavisualisation markdown file. <b> Whether the outliers
differed from the general sample in terms of psychiatric diagnosis is
assessed here using a chi-squared test.</b>
</p>

<p>

Firstly, we look at the raw proprtions in the outliers and no outliers
dataframes
</p>

``` python
#create outliers dataframe
Q1 = task_summary['mean_perseverative_er'].quantile(0.25)
Q3 = task_summary['mean_perseverative_er'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1- 1.5 *  IQR
upper_bound = Q3 + 1.5 *  IQR
if lower_bound < min(task_summary.mean_perseverative_er):
    lower_bound = min(task_summary.mean_perseverative_er)
outliers=task_summary[(task_summary['mean_perseverative_er']<lower_bound) | (task_summary['mean_perseverative_er']>upper_bound )]

outliers_dem=outliers[['participant_no', 'cleaned_diagnosis', 'prolific_age', 'prolific_sex']].drop_duplicates()
no_outliers_dem=task_summary[~task_summary['participant_no'].isin(list(outliers_dem.participant_no))][['participant_no', 'cleaned_diagnosis', 'prolific_age', 'prolific_sex']].drop_duplicates()


print(outliers_dem['cleaned_diagnosis'].value_counts(normalize=True))
```

    cleaned_diagnosis
    No     0.677419
    Yes    0.322581
    Name: proportion, dtype: float64

``` python
print(no_outliers_dem['cleaned_diagnosis'].value_counts(normalize=True))
```

    cleaned_diagnosis
    No     0.702265
    Yes    0.297735
    Name: proportion, dtype: float64

<p>

There is a small absolute difference in proportions (with the outliers
have more people with a diagnosis)- use a chi-squared test to assess
whether this difference is significant.
</p>

``` python
outliers_MH=outliers_dem['cleaned_diagnosis'].value_counts()['Yes']
outliers_noMH=outliers_dem['cleaned_diagnosis'].value_counts()['No']
no_outliers_MH=no_outliers_dem['cleaned_diagnosis'].value_counts()['Yes']
no_outliers_noMH=no_outliers_dem['cleaned_diagnosis'].value_counts()['No']

observed=np.array([[outliers_MH, outliers_noMH], 
                [no_outliers_MH, no_outliers_noMH]])
chi2, p, dof, expected = stats.chi2_contingency(observed)
print("chi-squared value: "+str(chi2)), print("p value: "+str(p))
```

    chi-squared value: 0.006760821042061202
    p value: 0.9344684202258998
    (None, None)

<p>

Results of this test show that there is no significant different in
diagnosis between the two samples.
</p>

<br>
<h3>

<b>Exploratory analysis 4:</b> probing whether effect of feedback-type
on perseverative error rate and lose-shift probability is better
explained by a difference between emotional (fear/disgust) and
non-emotional (points) learning, or between disgust-based and other
types (fear and points) of learning
</h3>

<p>

The main hypothesis testing analyses found a difference in learning
(indexed by perseverative errors and lose-shift probability) between the
points/loss-based feedback and disgust feedback.
</p>

<p>

However, no difference was found between <b>either</b> fear and points
OR fear and disgust. This makes interpretation difficult as we cannot
determine whether the result is better explained by a difference in
learning between the two emotional conditions and points-based learning
OR a distinct feature of disgust learning
</p>

<p>

To assess this, we run two competing models for both hypothesis tests:
<p>

- One assessing the presence of a difference between emotional learning
  (combining the fear and disgust block) and non-emotional learning (the
  points block)
- Another assessing the presence of a difference between disgust-based
  learning and learning which is not about digsust (combining the fear
  and points blocks)
  <p>

  We will use a) the presence/absence of significant results and b) the
  model fit (as indexed by BIC - as before) to guide interpretation of
  these competing models
  </p>

<p>

Firstly, add columns to the data-frame splitting conditions into
‘disgust or not’ (disgust vs fear and points) and ‘emotion or not’
(points vs disgust and fear)
</p>

<p>

Do in R and Python
</p>

``` r
task_summary <- task_summary %>%
  mutate(
    disgustOrNot = ifelse(block_type == "Disgust", "Disgust", "Not"),
    emotionOrNot = ifelse(block_type == "Points", "Not", "Emotion")
  )
```

``` python
task_summary.loc[task_summary['block_type']=='Disgust', 'disgustOrNot']='Disgust'
task_summary.loc[task_summary['block_type']!='Disgust', 'disgustOrNot']='Not'
task_summary.loc[task_summary['block_type']=='Points', 'emotionOrNot']='Not'
task_summary.loc[task_summary['block_type']!='Points', 'emotionOrNot']='Emotion'
```

<br>
<p>

<b>We will start with the perseverative error outcome</b>
</p>

<p>

Run the hypothesis test for ‘disgust or not’ (using the same model
specification as used for the hypothesis testing analysis).
</p>

<p>

In this case, a generalized mixed effects model with:
</p>

<p>

- Gamma probability distribution and inverse link function
- no additional random effects or slopes
- no additional covariates

</p>

``` r
task_summary$pos_perseverative_er <- task_summary$mean_perseverative_er + 0.01 ##+0.01 as all values must be positive (i.e., can't have 0s)
disgustOrNot <- glmer(pos_perseverative_er ~ disgustOrNot + (1|participant_no), data=task_summary, family=Gamma(link="inverse"))
summary(disgustOrNot)
```

    Generalized linear mixed model fit by maximum likelihood (Laplace
      Approximation) [glmerMod]
     Family: Gamma  ( inverse )
    Formula: pos_perseverative_er ~ disgustOrNot + (1 | participant_no)
       Data: task_summary

         AIC      BIC   logLik deviance df.resid 
      1585.7   1605.4   -788.9   1577.7     1016 

    Scaled residuals: 
        Min      1Q  Median      3Q     Max 
    -1.2631 -0.7119 -0.2032  0.5198  3.9830 

    Random effects:
     Groups         Name        Variance Std.Dev.
     participant_no (Intercept) 0.1445   0.3801  
     Residual                   0.6149   0.7841  
    Number of obs: 1020, groups:  participant_no, 340

    Fixed effects:
                    Estimate Std. Error t value Pr(>|z|)    
    (Intercept)      1.32188    0.07276  18.167   <2e-16 ***
    disgustOrNotNot  0.16036    0.06799   2.358   0.0184 *  
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Correlation of Fixed Effects:
                (Intr)
    dsgstOrNtNt -0.597

<p>

And then again for ‘emotion or ’not’:
<p>

``` r
task_summary$pos_perseverative_er <- task_summary$mean_perseverative_er + 0.01 ##+0.01 as all values must be positive (i.e., can't have 0s)
emotionOrNot <- glmer(pos_perseverative_er ~ emotionOrNot + (1|participant_no), data=task_summary, family=Gamma(link="inverse"))
summary(emotionOrNot)
```

    Generalized linear mixed model fit by maximum likelihood (Laplace
      Approximation) [glmerMod]
     Family: Gamma  ( inverse )
    Formula: pos_perseverative_er ~ emotionOrNot + (1 | participant_no)
       Data: task_summary

         AIC      BIC   logLik deviance df.resid 
      1588.2   1607.9   -790.1   1580.2     1016 

    Scaled residuals: 
        Min      1Q  Median      3Q     Max 
    -1.2593 -0.7077 -0.1818  0.4985  3.8923 

    Random effects:
     Groups         Name        Variance Std.Dev.
     participant_no (Intercept) 0.1460   0.3821  
     Residual                   0.6178   0.7860  
    Number of obs: 1020, groups:  participant_no, 340

    Fixed effects:
                    Estimate Std. Error t value Pr(>|z|)    
    (Intercept)      1.38443    0.06292  22.001   <2e-16 ***
    emotionOrNotNot  0.12797    0.07535   1.698   0.0894 .  
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Correlation of Fixed Effects:
                (Intr)
    emotnOrNtNt -0.371

<p>

Only the ‘disgust or not’ model finds a significant effect, suggesting
that the difference between disgust and other types of learning (both
fear and points learning) is the key driver of effects
<p>

<p>

BIC values also show that this model is the better fitting model,
further supporting the hypothesis
</p>

``` r
bic_values <- c(
    BIC(disgustOrNot),
    BIC(emotionOrNot)
)
model_names <- c("Disgust or not", "Emotion or not")
bic_df <- data.frame(Model = model_names, BIC = bic_values)
bic_df <- bic_df[order(bic_df$BIC), ]

print(bic_df)
```

               Model      BIC
    1 Disgust or not 1605.429
    2 Emotion or not 1607.863

<p>

Overall, this shows that the difference in perseverative error
identified in the planned hypothesis testing analyses is <b>better
explained</b> by a uniqueness of disgust-based learning (rather than a
more general difference between emotional and non-emotional learning
conditions)
</p>

<br>
<p>

<b>Now, we will run the same tests using for the lose-shift outcome</b>
</p>

<p>

Run the hypothesis test for ‘disgust or not’ (using the same model
specification as used for the hypothesis testing analysis).
</p>

``` python
data=task_summary
formula = 'lose_shift ~ disgustOrNot + prolific_age'
disgustOrNot=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
print(disgustOrNot.summary())
```

                Mixed Linear Model Regression Results
    =============================================================
    Model:               MixedLM  Dependent Variable:  lose_shift
    No. Observations:    1020     Method:              ML        
    No. Groups:          340      Scale:               0.0089    
    Min. group size:     3        Log-Likelihood:      662.4706  
    Max. group size:     3        Converged:           Yes       
    Mean group size:     3.0                                     
    -------------------------------------------------------------
                        Coef. Std.Err.   z    P>|z| [0.025 0.975]
    -------------------------------------------------------------
    Intercept           0.578    0.023 25.519 0.000  0.533  0.622
    disgustOrNot[T.Not] 0.019    0.006  2.996 0.003  0.007  0.031
    prolific_age        0.001    0.000  2.736 0.006  0.000  0.002
    Group Var           0.014    0.017                           
    =============================================================

<p>

And then again for ‘emotion or not’
</p>

``` python
data=task_summary
formula = 'lose_shift ~ emotionOrNot + prolific_age'
emotionOrNot=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
print(emotionOrNot.summary())
```

                Mixed Linear Model Regression Results
    =============================================================
    Model:               MixedLM  Dependent Variable:  lose_shift
    No. Observations:    1020     Method:              ML        
    No. Groups:          340      Scale:               0.0089    
    Min. group size:     3        Log-Likelihood:      663.9764  
    Max. group size:     3        Converged:           Yes       
    Mean group size:     3.0                                     
    -------------------------------------------------------------
                        Coef. Std.Err.   z    P>|z| [0.025 0.975]
    -------------------------------------------------------------
    Intercept           0.583    0.022 26.090 0.000  0.539  0.627
    emotionOrNot[T.Not] 0.022    0.006  3.469 0.001  0.009  0.034
    prolific_age        0.001    0.000  2.736 0.006  0.000  0.002
    Group Var           0.014    0.017                           
    =============================================================

<p>

Both show a significant effect - i.e., whether learning is disgusting or
not AND whether learning is emotional or not predicts lose shift.
</p>

<p>

But we can test which explains the data better by comparing the fit of
the data (using BIC - as done before)
</p>

``` python
bic=pd.DataFrame({'disgustOrNot': [disgustOrNot.bic],
                    'emotionOrNot': [emotionOrNot.bic]})
print(bic.sort_values(by=0, axis=1))
```

       emotionOrNot  disgustOrNot
    0   -1293.31501  -1290.303362

<p>

This shows that emotionOrNot is the better fitting model (although only
marginally as the difference in BIC is small).
</p>

<p>

Although there is a difference between <b>both</b> disgust-based
learning vs other learning and emotion-based vs points-based learning,
the model with a difference in emotional learning relative to points
learning (compared to disgust learning relative to other types of
learning) is slightly more parsimonious
</p>

<br>
<h3>

<b>Exploratory analysis 5:</b> running the video ratings analysis with
<b>all</b> of the fear and disgust videos
</h3>

<p>

The planned video rating analysis involved just the stimuli selected to
be used in the reversal learning task (to validate the stimulus
selection process)
</p>

<p>

Here, we look at valence and arousal ratings across <b>all 10 fear and
disgust videos</b>
</p>

<br>
<p>

Firstly, create a long-form dataframe to allow for this
</p>

<details class="code-fold">
<summary>Code</summary>

``` python
long_vid_ratings=pd.DataFrame()
for i in vid_ratings_df.index:
    row=vid_ratings_df.loc[i]
    timepoint_1=pd.DataFrame({
    'participant_no': [row.participant_no],
    #'age': [row.prolific_age],
    #'sex': [row. prolific_sex],
    'Vid': [str(row['Vid'])],
    'trial_type': [row.trial_type],
    'Valence': [row.unpleasant_1],
    'Arousal': [row.arousing_1],
    'Fear': [row.frightening_1],
    'Disgust': [row.disgusting_1],
    'Timepoint': 1.0
    })
    timepoint_2=pd.DataFrame({
        'participant_no': [row.participant_no],
        #'age': [row.prolific_age],
        #'sex': [row. prolific_sex],
        'Vid': [str(row['Vid'])],
        'trial_type': [row.trial_type],
        'Valence': [row.unpleasant_2],
        'Arousal': [row.arousing_2],
        'Fear': [row.frightening_2],
        'Disgust': [row.disgusting_2],
        'Timepoint': 2.0
    })
    long_vid_ratings_row=pd.concat([timepoint_1, timepoint_2])
    long_vid_ratings=pd.concat([long_vid_ratings_row, long_vid_ratings])
    long_vid_ratings=long_vid_ratings[long_vid_ratings.trial_type!="points"]

long_vid_ratings=pd.merge(long_vid_ratings, task_summary[['participant_no', 'prolific_age', 'prolific_sex']].drop_duplicates(), on='participant_no', how='outer')
```

</details>

<p>

<b>Models A and B show that differences in valence and arousal are
larger when looking across all videos (rather than just looking at the
‘chosen’ videos)</b>
</p>

<br>
<p>

Valence
</p>

``` python
data=long_vid_ratings.reset_index()
data.replace(['disgust', 'fear'], [1.0,2.0], inplace=True)

formula = 'Valence ~ trial_type*Timepoint + prolific_age'

model=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', vc_formula={'Vid': '0+Vid'}, re_formula='~trial_type')
results=model.fit(reml=False)
print(results.summary())
```

                  Mixed Linear Model Regression Results
    ==================================================================
    Model:               MixedLM    Dependent Variable:    Valence    
    No. Observations:    6800       Method:                ML         
    No. Groups:          340        Scale:                 1.8274     
    Min. group size:     20         Log-Likelihood:        -13945.4510
    Max. group size:     20         Converged:             Yes        
    Mean group size:     20.0                                         
    ------------------------------------------------------------------
                           Coef.  Std.Err.    z    P>|z| [0.025 0.975]
    ------------------------------------------------------------------
    Intercept               5.968    0.323  18.493 0.000  5.335  6.600
    trial_type             -1.300    0.129 -10.057 0.000 -1.553 -1.047
    Timepoint               0.312    0.104   3.007 0.003  0.109  0.515
    trial_type:Timepoint   -0.022    0.066  -0.341 0.733 -0.151  0.106
    prolific_age           -0.010    0.005  -1.863 0.062 -0.020  0.001
    Group Var               5.177    0.450                            
    Group x trial_type Cov -2.116    0.231                            
    trial_type Var          1.378    0.139                            
    Vid Var                 1.622    0.068                            
    ==================================================================

<p>

Arousal
</p>

``` python
data=long_vid_ratings.reset_index()
data.replace(['disgust', 'fear'], [1.0,2.0], inplace=True)

formula = 'Arousal ~ trial_type*Timepoint + prolific_age'

model=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', vc_formula={'Vid': '0+Vid'}, re_formula='~trial_type')
results=model.fit(reml=False)
print(results.summary())
```

                  Mixed Linear Model Regression Results
    =================================================================
    Model:                MixedLM   Dependent Variable:   Arousal    
    No. Observations:     6800      Method:               ML         
    No. Groups:           340       Scale:                1.6158     
    Min. group size:      20        Log-Likelihood:       -12712.9704
    Max. group size:      20        Converged:            Yes        
    Mean group size:      20.0                                       
    -----------------------------------------------------------------
                           Coef.  Std.Err.   z    P>|z| [0.025 0.975]
    -----------------------------------------------------------------
    Intercept               2.579    0.273  9.457 0.000  2.044  3.113
    trial_type              0.736    0.107  6.858 0.000  0.526  0.946
    Timepoint               0.262    0.097  2.691 0.007  0.071  0.453
    trial_type:Timepoint   -0.212    0.062 -3.434 0.001 -0.333 -0.091
    prolific_age            0.012    0.005  2.651 0.008  0.003  0.021
    Group Var               2.490    0.242                           
    Group x trial_type Cov -0.700    0.109                           
    trial_type Var          0.451    0.062                           
    Vid Var                 0.582    0.040                           
    =================================================================

<p>

<b>And similar results are found when looking just at T1 in comparison
to the points/loss feedback</b>
</p>

<br>
<p>

Valence
</p>

``` python
T1_and_points_data=pd.concat([vid_ratings_df,chosen_stim_df[chosen_stim_df.trial_type=='points']]).reset_index()
T1_and_points_data=T1_and_points_data[['participant_no', 'trial_type', 'unpleasant_1', 'arousing_1', 'disgusting_1', 'frightening_1', 'Vid']].sort_values('trial_type')
T1_and_points_data=pd.merge(T1_and_points_data, task_summary[['participant_no', 'prolific_age', 'prolific_sex']].drop_duplicates(), on='participant_no', how='outer')

data=T1_and_points_data
data.replace(['points'],['apoints'], inplace=True) ##makes comparison condition points
formula = 'unpleasant_1 ~ trial_type + prolific_age'

model=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop')
results=model.fit(reml=False)
print(results.summary())
```

                  Mixed Linear Model Regression Results
    =================================================================
    Model:               MixedLM   Dependent Variable:   unpleasant_1
    No. Observations:    3658      Method:               ML          
    No. Groups:          340       Scale:                3.9104      
    Min. group size:     10        Log-Likelihood:       -7966.4546  
    Max. group size:     11        Converged:            Yes         
    Mean group size:     10.8                                        
    -----------------------------------------------------------------
                          Coef.  Std.Err.    z    P>|z| [0.025 0.975]
    -----------------------------------------------------------------
    Intercept              5.210    0.266  19.590 0.000  4.689  5.731
    trial_type[T.disgust] -0.204    0.133  -1.533 0.125 -0.465  0.057
    trial_type[T.fear]    -1.527    0.133 -11.458 0.000 -1.788 -1.265
    prolific_age          -0.011    0.005  -2.209 0.027 -0.021 -0.001
    Group Var              1.546    0.078                            
    =================================================================

<p>

Arousal
</p>

``` python
formula = 'arousing_1 ~ trial_type + prolific_age'
model=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', re_formula='~trial_type')
results=model.fit(reml=False)
print(results.summary())
```

                              Mixed Linear Model Regression Results
    =========================================================================================
    Model:                       MixedLM            Dependent Variable:            arousing_1
    No. Observations:            3658               Method:                        ML        
    No. Groups:                  340                Scale:                         2.2275    
    Min. group size:             10                 Log-Likelihood:                -7100.7929
    Max. group size:             11                 Converged:                     Yes       
    Mean group size:             10.8                                                        
    -----------------------------------------------------------------------------------------
                                                   Coef.  Std.Err.   z    P>|z| [0.025 0.975]
    -----------------------------------------------------------------------------------------
    Intercept                                       4.333    0.226 19.136 0.000  3.889  4.777
    trial_type[T.disgust]                          -0.887    0.131 -6.771 0.000 -1.144 -0.630
    trial_type[T.fear]                             -0.363    0.134 -2.715 0.007 -0.625 -0.101
    prolific_age                                    0.010    0.004  2.473 0.013  0.002  0.018
    Group Var                                       1.769    0.248                           
    Group x trial_type[T.disgust] Cov              -1.150    0.236                           
    trial_type[T.disgust] Var                       1.931    0.285                           
    Group x trial_type[T.fear] Cov                 -1.359    0.246                           
    trial_type[T.disgust] x trial_type[T.fear] Cov  1.860    0.270                           
    trial_type[T.fear] Var                          2.170    0.298                           
    =========================================================================================

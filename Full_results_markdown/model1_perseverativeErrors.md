# Model 1: mean perseverative errors per reversal ~ feedback type


<p>

This file contains all model-agnostic tests run to test the effect of
feedback type (fear, disgust, points) on perseverative errors.
</p>

<br> Includes:
<p>

- data visualisation
- initial skew assessment (and resulting skew transformation)
- initial hypothesis testing mixed effects model
- assessment of assumptions of this model (which was violated)
- resulting generalized mixed effects model
- assessing whether adding video-ratings differences (identified in
  video-rating analyses) moderates results
- sensitivity analysis (including generalized mixed effects models)
- final conclusions

</p>

<h3>

Load in packages and data- in r and then in python
</h3>

<details class="code-fold">
<summary>Code</summary>

``` r
library(tidyverse, quietly=TRUE)
library(lme4)
library(emmeans)
library(DHARMa)

task_summary <- read.csv("U:/Documents/Disgust learning project/github/disgust_reversal_learning-final/csvs/dem_vids_task_excluded.csv")
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
```

</details>

<br>
<h3>

<b>Visualise the data</b>
</h3>

<details class="code-fold">
<summary>Code</summary>

``` python
palette = ["#F72585", "#3A0CA3", "#4CC9F0"]

##plot hypothesised results
fig, axes = plt.subplots(1,1, sharey=False)

sns.stripplot(data=task_summary, x="block_type", y="mean_perseverative_er", ax=axes, palette=palette, size=5, jitter=True, marker='.')
sns.violinplot(data=task_summary, x="block_type", y="mean_perseverative_er", ax=axes,fill=True, inner="quart", palette=palette, saturation=0.5, cut=0)
#axes.set_xlabel("Feedback type")
axes.set_xlabel("")
axes.set_xticklabels(axes.get_xticklabels(), rotation=0)
axes.set_ylabel("Mean perseverative errors per reversal") 
axes.set_title("Perseverative errors")
```

</details>

![](model1_perseverativeErrors_files/figure-commonmark/Visualisation-1.png)

<br>
<h3>

Assess and correct for skewness in perservative error outcome
</h3>

<details class="code-fold">
<summary>Code</summary>

``` python
pt=PowerTransformer(method='yeo-johnson', standardize=False)
skl_yeojohnson=pt.fit(pd.DataFrame(task_summary.mean_perseverative_er))
skl_yeojohnson=pt.transform(pd.DataFrame(task_summary.mean_perseverative_er))
task_summary['perseverative_er_transformed'] = pt.transform(pd.DataFrame(task_summary.mean_perseverative_er))


fig, axes = plt.subplots(1,2, sharey=True)
sns.histplot(data=task_summary, x="mean_perseverative_er", ax=axes[0]) 
sns.histplot(data=task_summary['perseverative_er_transformed'], ax=axes[1])
print('Perseverative error skew: '+str(skew(task_summary.mean_perseverative_er)))
```

</details>

    Perseverative error skew: 2.400818032551747

![](model1_perseverativeErrors_files/figure-commonmark/Skewness-3.png)

<h3>

<b>Hypothesis testing</b>
</h3>

In this case, the basic model (no random slopes or random intercepts,
and no covariates) produced the best fit (indexed by BIC scores).
<p>

The model shows no effect of feedback type (although disgust vs points
comparison is nearing significance).

``` python
data=task_summary.reset_index()
formula = 'perseverative_er_transformed ~ block_type'
results=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
print(results.summary())
```

                      Mixed Linear Model Regression Results
    ==========================================================================
    Model:            MixedLM Dependent Variable: perseverative_er_transformed
    No. Observations: 1020    Method:             ML                          
    No. Groups:       340     Scale:              0.0375                      
    Min. group size:  3       Log-Likelihood:     95.7445                     
    Max. group size:  3       Converged:          Yes                         
    Mean group size:  3.0                                                     
    ----------------------------------------------------------------------------
                            Coef.    Std.Err.     z      P>|z|   [0.025   0.975]
    ----------------------------------------------------------------------------
    Intercept                0.405      0.012   32.755   0.000    0.381    0.430
    block_type[T.Fear]      -0.022      0.015   -1.470   0.141   -0.051    0.007
    block_type[T.Points]    -0.028      0.015   -1.873   0.061   -0.057    0.001
    Group Var                0.015      0.013                                   
    ==========================================================================

<b>BUT</b> the residuals of this model are significantly non-normal! So
we will need to run a generalized mixed effects model.

``` python
#shapiro-Wilk test of normality of residuals
labels = ["Statistic", "p-value"]
norm_res = stats.shapiro(results.resid)
for key, val in dict(zip(labels, norm_res)).items():
    print(key, val)
```

    Statistic 0.9931933132269517
    p-value 0.00012825572082599747

<h4>

<b>Run a generalized mixed effects model (done in R)</b>
</h4>

Model details:
<p>

- Gamma probability distribution and inverse link function
  <p>

  - no additional random effects or slopes
    <p>

    - no additional covariates
      <p>

      This is the specification that produced the best fit (according to
      BIC)
      </p>

<p>

Results from this model show a <b>significant effect of block-type</b>:
specifically, a difference between disgust and points learning

``` r
task_summary$pos_perseverative_er <- task_summary$mean_perseverative_er + 0.01 ##+0.01 as all values must be positive (i.e., can't have 0s)
generalized_model <- glmer(pos_perseverative_er ~ block_type + (1|participant_no), data=task_summary, family=Gamma(link="inverse"))
summary(generalized_model)
```

    Generalized linear mixed model fit by maximum likelihood (Laplace
      Approximation) [glmerMod]
     Family: Gamma  ( inverse )
    Formula: pos_perseverative_er ~ block_type + (1 | participant_no)
       Data: task_summary

         AIC      BIC   logLik deviance df.resid 
      1587.3   1611.9   -788.7   1577.3     1015 

    Scaled residuals: 
        Min      1Q  Median      3Q     Max 
    -1.2647 -0.7161 -0.1944  0.5332  3.8144 

    Random effects:
     Groups         Name        Variance Std.Dev.
     participant_no (Intercept) 0.1441   0.3796  
     Residual                   0.6133   0.7831  
    Number of obs: 1020, groups:  participant_no, 340

    Fixed effects:
                     Estimate Std. Error t value Pr(>|z|)    
    (Intercept)       1.32174    0.07274  18.172   <2e-16 ***
    block_typeFear    0.13312    0.07953   1.674   0.0942 .  
    block_typePoints  0.18930    0.08252   2.294   0.0218 *  
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Correlation of Fixed Effects:
                (Intr) blck_F
    block_typFr -0.503       
    blck_typPnt -0.500  0.409

<br>
<p>

As this hypothesis test found a no difference between fear and disgust,
we will compute a Bayes Factor to test the strength of the evidence for
the null
</p>

``` python
#|code-fold: true
def bayes_factor(df, dependent_var, condition_1_name, condition_2_name):
    df=df[(df.block_type==condition_1_name)| (df.block_type==condition_2_name)][[dependent_var, 'block_type', 'participant_no']]
    df.dropna(inplace=True)
    df=df.pivot(index='participant_no', columns='block_type', values=dependent_var).reset_index()
    ttest=pg.ttest(df[condition_1_name], df[condition_2_name], paired=True)
    bf_null=1/float(ttest.BF10)
    return bf_null
```

``` python
print(bayes_factor(task_summary, 'perseverative_er_transformed', 'Disgust', 'Fear'))
```

    6.369426751592357

<p>

This means that there is <b>moderate</b> evidence for the null
</p>

<p>

We also look at fear vs points (which is not directly assessed by the
model)
</p>

``` python
print(bayes_factor(task_summary, 'perseverative_er_transformed', 'Points', 'Fear'))
```

    14.925373134328357

<p>

This means that there is <b>strong</b> evidence for the null
</p>

<br>
<h3>

<b>Adding video ratings</b>
</h3>

Finally, we will test whether this effect remains after video rating
differences between fear and disgust have been controlled for.
<p>

As before, the mixed effects model violated assumptions, so a
generalized mixed effects model is run.
</p>

<details class="code-fold">
<summary>Code</summary>

``` python
formula = 'perseverative_er_transformed ~ block_type + valence_diff + arousal_diff + valence_habdiff'
results=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
#shapiro-Wilk test of normality of residuals
labels = ["Statistic", "p-value"]
norm_res = stats.shapiro(results.resid)
#
for key, val in dict(zip(labels, norm_res)).items():
    print(key, val)
```

</details>

    Statistic 0.9932531693384103
    p-value 0.0001396501655398863

Model details:
<p>

- Gamma probability distribution and inverse link function
  <p>

  - no additional random effects or slopes
    <p>

    - no additional covariates
      <p>

      This is the specification that produced the best fit (according to
      BIC)
      </p>

      <br>
      <p>

      This has <b> no effect </b> on the results

``` r
task_summary$pos_perseverative_er <- task_summary$mean_perseverative_er + 0.01 ##+0.01 as all values must be positive (i.e., can't have 0s)
generalized_model <- glmer(pos_perseverative_er ~ block_type +valence_diff + arousal_diff + valence_habdiff + (1|participant_no), data=task_summary, family=Gamma(link="inverse"))
summary(generalized_model)
```

    Generalized linear mixed model fit by maximum likelihood (Laplace
      Approximation) [glmerMod]
     Family: Gamma  ( inverse )
    Formula: pos_perseverative_er ~ block_type + valence_diff + arousal_diff +  
        valence_habdiff + (1 | participant_no)
       Data: task_summary

         AIC      BIC   logLik deviance df.resid 
      1591.1   1630.5   -787.6   1575.1     1012 

    Scaled residuals: 
        Min      1Q  Median      3Q     Max 
    -1.2639 -0.7127 -0.2102  0.5384  3.9164 

    Random effects:
     Groups         Name        Variance Std.Dev.
     participant_no (Intercept) 0.1426   0.3776  
     Residual                   0.6140   0.7836  
    Number of obs: 1020, groups:  participant_no, 340

    Fixed effects:
                       Estimate Std. Error t value Pr(>|z|)    
    (Intercept)       1.3221837  0.0794095  16.650   <2e-16 ***
    block_typeFear    0.1331227  0.0795042   1.674   0.0940 .  
    block_typePoints  0.1892452  0.0824777   2.295   0.0218 *  
    valence_diff      0.0189348  0.0284999   0.664   0.5064    
    arousal_diff      0.0436398  0.0372152   1.173   0.2409    
    valence_habdiff  -0.0006996  0.0214278  -0.033   0.9740    
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Correlation of Fixed Effects:
                (Intr) blck_F blck_P vlnc_d arsl_d
    block_typFr -0.460                            
    blck_typPnt -0.456  0.409                     
    valence_dff  0.382 -0.001 -0.002              
    arousal_dff -0.198 -0.006 -0.007 -0.189       
    valnc_hbdff -0.218  0.000 -0.001 -0.380  0.121

<br> <br>
<h3>

<b> Sensitivity analysis </b>
</h3>

We also ran the same analyses after outliers had been excluded, to
assess whether outliers are driving this effect.

<p>

Firstly, exclude outliers from the dataframe (outliers are define as
those \>1.5 IQRs above or below the upper or lower quartile)

<details class="code-fold">
<summary>Code</summary>

``` python
#create outliers df --> removing those >1.5 IQRs above or below UQ and LQ
def replace_outliers_with_nan(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1- 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column]=df[column].apply(lambda x: np.nan if x<lower_bound or x>upper_bound else x)
    return df

key_outcomes=['percentage_correct', 'mean_perseverative_er', 'mean_regressive_er', 'median_till_correct', 'win_stay', 'lose_shift']
for col in key_outcomes:
    task_summary=replace_outliers_with_nan(task_summary, col)
task_summary.to_csv('sensitivity_df.csv')
sensitivity_df=task_summary
```

</details>

<br>
<h3>

Assess and correct for skewness in perservative error outcome (excluding
outliers)
</h3>

<details class="code-fold">
<summary>Code</summary>

``` python
pt=PowerTransformer(method='yeo-johnson', standardize=False)
skl_yeojohnson=pt.fit(pd.DataFrame(sensitivity_df.mean_perseverative_er))
skl_yeojohnson=pt.transform(pd.DataFrame(sensitivity_df.mean_perseverative_er))
sensitivity_df['perseverative_er_transformed'] = pt.transform(pd.DataFrame(sensitivity_df.mean_perseverative_er))


fig, axes = plt.subplots(1,2, sharey=True)
sns.histplot(data=sensitivity_df, x="mean_perseverative_er", ax=axes[0]) 
sns.histplot(data=sensitivity_df['perseverative_er_transformed'], ax=axes[1])
print('Perseverative error skew: '+str(skew(sensitivity_df.mean_perseverative_er.dropna())))
```

</details>

    Perseverative error skew: 0.9439083554848277

![](model1_perseverativeErrors_files/figure-commonmark/Skewness%20sensitivity-1.png)

<h3>

<b>Outlier-free hypothesis testing</b>
</h3>

In this case, the basic model (no random slopes or random intercepts,
and no covariates) produced the best fit (indexed by BIC scores).
<p>

The model shows no effect of feedback type (and this time no effect is
approaching significance)

``` python
data=sensitivity_df.reset_index()
formula = 'perseverative_er_transformed ~ block_type'

results=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
print(results.summary())
```

                      Mixed Linear Model Regression Results
    ==========================================================================
    Model:            MixedLM Dependent Variable: perseverative_er_transformed
    No. Observations: 983     Method:             ML                          
    No. Groups:       339     Scale:              0.0425                      
    Min. group size:  1       Log-Likelihood:     45.5258                     
    Max. group size:  3       Converged:          Yes                         
    Mean group size:  2.9                                                     
    ----------------------------------------------------------------------------
                            Coef.    Std.Err.     z      P>|z|   [0.025   0.975]
    ----------------------------------------------------------------------------
    Intercept                0.403      0.013   30.442   0.000    0.377    0.429
    block_type[T.Fear]      -0.005      0.016   -0.307   0.759   -0.037    0.027
    block_type[T.Points]    -0.010      0.016   -0.596   0.551   -0.041    0.022
    Group Var                0.014      0.013                                   
    ==========================================================================

<b>BUT, again</b> the residuals of this model are significantly
non-normal! So we will need to run a generalized mixed effects model.

``` python
#shapiro-Wilk test of normality of residuals
labels = ["Statistic", "p-value"]
norm_res = stats.shapiro(results.resid)
for key, val in dict(zip(labels, norm_res)).items():
    print(key, val)
```

    Statistic 0.9901267071073535
    p-value 3.606836892222057e-06

<h4>

Run a generalized mixed effects model (done in R)
</h4>

Model details:
<p>

- Gamma probability distribution and inverse link function
  <p>

  - no additional random effects or slopes
    <p>

    - no additional covariate
      <p>

      This is the specification that produced the best fit (according to
      BIC)
      </p>

<p>

Results from this model show <b>no effect of block-type</b>: suggests
that the difference seen before is driven by outliers.

``` r
sensitivity_df <- read.csv("sensitivity_df.csv")
sensitivity_df$pos_perseverative_er <- sensitivity_df$mean_perseverative_er + 0.01 ##+0.01 as all values must be positive (i.e., can't have 0s)
generalized_model <- glmer(pos_perseverative_er ~ block_type + (1|participant_no), data=sensitivity_df, family=Gamma(link="inverse"))
summary(generalized_model)
```

    Generalized linear mixed model fit by maximum likelihood (Laplace
      Approximation) [glmerMod]
     Family: Gamma  ( inverse )
    Formula: pos_perseverative_er ~ block_type + (1 | participant_no)
       Data: sensitivity_df

         AIC      BIC   logLik deviance df.resid 
      1317.7   1342.1   -653.8   1307.7      978 

    Scaled residuals: 
        Min      1Q  Median      3Q     Max 
    -1.2838 -0.7218 -0.1792  0.6114  2.9242 

    Random effects:
     Groups         Name        Variance Std.Dev.
     participant_no (Intercept) 0.07353  0.2712  
     Residual                   0.59070  0.7686  
    Number of obs: 983, groups:  participant_no, 339

    Fixed effects:
                     Estimate Std. Error t value Pr(>|z|)    
    (Intercept)      1.467350   0.086799  16.905   <2e-16 ***
    block_typeFear   0.002079   0.101212   0.021    0.984    
    block_typePoints 0.046761   0.102808   0.455    0.649    
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Correlation of Fixed Effects:
                (Intr) blck_F
    block_typFr -0.591       
    blck_typPnt -0.589  0.501

<br>
<p>

As this hypothesis test found a no difference between fear and disgust
or disgust and points, we will compute a Bayes Factor to test the
strength of the evidence for the null
</p>

<p>

Firstly for disgust vs fear:
</p>

``` python
print(bayes_factor(task_summary, 'perseverative_er_transformed', 'Disgust', 'Fear'))
```

    13.88888888888889

<p>

This means that there is <b>strong</b> evidence for the null
</p>

<br>
<p>

Next for disgust vs points:
</p>

``` python
print(bayes_factor(task_summary, 'perseverative_er_transformed', 'Disgust', 'Points'))
```

    12.658227848101266

<p>

This means that there is <b>strong</b> evidence for the null
</p>

<br>
<p>

We also look at fear vs points (which is not directly assessed by the
model)
</p>

``` python
print(bayes_factor(task_summary, 'perseverative_er_transformed', 'Points', 'Fear'))
```

    15.15151515151515

<p>

This means that there is <b>strong</b> evidence for the null
</p>

<br> <br>
<p>

<b>Finally, adding video rating values to this model has no effect: </b>
</p>

<p>

Again, a generalized mixed effects model must be run because assumptions
for original model were violated
</p>

<details class="code-fold">
<summary>Code</summary>

``` python
formula = 'perseverative_er_transformed ~ block_type + valence_diff + arousal_diff + valence_habdiff'
results=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
#shapiro-Wilk test of normality of residuals
labels = ["Statistic", "p-value"]
norm_res = stats.shapiro(results.resid)
#
for key, val in dict(zip(labels, norm_res)).items():
    print(key, val)
```

</details>

    Statistic 0.98989938998603
    p-value 2.7648916440248165e-06

Generalized mixed effects model details:
<p>

- Gamma probability distribution and inverse link function
  <p>

  - no additional random effects or slopes
    <p>

    - no additional covariate
      <p>

      This is the specification that produced the best fit (according to
      BIC)
      </p>

``` r
sensitivity_df <- read.csv("sensitivity_df.csv")
sensitivity_df$pos_perseverative_er <- sensitivity_df$mean_perseverative_er + 0.01 ##+0.01 as all values must be positive (i.e., can't have 0s)
generalized_model <- glmer(pos_perseverative_er ~ block_type + valence_diff + arousal_diff + valence_habdiff + (1|participant_no), data=sensitivity_df, family=Gamma(link="inverse"))
summary(generalized_model)
```

    Generalized linear mixed model fit by maximum likelihood (Laplace
      Approximation) [glmerMod]
     Family: Gamma  ( inverse )
    Formula: pos_perseverative_er ~ block_type + valence_diff + arousal_diff +  
        valence_habdiff + (1 | participant_no)
       Data: sensitivity_df

         AIC      BIC   logLik deviance df.resid 
      1320.5   1359.6   -652.3   1304.5      975 

    Scaled residuals: 
        Min      1Q  Median      3Q     Max 
    -1.2794 -0.7259 -0.1786  0.6050  3.0781 

    Random effects:
     Groups         Name        Variance Std.Dev.
     participant_no (Intercept) 0.06395  0.2529  
     Residual                   0.59502  0.7714  
    Number of obs: 983, groups:  participant_no, 339

    Fixed effects:
                     Estimate Std. Error t value Pr(>|z|)    
    (Intercept)      1.466766   0.093922  15.617   <2e-16 ***
    block_typeFear   0.001151   0.101617   0.011    0.991    
    block_typePoints 0.046087   0.103220   0.446    0.655    
    valence_diff     0.027701   0.028146   0.984    0.325    
    arousal_diff     0.045818   0.037232   1.231    0.218    
    valence_habdiff  0.002931   0.022500   0.130    0.896    
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Correlation of Fixed Effects:
                (Intr) blck_F blck_P vlnc_d arsl_d
    block_typFr -0.547                            
    blck_typPnt -0.545  0.501                     
    valence_dff  0.315  0.001  0.002              
    arousal_dff -0.192 -0.008 -0.009 -0.190       
    valnc_hbdff -0.206  0.002  0.001 -0.375  0.111

<h4>

<b>Exploratory analyses</b>
</h4>

<p>

- Points ratings
  <p>

  - error rates

# Model 4: lose shift probability ~ feedback type


<p>

This file contains all model-agnostic tests run to test the effect of
feedback type (fear, disgust, points) on lose-shift probability.
</p>

<br> Includes:
<p>

- initial skew assessment
- initial hypothesis testing mixed effects model
- assessment of assumptions of this model
- assessing whether adding video-ratings differences (identified in
  video-rating analyses) moderates results
- sensitivity analysis
- final conclusions

</p>

<h3>

Load in packages and data- in python and in r
</h3>

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

sns.stripplot(data=task_summary, x="block_type", y="lose_shift", ax=axes, palette=palette, size=5, jitter=True, marker='.')
sns.violinplot(data=task_summary, x="block_type", y="lose_shift", ax=axes,fill=True, inner="quart", palette=palette, saturation=0.5, cut=0)
#axes.set_xlabel("Feedback type")
axes.set_xlabel("")
axes.set_xticklabels(axes.get_xticklabels(), rotation=0)
axes.set_ylabel("Lose-shift probability") 
axes.set_title("Lose-shift")
```

</details>

![](model4_loseShift_files/figure-commonmark/Visualisation-1.png)

<br>
<h3>

The lose-shift variable is not skewed
</h3>

<p>

So no skewness transformation is required.
</p>

<details class="code-fold">
<summary>Code</summary>

``` python
sns.histplot(data=task_summary, x="lose_shift") 
print('lose-shift  skew: '+str(skew(task_summary.lose_shift, nan_policy='omit')))
```

</details>

    lose-shift  skew: 0.16189569452640257

![](model4_loseShift_files/figure-commonmark/Skewness-3.png)

<h3>

<b>Hypothesis testing</b>
</h3>

<p>

In this case, the basic model (no random slopes or random intercepts)
with an age covariate produced the best fit (indexed by BIC scores).
<p>

This model showed a <b>significant effect of feedback-type on lose-shift
probability</b>
</p>

<p>

Specifically, the points block had a significantly higher lose-shift
probability than the disgust-block.
</p>

<p>

There is also a significant effect of age on lose-shift probability.
</p>

``` python
data=task_summary
formula = 'lose_shift ~ block_type + prolific_age'
results=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
print(results.summary())
```

                Mixed Linear Model Regression Results
    ==============================================================
    Model:              MixedLM   Dependent Variable:   lose_shift
    No. Observations:   1020      Method:               ML        
    No. Groups:         340       Scale:                0.0089    
    Min. group size:    3         Log-Likelihood:       665.0506  
    Max. group size:    3         Converged:            Yes       
    Mean group size:    3.0                                       
    --------------------------------------------------------------
                         Coef. Std.Err.   z    P>|z| [0.025 0.975]
    --------------------------------------------------------------
    Intercept            0.578    0.023 25.522 0.000  0.533  0.622
    block_type[T.Fear]   0.011    0.007  1.467 0.142 -0.004  0.025
    block_type[T.Points] 0.027    0.007  3.743 0.000  0.013  0.041
    prolific_age         0.001    0.000  2.736 0.006  0.000  0.002
    Group Var            0.014    0.017                           
    ==============================================================

<p>

The assumptions for this model are not violated
</p>

``` python
#shapiro-Wilk test of normality of residuals
labels = ["Statistic", "p-value"]
norm_res = stats.shapiro(results.resid)
for key, val in dict(zip(labels, norm_res)).items():
    print(key, val)
```

    Statistic 0.9982479633219511
    p-value 0.3850884723504157

``` python
##homoskedasticity of variance 
#White Lagrange Multiplier Test for Heteroscedasticity
het_white_res = het_white(results.resid, results.model.exog)

labels = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]

for key, val in dict(zip(labels, het_white_res)).items():
    print(key, val)
```

    LM Statistic 7.570756342605051
    LM-Test p-value 0.2712665370232391
    F-Statistic 1.2625040586135379
    F-Test p-value 0.27200186565647244

<p>

And the results remain unchanged when the age covariate is dropped:
</p>

``` python
formula = 'lose_shift ~ block_type'
results=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
print(results.summary())
```

                Mixed Linear Model Regression Results
    ==============================================================
    Model:              MixedLM   Dependent Variable:   lose_shift
    No. Observations:   1020      Method:               ML        
    No. Groups:         340       Scale:                0.0089    
    Min. group size:    3         Log-Likelihood:       661.3490  
    Max. group size:    3         Converged:            Yes       
    Mean group size:    3.0                                       
    --------------------------------------------------------------
                         Coef. Std.Err.   z    P>|z| [0.025 0.975]
    --------------------------------------------------------------
    Intercept            0.636    0.008 76.839 0.000  0.619  0.652
    block_type[T.Fear]   0.011    0.007  1.467 0.142 -0.004  0.025
    block_type[T.Points] 0.027    0.007  3.743 0.000  0.013  0.041
    Group Var            0.014    0.017                           
    ==============================================================

<br>
<p>

As this hypothesis test found a no difference between fear and disgust,
we will compute a Bayes Factor to test the strength of the evidence for
the null
</p>

<details class="code-fold">
<summary>Code</summary>

``` python
def bayes_factor(df, dependent_var, condition_1_name, condition_2_name):
    df=df[(df.block_type==condition_1_name)| (df.block_type==condition_2_name)][[dependent_var, 'block_type', 'participant_no']]
    df.dropna(inplace=True)
    df=df.pivot(index='participant_no', columns='block_type', values=dependent_var).reset_index()
    ttest=pg.ttest(df[condition_1_name], df[condition_2_name], paired=True)
    bf_null=1/float(ttest.BF10)
    return bf_null
```

</details>

``` python
print(bayes_factor(task_summary, 'lose_shift', 'Disgust', 'Fear'))
```

    6.172839506172839

<p>

This means that there is <b>moderate</b> evidence for the null
</p>

<p>

We also look at fear vs points (which is not directly assessed by the
model)
</p>

``` python
print(bayes_factor(task_summary, 'lose_shift', 'Points', 'Fear'))
```

    1.1013215859030836

<p>

This means that there is <b>weak</b> evidence for the null
</p>

<br>
<p>

<b>Next, we showed that this result is unchanged by the addition of
video-rating covariates.</b>
</p>

<p>

I.e., there is a main effect of block-type (driven by a difference
between disgust and points) and a main effect of age
<p>

(again, the model with no additional random effects/slopes, with an age
covariate produced the best fit)
</p>

``` python
formula = 'lose_shift ~ block_type + valence_diff + arousal_diff + valence_habdiff + prolific_age'
results=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
print(results.summary())
```

                 Mixed Linear Model Regression Results
    ===============================================================
    Model:               MixedLM   Dependent Variable:   lose_shift
    No. Observations:    1020      Method:               ML        
    No. Groups:          340       Scale:                0.0089    
    Min. group size:     3         Log-Likelihood:       666.1434  
    Max. group size:     3         Converged:            Yes       
    Mean group size:     3.0                                       
    ---------------------------------------------------------------
                         Coef.  Std.Err.   z    P>|z| [0.025 0.975]
    ---------------------------------------------------------------
    Intercept             0.585    0.023 24.989 0.000  0.539  0.630
    block_type[T.Fear]    0.011    0.007  1.467 0.142 -0.004  0.025
    block_type[T.Points]  0.027    0.007  3.743 0.000  0.013  0.041
    valence_diff          0.005    0.005  1.150 0.250 -0.004  0.014
    arousal_diff         -0.002    0.006 -0.302 0.763 -0.014  0.010
    valence_habdiff      -0.004    0.003 -1.269 0.205 -0.011  0.002
    prolific_age          0.001    0.000  2.694 0.007  0.000  0.002
    Group Var             0.014    0.017                           
    ===============================================================

<p>

However, this model fails the homoskedasticity of variance test
</p>

``` python
##homoskedasticity of variance 
#White Lagrange Multiplier Test for Heteroscedasticity
het_white_res = het_white(results.resid, results.model.exog)

labels = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]

for key, val in dict(zip(labels, het_white_res)).items():
    print(key, val)
```

    LM Statistic 43.86646288220306
    LM-Test p-value 0.007907172665762914
    F-Statistic 1.8630959506774956
    F-Test p-value 0.0071939874304332374

<p>

So a generalized mixed effects model needs to be run (using R)
</p>

<p>

Model details:
<p>

- Gamma probability distribution and identity link function
- no additional random intercepts or by-participant random slopes
- no additional covariates

</p>

<p>

This is the specification that produced the best fit (according to BIC)
</p>

``` r
generalized_model <- glmer(lose_shift ~ block_type + valence_diff + arousal_diff + valence_habdiff + (1|participant_no), data=task_summary, family=Gamma(link="identity"))
summary(generalized_model)
```

    Generalized linear mixed model fit by maximum likelihood (Laplace
      Approximation) [glmerMod]
     Family: Gamma  ( identity )
    Formula: 
    lose_shift ~ block_type + valence_diff + arousal_diff + valence_habdiff +  
        (1 | participant_no)
       Data: task_summary

         AIC      BIC   logLik deviance df.resid 
     -1633.1  -1593.7    824.6  -1649.1     1012 

    Scaled residuals: 
         Min       1Q   Median       3Q      Max 
    -2.57848 -0.57964 -0.00125  0.55944  2.67923 

    Random effects:
     Groups         Name        Variance Std.Dev.
     participant_no (Intercept) 0.009516 0.09755 
     Residual                   0.026906 0.16403 
    Number of obs: 1020, groups:  participant_no, 340

    Fixed effects:
                       Estimate Std. Error t value Pr(>|z|)    
    (Intercept)       0.6573586  0.0126194  52.091  < 2e-16 ***
    block_typeFear    0.0085007  0.0064622   1.315    0.188    
    block_typePoints  0.0295101  0.0065569   4.501 6.78e-06 ***
    valence_diff      0.0058092  0.0061681   0.942    0.346    
    arousal_diff     -0.0003108  0.0081260  -0.038    0.969    
    valence_habdiff  -0.0039949  0.0046604  -0.857    0.391    
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Correlation of Fixed Effects:
                (Intr) blck_F blck_P vlnc_d arsl_d
    block_typFr -0.254                            
    blck_typPnt -0.247  0.485                     
    valence_dff  0.482 -0.004  0.001              
    arousal_dff -0.341 -0.001  0.002 -0.192       
    valnc_hbdff -0.288  0.000 -0.004 -0.353  0.137

<br> <br>
<h3>

<b> Sensitivity analysis </b>
</h3>

<p>

No sensitivity analysis is required for this outcome variable, because
no values meet the criteria of being an outlier (\>1.5 IQR below LQ or
above UQ)
</p>

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

print("Number of lose-shift outliers: "+str(len(sensitivity_df[sensitivity_df.lose_shift.isna()])))
```

</details>

    Number of lose-shift outliers: 0

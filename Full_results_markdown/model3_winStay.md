# Model 3: win stay probability ~ feedback type


<p>

This file contains all model-agnostic tests run to test the effect of
feedback type (fear, disgust, points) on win-stay probability.
</p>

<br> Includes:
<p>

- initial skew assessment (and resulting skew transformation)
- initial hypothesis testing mixed effects model
- assessment of assumptions of this model
- assessing whether adding video-ratings differences (identified in
  video-rating analyses) moderates results
- sensitivity analysis
- final conclusions

</p>

<h3>

Load in packages and data- in python
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

sns.stripplot(data=task_summary, x="block_type", y="win_stay", ax=axes, palette=palette, size=5, jitter=True, marker='.')
sns.violinplot(data=task_summary, x="block_type", y="win_stay", ax=axes,fill=True, inner="quart", palette=palette, saturation=0.5, cut=0)
#axes.set_xlabel("Feedback type")
axes.set_xlabel("")
axes.set_xticklabels(axes.get_xticklabels(), rotation=0)
axes.set_ylabel("Win-stay probability") 
axes.set_title("Win-stay")
```

</details>

    C:\Users\eb08\AppData\Local\Temp\ipykernel_17808\2893565744.py:10: UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
      axes.set_xticklabels(axes.get_xticklabels(), rotation=0)

    Text(0.5, 1.0, 'Win-stay')

<img
src="model3_winStay_files/figure-commonmark/visualisation-output-3.png"
id="visualisation-2" />

<br>

<br>
<h3>

Assess and correct for skewness in win-stay outcome
</h3>

<details class="code-fold">
<summary>Code</summary>

``` python
pt=PowerTransformer(method='yeo-johnson', standardize=False)
skl_yeojohnson=pt.fit(pd.DataFrame(task_summary.win_stay))
skl_yeojohnson=pt.transform(pd.DataFrame(task_summary.win_stay))
task_summary['win_stay_transformed'] = pt.transform(pd.DataFrame(task_summary.win_stay))

fig, axes = plt.subplots(1,2, sharey=True)
sns.histplot(data=task_summary, x="win_stay", ax=axes[0]) 
sns.histplot(data=task_summary['win_stay_transformed'], ax=axes[1])
print('Win-stay skew: '+str(skew(task_summary.win_stay)))
```

</details>

    Win-stay skew: -1.4392950153974455

<img src="model3_winStay_files/figure-commonmark/skewness-output-2.png"
id="skewness" />

<h3>

<b>Hypothesis testing</b>
</h3>

In this case, the basic model (no random slopes or random intercepts)
with an age covariate produced the best fit (indexed by BIC scores).
<p>

The model shows no effect of feedback type, and a significant effect of
age.
</p>

``` python
data=task_summary
formula = 'win_stay_transformed ~ block_type + prolific_age'
results=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
print(results.summary())
```

                  Mixed Linear Model Regression Results
    ==================================================================
    Model:            MixedLM Dependent Variable: win_stay_transformed
    No. Observations: 1020    Method:             ML                  
    No. Groups:       340     Scale:              1039.4970           
    Min. group size:  3       Log-Likelihood:     -5305.6971          
    Max. group size:  3       Converged:          Yes                 
    Mean group size:  3.0                                             
    ------------------------------------------------------------------
                          Coef.   Std.Err.   z    P>|z| [0.025  0.975]
    ------------------------------------------------------------------
    Intercept              89.746    8.173 10.981 0.000 73.728 105.764
    block_type[T.Fear]     -1.240    2.473 -0.501 0.616 -6.086   3.607
    block_type[T.Points]   -1.491    2.473 -0.603 0.547 -6.337   3.356
    prolific_age            0.507    0.170  2.976 0.003  0.173   0.840
    Group Var            1872.375    6.465                            
    ==================================================================

<p>

And the model assumptions are not violated
</p>

``` python
#shapiro-Wilk test of normality of residuals
labels = ["Statistic", "p-value"]
norm_res = stats.shapiro(results.resid)
for key, val in dict(zip(labels, norm_res)).items():
    print(key, val)
```

    Statistic 0.9975892480030629
    p-value 0.13936987905953396

``` python
##homoskedasticity of variance 
#White Lagrange Multiplier Test for Heteroscedasticity
het_white_res = het_white(results.resid, results.model.exog)

labels = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]

for key, val in dict(zip(labels, het_white_res)).items():
    print(key, val)
```

    LM Statistic 4.33424895515879
    LM-Test p-value 0.6315416360898078
    F-Statistic 0.7204788562016422
    F-Test p-value 0.6331641085900357

<p>

And the results remain unchanged when the age covariate is dropped:
</p>

``` python
formula = 'win_stay_transformed ~ block_type'
results=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
print(results.summary())
```

                   Mixed Linear Model Regression Results
    ===================================================================
    Model:             MixedLM Dependent Variable: win_stay_transformed
    No. Observations:  1020    Method:             ML                  
    No. Groups:        340     Scale:              1039.4992           
    Min. group size:   3       Log-Likelihood:     -5310.0694          
    Max. group size:   3       Converged:          Yes                 
    Mean group size:   3.0                                             
    -------------------------------------------------------------------
                          Coef.   Std.Err.   z    P>|z|  [0.025  0.975]
    -------------------------------------------------------------------
    Intercept             112.458    2.955 38.052 0.000 106.665 118.250
    block_type[T.Fear]     -1.240    2.473 -0.501 0.616  -6.086   3.607
    block_type[T.Points]   -1.491    2.473 -0.603 0.547  -6.337   3.356
    Group Var            1930.173    6.633                             
    ===================================================================

<br>
<p>

As this hypothesis test found a no difference between fear and disgust
or disgust and points, we will compute a Bayes Factor to test the
strength of the evidence for the null
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

<p>

Firstly for disgust vs fear:
</p>

``` python
print(bayes_factor(task_summary, 'win_stay_transformed', 'Disgust', 'Fear'))
```

    14.705882352941176

<p>

This means that there is <b>strong</b> evidence for the null
</p>

<br>
<p>

Next for disgust vs points:
</p>

``` python
print(bayes_factor(task_summary, 'win_stay_transformed', 'Disgust', 'Points'))
```

    13.698630136986303

<p>

This means that there is <b>strong</b> evidence for the null
</p>

<br>
<p>

We also look at fear vs points (which is not directly assessed by the
model)
</p>

``` python
print(bayes_factor(task_summary, 'win_stay_transformed', 'Points', 'Fear'))
```

    16.39344262295082

<p>

This means that there is <b>strong</b> evidence for the null
</p>

<br> <br>
<p>

<b>Next, we showed that this result is unchanged by the addition of
video-rating covariates.</b>
</p>

<p>

(again, the model with no additional random effects/slopes, with an age
covariate produced the best fit)
</p>

``` python
formula = 'win_stay_transformed ~ block_type + valence_diff + arousal_diff + valence_habdiff + prolific_age'
results=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
print(results.summary())
```

                  Mixed Linear Model Regression Results
    ==================================================================
    Model:            MixedLM Dependent Variable: win_stay_transformed
    No. Observations: 1020    Method:             ML                  
    No. Groups:       340     Scale:              1039.4968           
    Min. group size:  3       Log-Likelihood:     -5305.3916          
    Max. group size:  3       Converged:          Yes                 
    Mean group size:  3.0                                             
    ------------------------------------------------------------------
                          Coef.   Std.Err.   z    P>|z| [0.025  0.975]
    ------------------------------------------------------------------
    Intercept              91.475    8.465 10.807 0.000 74.884 108.065
    block_type[T.Fear]     -1.240    2.473 -0.501 0.616 -6.086   3.607
    block_type[T.Points]   -1.491    2.473 -0.603 0.547 -6.337   3.356
    valence_diff            1.262    1.639  0.770 0.441 -1.951   4.475
    arousal_diff           -0.074    2.197 -0.034 0.973 -4.380   4.232
    valence_habdiff        -0.219    1.247 -0.175 0.861 -2.663   2.225
    prolific_age            0.493    0.172  2.866 0.004  0.156   0.831
    Group Var            1868.392    6.453                            
    ==================================================================

<p>

Model assumptions are not violated
</p>

``` python
#shapiro-Wilk test of normality of residuals
labels = ["Statistic", "p-value"]
norm_res = stats.shapiro(results.resid)
for key, val in dict(zip(labels, norm_res)).items():
    print(key, val)
```

    Statistic 0.9975463051132777
    p-value 0.1298893515587406

``` python
##homoskedasticity of variance 
#White Lagrange Multiplier Test for Heteroscedasticity
het_white_res = het_white(results.resid, results.model.exog)

labels = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]

for key, val in dict(zip(labels, het_white_res)).items():
    print(key, val)
```

    LM Statistic 14.667815295792781
    LM-Test p-value 0.9301030557168437
    F-Statistic 0.6048778553564984
    F-Test p-value 0.9326863876596916

<p>

And the results remain unchanged when the age covariate is dropped:
</p>

``` python
formula = 'win_stay_transformed ~ block_type + valence_diff + arousal_diff + valence_habdiff'
results=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
print(results.summary())
```

                   Mixed Linear Model Regression Results
    ===================================================================
    Model:             MixedLM Dependent Variable: win_stay_transformed
    No. Observations:  1020    Method:             ML                  
    No. Groups:        340     Scale:              1039.4988           
    Min. group size:   3       Log-Likelihood:     -5309.4500          
    Max. group size:   3       Converged:          Yes                 
    Mean group size:   3.0                                             
    -------------------------------------------------------------------
                          Coef.   Std.Err.   z    P>|z|  [0.025  0.975]
    -------------------------------------------------------------------
    Intercept             113.594    3.512 32.345 0.000 106.711 120.477
    block_type[T.Fear]     -1.240    2.473 -0.501 0.616  -6.086   3.607
    block_type[T.Points]   -1.491    2.473 -0.603 0.547  -6.337   3.356
    valence_diff            1.624    1.654  0.982 0.326  -1.618   4.866
    arousal_diff            0.560    2.212  0.253 0.800  -3.775   4.896
    valence_habdiff        -0.063    1.261 -0.050 0.960  -2.534   2.407
    Group Var            1921.895    6.609                             
    ===================================================================

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
    lower_bound = Q1- 1.5 *  IQR
    upper_bound = Q3 + 1.5 *  IQR
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

Assess and correct for skewness in win-stay outcome (excluding outliers)
</h3>

<details class="code-fold">
<summary>Code</summary>

``` python
pt=PowerTransformer(method='yeo-johnson', standardize=False)
skl_yeojohnson=pt.fit(pd.DataFrame(sensitivity_df.win_stay))
skl_yeojohnson=pt.transform(pd.DataFrame(sensitivity_df.win_stay))
sensitivity_df['win_stay_transformed'] = pt.transform(pd.DataFrame(sensitivity_df.win_stay))

fig, axes = plt.subplots(1,2, sharey=True)
sns.histplot(data=sensitivity_df['win_stay'], ax=axes[0])
sns.histplot(data=sensitivity_df['win_stay_transformed'], ax=axes[1])
print('Win-stay skew: '+str(skew(sensitivity_df.win_stay.dropna())))
```

</details>

    Win-stay skew: -1.1512521637523532

![](model3_winStay_files/figure-commonmark/cell-18-output-2.png)

<h3>

<b>Outlier-free hypothesis testing</b>
</h3>

In this case, the basic model (no random slopes or random intercepts)
with just the age covariate produced the best fit (indexed by BIC
scores).
<p>

The model shows no effect of feedback type but found an effect of age
(as with the original analysis)
</p>

``` python
data=sensitivity_df.reset_index()
formula = 'win_stay_transformed ~ block_type +prolific_age'
results=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
print(results.summary())
```

                   Mixed Linear Model Regression Results
    ===================================================================
    Model:             MixedLM Dependent Variable: win_stay_transformed
    No. Observations:  955     Method:             ML                  
    No. Groups:        334     Scale:              6992.8863           
    Min. group size:   1       Log-Likelihood:     -5842.1169          
    Max. group size:   3       Converged:          Yes                 
    Mean group size:   2.9                                             
    -------------------------------------------------------------------
                          Coef.   Std.Err.   z    P>|z|  [0.025  0.975]
    -------------------------------------------------------------------
    Intercept             220.523   19.222 11.472 0.000 182.849 258.198
    block_type[T.Fear]     -1.130    6.690 -0.169 0.866 -14.243  11.983
    block_type[T.Points]   -5.629    6.673 -0.843 0.399 -18.708   7.450
    prolific_age            1.231    0.397  3.098 0.002   0.452   2.009
    Group Var            9276.049   13.870                             
    ===================================================================

<p>

The assumptions are not violated for this model
</p>

``` python
#shapiro-Wilk test of normality of residuals
labels = ["Statistic", "p-value"]
norm_res = stats.shapiro(results.resid)
for key, val in dict(zip(labels, norm_res)).items():
    print(key, val)
```

    Statistic 0.996929605796315
    p-value 0.06302663951601054

``` python
##homoskedasticity of variance 
#White Lagrange Multiplier Test for Heteroscedasticity
het_white_res = het_white(results.resid, results.model.exog)

labels = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]

for key, val in dict(zip(labels, het_white_res)).items():
    print(key, val)
```

    LM Statistic 1.2916633127210986
    LM-Test p-value 0.9721154055677161
    F-Statistic 0.2139886960816659
    F-Test p-value 0.9724366577776193

<p>

And the results are unchanged when the age covariate is dropped
</p>

``` python
data=sensitivity_df.reset_index()
formula = 'win_stay_transformed ~ block_type'
results=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
print(results.summary())
```

                   Mixed Linear Model Regression Results
    ===================================================================
    Model:             MixedLM Dependent Variable: win_stay_transformed
    No. Observations:  955     Method:             ML                  
    No. Groups:        334     Scale:              7003.2210           
    Min. group size:   1       Log-Likelihood:     -5846.8628          
    Max. group size:   3       Converged:          Yes                 
    Mean group size:   2.9                                             
    -------------------------------------------------------------------
                          Coef.   Std.Err.   z    P>|z|  [0.025  0.975]
    -------------------------------------------------------------------
    Intercept             275.798    7.160 38.519 0.000 261.765 289.831
    block_type[T.Fear]     -1.305    6.696 -0.195 0.845 -14.428  11.819
    block_type[T.Points]   -5.676    6.679 -0.850 0.395 -18.766   7.414
    Group Var            9579.353   14.266                             
    ===================================================================

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
print(bayes_factor(sensitivity_df, 'win_stay_transformed', 'Disgust', 'Fear'))
```

    15.384615384615383

<p>

This means that there is <b>strong</b> evidence for the null
</p>

<br>
<p>

Next for disgust vs points:
</p>

``` python
print(bayes_factor(sensitivity_df, 'win_stay_transformed', 'Disgust', 'Points'))
```

    11.76470588235294

<p>

This means that there is <b>strong</b> evidence for the null
</p>

<br>
<p>

We also look at fear vs points (which is not directly assessed by the
model)
</p>

``` python
print(bayes_factor(sensitivity_df, 'win_stay_transformed', 'Points', 'Fear'))
```

    11.363636363636365

<p>

This means that there is <b>strong</b> evidence for the null
</p>

<br>
<p>

<b>Finally, adding video rating values to this model has no effect: </b>
</p>

(NB model with the age covariate, again, produced the best fit)

``` python
data=sensitivity_df.reset_index()
formula = 'win_stay_transformed ~ block_type + valence_diff + arousal_diff + valence_habdiff + prolific_age'
results=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
print(results.summary())
```

                   Mixed Linear Model Regression Results
    ===================================================================
    Model:             MixedLM Dependent Variable: win_stay_transformed
    No. Observations:  955     Method:             ML                  
    No. Groups:        334     Scale:              6993.6163           
    Min. group size:   1       Log-Likelihood:     -5841.5752          
    Max. group size:   3       Converged:          Yes                 
    Mean group size:   2.9                                             
    -------------------------------------------------------------------
                          Coef.   Std.Err.   z    P>|z|  [0.025  0.975]
    -------------------------------------------------------------------
    Intercept             225.724   19.916 11.334 0.000 186.690 264.758
    block_type[T.Fear]     -1.065    6.691 -0.159 0.874 -14.179  12.050
    block_type[T.Points]   -5.572    6.674 -0.835 0.404 -18.652   7.509
    valence_diff            3.665    3.832  0.957 0.339  -3.845  11.175
    arousal_diff           -2.674    5.169 -0.517 0.605 -12.805   7.458
    valence_habdiff        -1.633    2.957 -0.552 0.581  -7.428   4.162
    prolific_age            1.217    0.401  3.036 0.002   0.431   2.002
    Group Var            9235.265   13.825                             
    ===================================================================

<p>

The assumptions for this test are not violated:
</p>

``` python
#shapiro-Wilk test of normality of residuals
labels = ["Statistic", "p-value"]
norm_res = stats.shapiro(results.resid)
for key, val in dict(zip(labels, norm_res)).items():
    print(key, val)
```

    Statistic 0.9968174179331105
    p-value 0.0527714290007855

``` python
##homoskedasticity of variance 
#White Lagrange Multiplier Test for Heteroscedasticity
het_white_res = het_white(results.resid, results.model.exog)

labels = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]

for key, val in dict(zip(labels, het_white_res)).items():
    print(key, val)
```

    LM Statistic 16.189293421674396
    LM-Test p-value 0.8811225232117103
    F-Statistic 0.6682232272108647
    F-Test p-value 0.8848727181148472

<p>

And this remains unchanged when the age covariate is removed:
</p>

``` python
data=sensitivity_df.reset_index()
formula = 'win_stay_transformed ~ block_type + valence_diff + arousal_diff + valence_habdiff'
results=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
print(results.summary())
```

                   Mixed Linear Model Regression Results
    ===================================================================
    Model:             MixedLM Dependent Variable: win_stay_transformed
    No. Observations:  955     Method:             ML                  
    No. Groups:        334     Scale:              7002.9994           
    Min. group size:   1       Log-Likelihood:     -5846.1342          
    Max. group size:   3       Converged:          Yes                 
    Mean group size:   2.9                                             
    -------------------------------------------------------------------
                          Coef.   Std.Err.   z    P>|z|  [0.025  0.975]
    -------------------------------------------------------------------
    Intercept             280.540    8.438 33.249 0.000 264.003 297.078
    block_type[T.Fear]     -1.246    6.696 -0.186 0.852 -14.370  11.878
    block_type[T.Points]   -5.621    6.679 -0.842 0.400 -18.712   7.469
    valence_diff            4.667    3.866  1.207 0.227  -2.910  12.243
    arousal_diff           -1.314    5.215 -0.252 0.801 -11.535   8.907
    valence_habdiff        -1.330    2.992 -0.445 0.657  -7.194   4.534
    Group Var            9527.405   14.205                             
    ===================================================================

<h4>

<b>Exploratory analyses</b>
</h4>

<p>

- Effect of age
- Points ratings
- error rates

</p>

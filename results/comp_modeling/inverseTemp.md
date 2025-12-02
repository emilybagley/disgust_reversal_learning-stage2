# Computational modeling hypothesis testing (inverse temperature)


<p>

This file contains hypothesis testing carried out on the winning model
(1LR_stick1_allparamsep) using the <b>inverse temperature</b>
parameters.
<p>

<p>

Inverse temperature parameters are compared between the three feedback
types using the same mixed effects modeling strategy as was used in the
model-agnostic hypothesis testing analyses (including skew transforms,
assumptions testing, generalized models for failed model assumptions,
video-ratings covariates, sensitivity analyses, bayes factors for null
results, and BIC model copmarison as a means of selecting random effects
and covariates).

<br>
<h3>

Load in packages and data- in python and then in r
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
from scipy.stats import ttest_1samp
import itertools

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.copy_on_write = True

filepath="//cbsu/data/Group/Nord/DisgustReversalLearningModeling/finalModelComp/1LR_stick1_blk3_allparamsep_params.csv"
params = pd.read_csv(filepath)
task_summary=pd.read_csv('U:/Documents/Disgust learning project/github/disgust_reversal_learning-final/csvs/dem_vids_task_excluded.csv')
task_summary.sort_values(by=['participant_no', 'block_type'], inplace=True)
params['participant_no']=list(set(task_summary.participant_no))

#convert to long df
long_params=pd.DataFrame()
for subj in set(params['participant_no']):
    subj_params= params[params['participant_no']==subj]
    disgust_row=pd.DataFrame({
        'participant_no': [float(subj_params['participant_no'])],
        'LR':  [float(subj_params['d_alpha'])],
        'invTemp': [float(subj_params['d_beta'])],
        'stickiness': [float(subj_params['d_omega'])],
        'block_type': ['Disgust']
    })
    fear_row=pd.DataFrame({
        'participant_no': [float(subj_params['participant_no'])],
        'LR':  [float(subj_params['f_alpha'])],
        'invTemp': [float(subj_params['f_beta'])],
        'stickiness': [float(subj_params['f_omega'])],
        'block_type': ['Fear']
    })
    points_row=pd.DataFrame({
        'participant_no': [float(subj_params['participant_no'])],
        'LR':  [float(subj_params['p_alpha'])],
        'invTemp': [float(subj_params['p_beta'])],
        'stickiness': [float(subj_params['p_omega'])],
        'block_type': ['Points']
    })
    long_params=pd.concat([long_params, disgust_row, fear_row, points_row])

##combine with task_summary_df
df=pd.merge(task_summary, long_params, on=['participant_no', 'block_type'], how='inner')
df.to_csv("winningModelOutput.csv")

pvals_file = 'pvals/ModelingPvalsForPlotting.xlsx'
```

</details>

<details class="code-fold">
<summary>Code</summary>

``` r
library(tidyverse, quietly=TRUE)
library(lme4)
library(emmeans)
library(DHARMa)
library('readxl')
library('xlsx')

df <- read.csv("winningModelOutput.csv")
pvals_file <-'pvals/ModelingPvalsForPlotting.xlsx'
```

</details>

<b>Assess and correct for skewness</b>

<details class="code-fold">
<summary>Code</summary>

``` python
pt=PowerTransformer(method='yeo-johnson', standardize=False)
skl_yeojohnson=pt.fit(pd.DataFrame(df.invTemp))
skl_yeojohnson=pt.transform(pd.DataFrame(df.invTemp))
df['invTemp_transformed'] = pt.transform(pd.DataFrame(df.invTemp))

fig, axes = plt.subplots(1,2, sharey=True)
sns.histplot(data=df, x="invTemp", ax=axes[0]) 
sns.histplot(data=df['invTemp_transformed'], ax=axes[1])
print('invTemp skew: '+str(skew(df.invTemp)))
```

</details>

    invTemp skew: 0.8591997855930473

![](inverseTemp_files/figure-commonmark/Skewness%20invTemp-1.jpeg)

<br>

<b>Mixed effects model assumptions violated</b>
<p>

In this case, a basic model (no random slopes or random intercepts) with
an age covariate produced the best fit (as indexed by BIC scores). But
the model assumptions were violated:

<p>

Select the winning mixed effects model:
</p>

<details class="code-fold">
<summary>Code</summary>

``` python
#inverse temperature
data=df.reset_index()
formula = 'invTemp_transformed ~ block_type'
basic_model=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
#basic_model.summary()

#test which random effects to include
#feedback_randint=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', vc_formula={'feedback_details': '0+feedback_details'}).fit(reml=False) 
#fractals_randint=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', vc_formula={'fractals': '0+fractals'}).fit(reml=False) 
feedback_fractals_randint=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', vc_formula={"feedback_details": "0 + feedback_details", "fractals": "0 + fractals"}).fit(reml=False)

randslope=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', re_formula='~block_type').fit(reml=False)
feedback_randint_randslope=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', vc_formula={'feedback_details': '0+feedback_details'}, re_formula='~block_type').fit(reml=False)
#feedback_fractals_randint_randslope=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', vc_formula={'feedback_details': '0+feedback_details', "fractals": "0 + fractals"}, re_formula='~block_type').fit(reml=False)


bic=pd.DataFrame({'basic_model': [basic_model.bic], 
                    #'feedback_randint': [feedback_randint.bic], 
                    #'fractals_randint': [fractals_randint.bic],
                    'feedback_fractals_randint': [feedback_fractals_randint.bic], ##added manually
                    'randslope': [randslope.bic],
                    'feedback_randint_randslope':[feedback_randint_randslope.bic],
                    #'feedback_fractals_randint_randslope': [feedback_fractals_randint_randslope.bic]
                    })
win1=bic.sort_values(by=0, axis=1).columns[0]

##test which covariates to add -- Using the random effects which were best above (basic model in this case)
no_covariate=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
sex_covariate=smf.mixedlm(formula+str('+prolific_sex'), data, groups=data['participant_no'], missing='drop').fit(reml=False)
age_covariate=smf.mixedlm(formula+str('+prolific_age'), data, groups=data['participant_no'], missing='drop').fit(reml=False)
digit_span_covariate=smf.mixedlm(formula+str('+digit_span'), data, groups=data['participant_no'], missing='drop').fit(reml=False)
sex_age_covariate=smf.mixedlm(formula+str('+prolific_sex+prolific_age'), data, groups=data['participant_no'], missing='drop').fit(reml=False)
sex_digit_span_covariate=smf.mixedlm(formula+str('+prolific_sex+digit_span'), data, groups=data['participant_no'], missing='drop').fit(reml=False)
digit_span_age_covariate=smf.mixedlm(formula+str('+digit_span+prolific_age'), data, groups=data['participant_no'], missing='drop').fit(reml=False)
sex_age_digit_span_covariate=smf.mixedlm(formula+str('+prolific_sex+prolific_age+digit_span'), data, groups=data['participant_no'], missing='drop').fit(reml=False)

bic=pd.DataFrame({'no_covariate': [no_covariate.bic], 
                    'sex_covariate': [sex_covariate.bic], 
                    'age_covariate': [age_covariate.bic],
                    'digit_span_covariate': [digit_span_covariate.bic],
                    'sex_age_covariate': [sex_age_covariate.bic],
                    'sex_digit_span_covariate': [sex_digit_span_covariate.bic],
                    'digit_span_age_covariate': [digit_span_age_covariate.bic],
                    'sex_age_digit_span_covariate': [sex_age_digit_span_covariate.bic]})
win2=bic.sort_values(by=0, axis=1).columns[0]
print("Winning models: "+ win1 +" "+ win2)
```

</details>

    Winning models: basic_model age_covariate

<p>

Shapiro-Wilk test of normality of residuals
</p>

<details class="code-fold">
<summary>Code</summary>

``` python
#chosen model
results=age_covariate

#shapiro-Wilk test of normality of residuals
labels = ["Statistic", "p-value"]
norm_res = stats.shapiro(results.resid)

for key, val in dict(zip(labels, norm_res)).items():
    print(key, val)
```

</details>

    Statistic 0.9897122304204978
    p-value 1.438712886650115e-06

<p>

White Lagrange multiplier Test for Heteroscedasticity
</p>

<details class="code-fold">
<summary>Code</summary>

``` python
#chosen model
##homoskedasticity of variance 
#White Lagrange Multiplier Test for Heteroscedasticity
het_white_res = het_white(results.resid, results.model.exog)

labels = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]

for key, val in dict(zip(labels, het_white_res)).items():
    print(key, val)
```

</details>

    LM Statistic 8.44107447998116
    LM-Test p-value 0.20753629085886294
    F-Statistic 1.4088499497323073
    F-Test p-value 0.2079586941021339

<h4>

<b>So instead we run a generalized mixed effects model (done in R)</b>
</h4>

<details class="code-fold">
<summary>Code</summary>

``` r
#inv Temp model
gamma_log <- glmer(invTemp~ block_type + (1|participant_no), data=df, family=Gamma(link="log"))
gamma_inverse <- glmer(invTemp~ block_type + (1|participant_no), data=df, family=Gamma(link="inverse"))
gamma_identity <- glmer(invTemp~ block_type + (1|participant_no), data=df, family=Gamma(link="identity"))

invgaus_log <- glmer(invTemp~ block_type + (1|participant_no), data=df, family=inverse.gaussian(link="log"))
invgaus_inverse <- glmer(invTemp~ block_type + (1|participant_no), data=df, family=inverse.gaussian(link="inverse"))
invgaus_identity <- glmer(invTemp~ block_type + (1|participant_no), data=df, family=inverse.gaussian(link="identity"))

bic_values <- c(
  BIC(gamma_log),
  BIC(gamma_inverse),
  BIC(gamma_identity),
  BIC(invgaus_log),
  BIC(invgaus_inverse),
  BIC(invgaus_identity)
)
model_names <- c("Gamma (log)", "Gamma (inverse)", "Gamma (identity)", "Inverse gaussian (log)", "Inverse gaussian (inverse)", "Inverse gaussian (identity)")

bic_df <- data.frame(Model = model_names, BIC = bic_values)
win1 <- bic_df[which.min(bic_df$BIC), ]$Model

basic_model <- glmer(invTemp~ block_type + (1|participant_no), data=df, family=Gamma(link="identity"))

feedback_randint <- glmer(invTemp~ block_type + (1|participant_no) + (1|feedback_details), data=df, family=Gamma(link="identity"))
#fractals_randint <- glmer(invTemp~ block_type + (1|participant_no) + (1|fractals), data=df, family=Gamma(link="identity"))
#feedback_fractals_randint <- glmer(invTemp~ block_type + (1|participant_no) + (1|fractals) + (1|feedback_details), data=df, family=Gamma(link="identity"))

#randslope <- glmer(invTemp~ block_type + (block_type|participant_no), data=df, family=Gamma(link="identity"))
#feedback_randint_randslope <- glmer(invTemp~ block_type + (block_type|participant_no) + (1|feedback_details), data=df, family=Gamma(link="identity"))
#feedback_fractals_randint_randslope <- glmer(invTemp~ block_type + (block_type|participant_no) + (1|feedback_details) + (1|fractals), data=df, family=Gamma(link="identity"))
bic_values <- c(
  BIC(basic_model),
  BIC(feedback_randint)
  #BIC(fractals_randint),
  #BIC(feedback_fractals_randint),
  #BIC(randslope),
  #BIC(feedback_randint_randslope)
  #BIC(feedback_fractals_randint_randslope)
)
model_names <- c("basic model", "feedback_randint")

bic_df <- data.frame(Model = model_names, BIC = bic_values)
win2 <- bic_df[which.min(bic_df$BIC), ]$Model

#choose covariate structure for basic model
no_covariate <- basic_model
sex_covariate <- glmer(invTemp ~ block_type + (1|participant_no) + prolific_sex, data=df, family=Gamma(link="identity"))
#age_covariate <- glmer(invTemp ~ block_type + (1|participant_no) + prolific_age, data=df, family=Gamma(link="identity"))
digit_span_covariate <- glmer(invTemp ~ block_type + (1|participant_no) + digit_span, data=df, family=Gamma(link="identity"))
#sex_age_covariate <- glmer(invTemp ~ block_type + (1|participant_no) + prolific_sex + prolific_age, data=df, family=Gamma(link="identity"))
sex_digit_span_covariate <- glmer(invTemp ~ block_type + (1|participant_no) + prolific_sex + digit_span, data=df, family=Gamma(link="identity"))
#digit_span_age_covariate <- glmer(invTemp ~ block_type + (1|participant_no) + prolific_age + digit_span, data=df, family=Gamma(link="identity"))
#sex_digit_span_age_covariate <- glmer(invTemp ~ block_type + (1|participant_no) + prolific_age + prolific_sex + digit_span, data=df, family=Gamma(link="identity"))

bic_values <- c(
  BIC(no_covariate),
  BIC(sex_covariate),
  BIC(digit_span_covariate),
  BIC(sex_digit_span_covariate)
)
model_names <- c("no_covariate", "sex_covariate", "digit_span_covariate", "sex_digit_span_covariate")

bic_df <- data.frame(Model = model_names, BIC = bic_values)
win3 <- bic_df[which.min(bic_df$BIC), ]$Model

print(paste0("Winning models: ", win1, " ", win2," ",win3))
```

</details>

    [1] "Winning models: Gamma (identity) basic model no_covariate"

<p>

Results from this model show no <b>significant effect of block-type</b>
</p>

``` r
summary(no_covariate)
```

    Generalized linear mixed model fit by maximum likelihood (Laplace
      Approximation) [glmerMod]
     Family: Gamma  ( identity )
    Formula: invTemp ~ block_type + (1 | participant_no)
       Data: df

         AIC      BIC   logLik deviance df.resid 
      2160.1   2184.7  -1075.0   2150.1     1015 

    Scaled residuals: 
         Min       1Q   Median       3Q      Max 
    -1.68620 -0.53014 -0.05218  0.52842  2.88155 

    Random effects:
     Groups         Name        Variance Std.Dev.
     participant_no (Intercept) 0.3796   0.6161  
     Residual                   0.2882   0.5368  
    Number of obs: 1020, groups:  participant_no, 340

    Fixed effects:
                      Estimate Std. Error t value Pr(>|z|)    
    (Intercept)       1.649548   0.056613  29.137   <2e-16 ***
    block_typeFear   -0.017525   0.021078  -0.831    0.406    
    block_typePoints  0.006681   0.024278   0.275    0.783    
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Correlation of Fixed Effects:
                (Intr) blck_F
    block_typFr -0.203       
    blck_typPnt -0.212  0.495

<p>

Extract confidence intervals
</p>

``` r
print(confint.merMod(no_covariate, method='Wald'))
```

                           2.5 %     97.5 %
    .sig01                    NA         NA
    .sigma                    NA         NA
    (Intercept)       1.53858851 1.76050756
    block_typeFear   -0.05883750 0.02378696
    block_typePoints -0.04090394 0.05426557

<p>

As this hypothesis test found a no difference between fear and disgust
or points and disgust, we will compute Bayes Factors to test the
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
    return ttest, bf_null
```

</details>

<details class="code-fold">
<summary>Code</summary>

``` python
ttest, bf_null = bayes_factor(df, 'invTemp', 'Disgust', 'Points')

print(f"Disgust vs Points: BF01 = {bf_null}")
```

</details>

    Disgust vs Points: BF01 = 3.2051282051282053

<details class="code-fold">
<summary>Code</summary>

``` python
#print(f"Disgust vs Points: T = {ttest['T'][0]}, CI95% = {ttest['CI95%'][0]}, p = {ttest['p-val'][0]}")
```

</details>

<details class="code-fold">
<summary>Code</summary>

``` python
ttest, bf_null = bayes_factor(df, 'invTemp', 'Disgust', 'Fear')

print(f"Disgust vs Fear: BF01 = {bf_null}")
```

</details>

    Disgust vs Fear: BF01 = 12.5

<p>

We also look at fear vs points (which is not directly assessed by the
model)
</p>

<details class="code-fold">
<summary>Code</summary>

``` python
ttest, bf_null = bayes_factor(df, 'invTemp', 'Points', 'Fear')

print(f"Points vs Fear: T = {ttest['T'][0]}, CI95% = {ttest['CI95%'][0]}, p = {ttest['p-val'][0]}")
```

</details>

    Points vs Fear: T = 1.0137354501695723, CI95% = [-0.05  0.17], p = 0.31143197844956166

<p>

And because the result is null, also get a Bayes factor:
</p>

<details class="code-fold">
<summary>Code</summary>

``` python
print(f"Points vs Fear: BF01 = {bf_null}")
```

</details>

    Points vs Fear: BF01 = 9.900990099009901

    U:\Documents\envs\disgust_reversal_venv\Lib\site-packages\openpyxl\styles\stylesheet.py:237: UserWarning: Workbook contains no default style, apply openpyxl's default
      warn("Workbook contains no default style, apply openpyxl's default")

<br>
<h3>

<b>Adding video ratings</b>
</h3>

We will next test whether this effect remains after video rating
differences between fear and disgust have been controlled for.
<p>

As before, the mixed effects model violated assumptions, so a
generalized mixed effects model must be run.
</p>

<details class="code-fold">
<summary>Code</summary>

``` python
#run basic model 
data=df.reset_index()
formula = 'invTemp_transformed ~ block_type +valence_diff + arousal_diff + valence_habdiff'
basic_model=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)

#test which random effects to include
#feedback_randint=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', vc_formula={'feedback_details': '0+feedback_details'}).fit(reml=False) 
#fractals_randint=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', vc_formula={'fractals': '0+fractals'}).fit(reml=False) 
#feedback_fractals_randint=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', vc_formula={"feedback_details": "0 + feedback_details", "fractals": "0 + fractals"}).fit(reml=False)

randslope=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', re_formula='~block_type').fit(reml=False)
feedback_randint_randslope=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', vc_formula={'feedback_details': '0+feedback_details'}, re_formula='~block_type').fit(reml=False)
#feedback_fractals_randint_randslope=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', vc_formula={'feedback_details': '0+feedback_details', "fractals": "0 + fractals"}, re_formula='~block_type').fit(reml=False)


bic=pd.DataFrame({'basic_model': [basic_model.bic], 
                    #'feedback_randint': [feedback_randint.bic], 
                    #'fractals_randint': [fractals_randint.bic],
                    'feedback_fractals_randint': [feedback_fractals_randint.bic], ##added manually
                    'randslope': [randslope.bic],
                    'feedback_randint_randslope':[feedback_randint_randslope.bic],
                    #'feedback_fractals_randint_randslope': [feedback_fractals_randint_randslope.bic]
                    })
win1=bic.sort_values(by=0, axis=1).columns[0]

no_covariate=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
sex_covariate=smf.mixedlm(formula+str('+prolific_sex'), data, groups=data['participant_no'], missing='drop').fit(reml=False)
age_covariate=smf.mixedlm(formula+str('+prolific_age'), data, groups=data['participant_no'], missing='drop').fit(reml=False)
digit_span_covariate=smf.mixedlm(formula+str('+digit_span'), data, groups=data['participant_no'], missing='drop').fit(reml=False)
sex_age_covariate=smf.mixedlm(formula+str('+prolific_sex+prolific_age'), data, groups=data['participant_no'], missing='drop').fit(reml=False)
sex_digit_span_covariate=smf.mixedlm(formula+str('+prolific_sex+digit_span'), data, groups=data['participant_no'], missing='drop').fit(reml=False)
digit_span_age_covariate=smf.mixedlm(formula+str('+digit_span+prolific_age'), data, groups=data['participant_no'], missing='drop').fit(reml=False)
sex_age_digit_span_covariate=smf.mixedlm(formula+str('+prolific_sex+prolific_age+digit_span'), data, groups=data['participant_no'], missing='drop').fit(reml=False)

bic=pd.DataFrame({'no_covariate': [no_covariate.bic], 
                    'sex_covariate': [sex_covariate.bic], 
                    'age_covariate': [age_covariate.bic],
                    'digit_span_covariate': [digit_span_covariate.bic],
                    'sex_age_covariate': [sex_age_covariate.bic],
                    'sex_digit_span_covariate': [sex_digit_span_covariate.bic],
                    'digit_span_age_covariate': [digit_span_age_covariate.bic],
                    'sex_age_digit_span_covariate': [sex_age_digit_span_covariate.bic]})

win2=bic.sort_values(by=0, axis=1).columns[0]
print("Winning models: "+ win1 +" "+ win2)
```

</details>

    Winning models: basic_model age_covariate

<p>

Shapiro-Wilk test of normality of residuals
</p>

<details class="code-fold">
<summary>Code</summary>

``` python
results=age_covariate
labels = ["Statistic", "p-value"]
norm_res = stats.shapiro(results.resid)

for key, val in dict(zip(labels, norm_res)).items():
    print(key, val)
```

</details>

    Statistic 0.9897352997058386
    p-value 1.4782642125422706e-06

<p>

White Lagrange multiplier Test for Heteroscedasticity
</p>

<details class="code-fold">
<summary>Code</summary>

``` python
het_white_res = het_white(results.resid, results.model.exog)

labels = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]

for key, val in dict(zip(labels, het_white_res)).items():
    print(key, val)
```

</details>

    LM Statistic 20.946832806987924
    LM-Test p-value 0.6418597244172857
    F-Statistic 0.8692438053418764
    F-Test p-value 0.6461749021589878

<h4>

so we run a generalised mixed effects model
</h4>

<details class="code-fold">
<summary>Code</summary>

``` r
gamma_log <- glmer(invTemp~ block_type + valence_diff + arousal_diff + valence_habdiff  + (1|participant_no), data=df, family=Gamma(link="log"))
gamma_inverse <- glmer(invTemp~ block_type + valence_diff + arousal_diff + valence_habdiff  + (1|participant_no), data=df, family=Gamma(link="inverse"))
gamma_identity <- glmer(invTemp~ block_type + valence_diff + arousal_diff + valence_habdiff  + (1|participant_no), data=df, family=Gamma(link="identity"))

invgaus_log <- glmer(invTemp~ block_type + valence_diff + arousal_diff + valence_habdiff  + (1|participant_no), data=df, family=inverse.gaussian(link="log"))
invgaus_inverse <- glmer(invTemp~ block_type + valence_diff + arousal_diff + valence_habdiff  + (1|participant_no), data=df, family=inverse.gaussian(link="inverse"))
invgaus_identity <- glmer(invTemp~ block_type + valence_diff + arousal_diff + valence_habdiff  + (1|participant_no), data=df, family=inverse.gaussian(link="identity"))

bic_values <- c(
  BIC(gamma_log),
  BIC(gamma_inverse),
  BIC(gamma_identity),
  BIC(invgaus_log),
  BIC(invgaus_inverse),
  BIC(invgaus_identity)
)
model_names <- c("Gamma (log)", 
                "Gamma (inverse)", 
                "Gamma (identity)",
                 "Inverse gaussian (log)", 
                 "Inverse gaussian (inverse)", 
                 "Inverse gaussian (identity)")

bic_df <- data.frame(Model = model_names, BIC = bic_values)
win1 <- bic_df[which.min(bic_df$BIC), ]$Model

basic_model <- glmer(invTemp~ block_type + valence_diff + arousal_diff + valence_habdiff + (1|participant_no), data=df, family=Gamma(link="identity"))

feedback_randint <- glmer(invTemp~ block_type + valence_diff + arousal_diff + valence_habdiff + (1|participant_no) + (1|feedback_details), data=df, family=Gamma(link="identity"))
#fractals_randint <- glmer(invTemp~ block_type + valence_diff + arousal_diff + valence_habdiff + (1|participant_no) + (1|fractals), data=df, family=Gamma(link="identity"))
#feedback_fractals_randint <- glmer(invTemp~ block_type + valence_diff + arousal_diff + valence_habdiff + (1|participant_no) + (1|fractals) + (1|feedback_details), data=df, family=Gamma(link="identity"))

#randslope <- glmer(invTemp~ block_type + valence_diff + arousal_diff + valence_habdiff + (block_type + valence_diff + arousal_diff + valence_habdiff|participant_no), data=df, family=Gamma(link="identity"))
#feedback_randint_randslope <- glmer(invTemp~ block_type + valence_diff + arousal_diff + valence_habdiff + (block_type + valence_diff + arousal_diff + valence_habdiff|participant_no) + (1|feedback_details), data=df, family=Gamma(link="identity"))
#feedback_fractals_randint_randslope <- glmer(invTemp~ block_type + valence_diff + arousal_diff + valence_habdiff + (block_type + valence_diff + arousal_diff + valence_habdiff|participant_no) + (1|feedback_details) + (1|fractals), data=df, family=Gamma(link="identity"))

bic_values <- c(
  BIC(basic_model),
  BIC(feedback_randint)
 # BIC(fractals_randint),
  #BIC(feedback_fractals_randint)
  #BIC(randslope),
  #BIC(feedback_randint_randslope)
  #BIC(feedback_fractals_ randint_randslope)
)
model_names <- c("basic model", 
                "feedback_randint"
                #"fractals_randint", 
              #  "feedback_fractals_randint"
                #"randslope",
                #"feedback_randint_randslope",
                #"feedback_fractals_randint_randslope"
                )

bic_df <- data.frame(Model = model_names, BIC = bic_values)
win2 <- bic_df[which.min(bic_df$BIC), ]$Model

#choose covariates
no_covariate <- basic_model
sex_covariate <- glmer(invTemp ~ block_type + valence_diff + arousal_diff + valence_habdiff + (1|participant_no) + prolific_sex, data=df, family=Gamma(link="identity"))
#age_covariate <- glmer(invTemp ~ block_type + valence_diff + arousal_diff + valence_habdiff + (1|participant_no) + prolific_age, data=df, family=Gamma(link="identity"))
digit_span_covariate <- glmer(invTemp ~ block_type + valence_diff + arousal_diff + valence_habdiff + (1|participant_no) + digit_span, data=df, family=Gamma(link="identity"))
#sex_age_covariate <- glmer(invTemp ~ block_type + valence_diff + arousal_diff + valence_habdiff + (1|participant_no) + prolific_sex + prolific_age, data=df, family=Gamma(link="identity"))
#sex_digit_span_covariate <- glmer(invTemp ~ block_type + valence_diff + arousal_diff + valence_habdiff + (1|participant_no) + prolific_sex + digit_span, data=df, family=Gamma(link="identity"))
#digit_span_age_covariate <- glmer(invTemp ~ block_type + valence_diff + arousal_diff + valence_habdiff + (1|participant_no) + prolific_age + digit_span, data=df, family=Gamma(link="identity"))
#sex_digit_span_age_covariate <- glmer(invTemp ~ block_type + valence_diff + arousal_diff + valence_habdiff + (1|participant_no) + prolific_age + prolific_sex + digit_span, data=df, family=Gamma(link="identity"))

bic_values <- c(
  BIC(no_covariate),
  BIC(sex_covariate),
  BIC(digit_span_covariate)
)
model_names <- c("no_covariate",
                "sex_covariate",
                "digit_span_covariate"
                )

bic_df <- data.frame(Model = model_names, BIC = bic_values)
win3 <- bic_df[which.min(bic_df$BIC), ]$Model

print(paste0("Winning models: ", win1, " ", win2," ",win3))
```

</details>

    [1] "Winning models: Gamma (identity) basic model no_covariate"

<p>

Adding video ratings has <b> no effect </b> on the results

``` r
summary(no_covariate)
```

    Generalized linear mixed model fit by maximum likelihood (Laplace
      Approximation) [glmerMod]
     Family: Gamma  ( identity )
    Formula: 
    invTemp ~ block_type + valence_diff + arousal_diff + valence_habdiff +  
        (1 | participant_no)
       Data: df

         AIC      BIC   logLik deviance df.resid 
      2164.8   2204.2  -1074.4   2148.8     1012 

    Scaled residuals: 
         Min       1Q   Median       3Q      Max 
    -1.68845 -0.52382 -0.04632  0.53494  2.89824 

    Random effects:
     Groups         Name        Variance Std.Dev.
     participant_no (Intercept) 0.3780   0.6148  
     Residual                   0.2879   0.5366  
    Number of obs: 1020, groups:  participant_no, 340

    Fixed effects:
                      Estimate Std. Error t value Pr(>|z|)    
    (Intercept)       1.689151   0.067923  24.869   <2e-16 ***
    block_typeFear   -0.017530   0.021071  -0.832    0.405    
    block_typePoints  0.006782   0.024284   0.279    0.780    
    valence_diff      0.036441   0.032870   1.109    0.268    
    arousal_diff     -0.016705   0.043054  -0.388    0.698    
    valence_habdiff  -0.012329   0.024811  -0.497    0.619    
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Correlation of Fixed Effects:
                (Intr) blck_F blck_P vlnc_d arsl_d
    block_typFr -0.169                            
    blck_typPnt -0.175  0.494                     
    valence_dff  0.478 -0.001  0.002              
    arousal_dff -0.338  0.001  0.001 -0.178       
    valnc_hbdff -0.294 -0.008 -0.007 -0.342  0.148

<p>

Extract confidence intervals
</p>

``` r
print(confint.merMod(no_covariate, method='Wald'))
```

                           2.5 %     97.5 %
    .sig01                    NA         NA
    .sigma                    NA         NA
    (Intercept)       1.55602423 1.82227684
    block_typeFear   -0.05882787 0.02376758
    block_typePoints -0.04081366 0.05437759
    valence_diff     -0.02798195 0.10086425
    arousal_diff     -0.10108859 0.06767827
    valence_habdiff  -0.06095809 0.03629936

<br>
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
def replace_outliers_with_nan(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1- 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column]=df[column].apply(lambda x: np.nan if x<lower_bound or x>upper_bound else x)
    return df

key_outcomes=['LR', 'invTemp', 'stickiness']
for col in key_outcomes:
    df=replace_outliers_with_nan(df, col)

df.to_csv("sensitivity_winningModelOutput.csv")
```

</details>

<p>

In the case of the inverse temperature outcome, 19 datapoints are
excluded
</p>

``` python
print("Number of inverse temperature outliers: "+str(len(df[df.invTemp.isna()])))
```

    Number of inverse temperature outliers: 19

<p>

Assess and correct for skewness
</p>

<details class="code-fold">
<summary>Code</summary>

``` python
pt=PowerTransformer(method='yeo-johnson', standardize=False)
skl_yeojohnson=pt.fit(pd.DataFrame(df.invTemp))
skl_yeojohnson=pt.transform(pd.DataFrame(df.invTemp))
df['invTemp_transformed'] = pt.transform(pd.DataFrame(df.invTemp))

fig, axes = plt.subplots(1,2, sharey=True)
sns.histplot(data=df, x="invTemp", ax=axes[0]) 
sns.histplot(data=df['invTemp_transformed'], ax=axes[1])
print('invTemp skew: '+str(skew(df.invTemp.dropna())))
```

</details>

    invTemp skew: 0.7038266269559886

![](inverseTemp_files/figure-commonmark/Skewness%20sensitivity-1.jpeg)

<br>

<b>Mixed effects model assumptions violated</b>
<p>

In this case, a model with a random by-participant slope with and no
covariate produced the best fit (as indexed by BIC scores). But the
model assumptions were violated:

<p>

Select the winning mixed effects model:
</p>

<details class="code-fold">
<summary>Code</summary>

``` python
data=df.reset_index()
formula = 'invTemp_transformed ~ block_type'
basic_model=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
#basic_model.summary()

#test which random effects to include
#feedback_randint=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', vc_formula={'feedback_details': '0+feedback_details'}).fit(reml=False) 
#fractals_randint=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', vc_formula={'fractals': '0+fractals'}).fit(reml=False) 
feedback_fractals_randint=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', vc_formula={"feedback_details": "0 + feedback_details", "fractals": "0 + fractals"}).fit(reml=False)

randslope=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', re_formula='~block_type').fit(reml=False)
feedback_randint_randslope=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', vc_formula={'feedback_details': '0+feedback_details'}, re_formula='~block_type').fit(reml=False)
#feedback_fractals_randint_randslope=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop', vc_formula={'feedback_details': '0+feedback_details', "fractals": "0 + fractals"}, re_formula='~block_type').fit(reml=False)


bic=pd.DataFrame({'basic_model': [basic_model.bic], 
                    #'feedback_randint': [feedback_randint.bic], 
                    #'fractals_randint': [fractals_randint.bic],
                    'feedback_fractals_randint': [feedback_fractals_randint.bic], ##added manually
                    'randslope': [randslope.bic],
                    'feedback_randint_randslope':[feedback_randint_randslope.bic],
                    #'feedback_fractals_randint_randslope': [feedback_fractals_randint_randslope.bic]
                    })
win1=bic.sort_values(by=0, axis=1).columns[0]

#test which covariates to add
##test which covariates to add -- Using the random effects which were best above (basic model in this case)
no_covariate=smf.mixedlm(formula, data, groups=data['participant_no'], missing='drop').fit(reml=False)
sex_covariate=smf.mixedlm(formula+str('+prolific_sex'), data, groups=data['participant_no'], missing='drop').fit(reml=False)
age_covariate=smf.mixedlm(formula+str('+prolific_age'), data, groups=data['participant_no'], missing='drop').fit(reml=False)
digit_span_covariate=smf.mixedlm(formula+str('+digit_span'), data, groups=data['participant_no'], missing='drop').fit(reml=False)
sex_age_covariate=smf.mixedlm(formula+str('+prolific_sex+prolific_age'), data, groups=data['participant_no'], missing='drop').fit(reml=False)
sex_digit_span_covariate=smf.mixedlm(formula+str('+prolific_sex+digit_span'), data, groups=data['participant_no'], missing='drop').fit(reml=False)
digit_span_age_covariate=smf.mixedlm(formula+str('+digit_span+prolific_age'), data, groups=data['participant_no'], missing='drop').fit(reml=False)
sex_age_digit_span_covariate=smf.mixedlm(formula+str('+prolific_sex+prolific_age+digit_span'), data, groups=data['participant_no'], missing='drop').fit(reml=False)

bic=pd.DataFrame({'no_covariate': [no_covariate.bic], 
                    'sex_covariate': [sex_covariate.bic], 
                    'age_covariate': [age_covariate.bic],
                    'digit_span_covariate': [digit_span_covariate.bic],
                    'sex_age_covariate': [sex_age_covariate.bic],
                    'sex_digit_span_covariate': [sex_digit_span_covariate.bic],
                    'digit_span_age_covariate': [digit_span_age_covariate.bic],
                    'sex_age_digit_span_covariate': [sex_age_digit_span_covariate.bic]})
win2=bic.sort_values(by=0, axis=1).columns[0]
print("Winning models: "+ win1 +" "+ win2)
```

</details>

    Winning models: basic_model no_covariate

<p>

Shapiro-Wilk test of normality of residuals
</p>

<details class="code-fold">
<summary>Code</summary>

``` python
#chosen model
results=no_covariate

#shapiro-Wilk test of normality of residuals
labels = ["Statistic", "p-value"]
norm_res = stats.shapiro(results.resid)

for key, val in dict(zip(labels, norm_res)).items():
    print(key, val)
```

</details>

    Statistic 0.9903084045016042
    p-value 3.636461997114105e-06

<p>

White Lagrange multiplier Test for Heteroscedasticity
</p>

<details class="code-fold">
<summary>Code</summary>

``` python
#chosen model
##homoskedasticity of variance 
#White Lagrange Multiplier Test for Heteroscedasticity
het_white_res = het_white(results.resid, results.model.exog)

labels = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]

for key, val in dict(zip(labels, het_white_res)).items():
    print(key, val)
```

</details>

    LM Statistic 0.18343225563776766
    LM-Test p-value 0.9123641075770508
    F-Statistic 0.09145801389911891
    F-Test p-value 0.9126072797449607

<h4>

Run a generalized mixed effects model (done in R)
</h4>

Model details:
<p>

- Gamma probability distribution and identity link function
- no additional random effects
- no additional covariate

</p>

<p>

This is the specification that produced the best fit (according to BIC)
</p>

<details class="code-fold">
<summary>Code</summary>

``` r
df <- read.csv("sensitivity_winningModelOutput.csv")

#inv Temp model
gamma_log <- glmer(invTemp~ block_type + (1|participant_no), data=df, family=Gamma(link="log"))
gamma_inverse <- glmer(invTemp~ block_type + (1|participant_no), data=df, family=Gamma(link="inverse"))
gamma_identity <- glmer(invTemp~ block_type + (1|participant_no), data=df, family=Gamma(link="identity"))

invgaus_log <- glmer(invTemp~ block_type + (1|participant_no), data=df, family=inverse.gaussian(link="log"))
invgaus_inverse <- glmer(invTemp~ block_type + (1|participant_no), data=df, family=inverse.gaussian(link="inverse"))
invgaus_identity <- glmer(invTemp~ block_type + (1|participant_no), data=df, family=inverse.gaussian(link="identity"))

bic_values <- c(
  BIC(gamma_log),
  BIC(gamma_inverse),
  BIC(gamma_identity),
  BIC(invgaus_log),
  BIC(invgaus_inverse),
  BIC(invgaus_identity)
)
model_names <- c("Gamma (log)", "Gamma (inverse)", "Gamma (identity)", "Inverse gaussian (log)", "Inverse gaussian (inverse)", "Inverse gaussian (identity)")

bic_df <- data.frame(Model = model_names, BIC = bic_values)
win1 <- bic_df[which.min(bic_df$BIC), ]$Model

basic_model <- glmer(invTemp~ block_type + (1|participant_no), data=df, family=Gamma(link="identity"))

feedback_randint <- glmer(invTemp~ block_type + (1|participant_no) + (1|feedback_details), data=df, family=Gamma(link="identity"))
#fractals_randint <- glmer(invTemp~ block_type + (1|participant_no) + (1|fractals), data=df, family=Gamma(link="identity"))
#feedback_fractals_randint <- glmer(invTemp~ block_type + (1|participant_no) + (1|fractals) + (1|feedback_details), data=df, family=Gamma(link="identity"))

#randslope <- glmer(invTemp~ block_type + (block_type|participant_no), data=df, family=Gamma(link="identity"))
#feedback_randint_randslope <- glmer(invTemp~ block_type + (block_type|participant_no) + (1|feedback_details), data=df, family=Gamma(link="identity"))
#feedback_fractals_randint_randslope <- glmer(invTemp~ block_type + (block_type|participant_no) + (1|feedback_details) + (1|fractals), data=df, family=Gamma(link="identity"))

bic_values <- c(
  BIC(basic_model),
  BIC(feedback_randint)
  #BIC(fractals_randint),
  #BIC(feedback_fractals_randint),
  #BIC(randslope),
  #BIC(feedback_randint_randslope)
  #BIC(feedback_fractals_randint_randslope)
)
model_names <- c("basic model", "feedback_randint")

bic_df <- data.frame(Model = model_names, BIC = bic_values)
win2 <- bic_df[which.min(bic_df$BIC), ]$Model

#choose covariate structure for basic model
no_covariate <- basic_model
sex_covariate <- glmer(invTemp ~ block_type + (1|participant_no) + prolific_sex, data=df, family=Gamma(link="identity"))
#age_covariate <- glmer(invTemp ~ block_type + (1|participant_no) + prolific_age, data=df, family=Gamma(link="identity"))
#digit_span_covariate <- glmer(invTemp ~ block_type + (1|participant_no) + digit_span, data=df, family=Gamma(link="identity"))
#sex_age_covariate <- glmer(invTemp ~ block_type + (1|participant_no) + prolific_sex + prolific_age, data=df, family=Gamma(link="identity"))
sex_digit_span_covariate <- glmer(invTemp ~ block_type + (1|participant_no) + prolific_sex + digit_span, data=df, family=Gamma(link="identity"))
#digit_span_age_covariate <- glmer(invTemp ~ block_type + (1|participant_no) + prolific_age + digit_span, data=df, family=Gamma(link="identity"))
#sex_digit_span_age_covariate <- glmer(invTemp ~ block_type + (1|participant_no) + prolific_age + prolific_sex + digit_span, data=df, family=Gamma(link="identity"))

bic_values <- c(
  BIC(no_covariate),
  BIC(sex_covariate),
  BIC(sex_digit_span_covariate)
)
model_names <- c("no_covariate", "sex_covariate", "sex_digit_span_covariate")

bic_df <- data.frame(Model = model_names, BIC = bic_values)
win3 <- bic_df[which.min(bic_df$BIC), ]$Model

print(paste0("Winning models: ", win1, " ", win2," ",win3))
```

</details>

    [1] "Winning models: Gamma (identity) basic model no_covariate"

<p>

Results show no effect of feedback-type
</p>

``` r
summary(no_covariate)
```

    Generalized linear mixed model fit by maximum likelihood (Laplace
      Approximation) [glmerMod]
     Family: Gamma  ( identity )
    Formula: invTemp ~ block_type + (1 | participant_no)
       Data: df

         AIC      BIC   logLik deviance df.resid 
      2027.5   2052.0  -1008.7   2017.5      996 

    Scaled residuals: 
         Min       1Q   Median       3Q      Max 
    -1.72184 -0.53135 -0.04576  0.54593  2.69121 

    Random effects:
     Groups         Name        Variance Std.Dev.
     participant_no (Intercept) 0.3373   0.5807  
     Residual                   0.2764   0.5257  
    Number of obs: 1001, groups:  participant_no, 340

    Fixed effects:
                      Estimate Std. Error t value Pr(>|z|)    
    (Intercept)       1.601342   0.054210  29.540   <2e-16 ***
    block_typeFear   -0.020780   0.020880  -0.995    0.320    
    block_typePoints  0.005959   0.024176   0.247    0.805    
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    Correlation of Fixed Effects:
                (Intr) blck_F
    block_typFr -0.208       
    blck_typPnt -0.221  0.505

<p>

Extract confidence intervals
</p>

``` r
print(confint.merMod(no_covariate, method='Wald'))
```

                           2.5 %     97.5 %
    .sig01                    NA         NA
    .sigma                    NA         NA
    (Intercept)       1.49509167 1.70759158
    block_typeFear   -0.06170376 0.02014320
    block_typePoints -0.04142477 0.05334365

<p>

Compute a Bayesian t-test to compare Disgust and Points conditions
</p>

<details class="code-fold">
<summary>Code</summary>

``` python
ttest, bf_null = bayes_factor(df, 'invTemp', 'Disgust', 'Points')

print(f"Disgust vs Points: BF01 = {bf_null}")
```

</details>

    Disgust vs Points: BF01 = 4.25531914893617

<details class="code-fold">
<summary>Code</summary>

``` python
ttest, bf_null = bayes_factor(df, 'invTemp', 'Disgust', 'Fear')

print(f"Disgust vs Fear: BF01 = {bf_null}")
```

</details>

    Disgust vs Fear: BF01 = 16.129032258064516

<p>

We also look at fear vs points (which is not directly assessed by the
model)
</p>

<details class="code-fold">
<summary>Code</summary>

``` python
ttest, bf_null = bayes_factor(df, 'invTemp', 'Points', 'Fear')

print(f"Points vs Fear: T = {ttest['T'][0]}, CI95% = {ttest['CI95%'][0]}, p = {ttest['p-val'][0]}")
```

</details>

    Points vs Fear: T = 1.539624312216987, CI95% = [-0.02  0.18], p = 0.12462498805443724

<p>

And because the result is null, also get a Bayes factor:
</p>

<details class="code-fold">
<summary>Code</summary>

``` python
print(f"Points vs Fear: BF01 = {bf_null}")
```

</details>

    Points vs Fear: BF01 = 5.0

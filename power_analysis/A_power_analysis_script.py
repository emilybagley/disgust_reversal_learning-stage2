# %%
import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn
import pingouin as pg
from scipy.stats import multivariate_normal as mvn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %%
def mixed_effects_power_analysis(num_subjects, num_sims):
    sim_table=pd.DataFrame()
    for n in range(num_sims):
        #simulate data
        std_dev=1
        conditions=['Disgust', 'Fear', 'Points']
        mean_diff=effect_size*std_dev
        within_subject_corr=0.5
        means=[1+mean_diff, 1, 1]
        cov_matrix=np.eye(len(conditions))*(1-within_subject_corr)+within_subject_corr
        scores=mvn.rvs(mean=means, cov=cov_matrix, size=num_subjects)
        df=pd.DataFrame(data=scores, columns=['Disgust', 'Fear', 'Points'])
        df['participant_no']=df.index+1
        df=pd.melt(df, id_vars=['participant_no'], var_name='Condition', value_name='Perseveration') ##convert to long form

        #run mixed effects model on simulated data
        formula = 'Perseveration ~ Condition'
        md=smf.mixedlm(formula, df, groups=df['participant_no'], missing='drop')
        results=md.fit()
        
        #is it significant
        if results.pvalues['Condition[T.Fear]'] < alpha:
            result="Sig"
        else:
            result="Non Sig"
        
        #save into a table
        sim_row=pd.DataFrame({'simulation': [n], 'result': [result]})
        sim_table=pd.concat([sim_table, sim_row])
    return sim_table

# %% [markdown]
# The more liberal mixed effects analysis

# %%
##define changeable parameters
alpha=0.05
effect_size=0.5
power=0.95
min_sample=5
max_sample=100
num_sims=10000

# %%
mixed_effects_power_table=pd.DataFrame()
step=1
for i in range(min_sample, max_sample, step):
    num_subjects=i
    sim_table=mixed_effects_power_analysis(num_subjects, num_sims)
    power=sim_table.result.value_counts(normalize=True).Sig
    power_row=pd.DataFrame({'num_subjects': [num_subjects], 'power': [power]})
    mixed_effects_power_table=pd.concat([mixed_effects_power_table, power_row])

mixed_effects_power_table.to_csv('csvs/liberal_power_table.csv', index=False)


# Now do more conservative power analysis

# %%
alpha=0.05
effect_size=0.2
power=0.95
min_sample=100
max_sample=500
num_sims=10000

# %%
conservative_power_table=pd.DataFrame()
step=1
for i in range(min_sample, max_sample, step):
    num_subjects=i
    sim_table=mixed_effects_power_analysis(num_subjects, num_sims)
    power=sim_table.result.value_counts(normalize=True).Sig
    power_row=pd.DataFrame({'num_subjects': [num_subjects], 'power': [power]})
    conservative_power_table=pd.concat([conservative_power_table, power_row])

conservative_power_table.to_csv('csvs/conservative_power_table_10000.csv', index=False)


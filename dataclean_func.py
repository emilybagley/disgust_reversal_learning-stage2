#CONTAINS FUNCTIONS NECESSARY FOR EXTRACTING DATA FROM TASKS
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


#Extract diagnostic information
def make_diagnosis(df):
    diagnosis=pd.DataFrame()
    for i in set(df.participant_no): 
        sub_df=df[df.participant_no==float(i)]
        temp_diagnosis=sub_df[sub_df.trial_var=="psychiatric diagnosis"].reset_index()
        
        if len(temp_diagnosis.index)==0:
            temp_diagnosis=pd.DataFrame({'diagnosis': ['No data']})
        else:
            temp_diagnosis=pd.DataFrame({'diagnosis': [ast.literal_eval(temp_diagnosis.response[0])['Q0']]})
        temp_diagnosis['participant_no']=sub_df[0:1].participant_no.iloc[0]
        
        check_diagnosis=temp_diagnosis.diagnosis.iloc[0].lower()
        cleaned_diagnosis='MANUAL CHECK'
        depression=0
        anxiety=0
        adhd=0
        autism=0
        ptsd=0
        eating_disorder=0
        ocd=0
        personality_disorder=0
        fnd=0
        bipolar=0
        historical=0
        unspecified=0
        
        if 'depression' in check_diagnosis or 'dqepression' in check_diagnosis or 'depressive' in check_diagnosis:
            depression=1
            cleaned_diagnosis='Yes'
        if 'anxiety' in check_diagnosis or 'panic disorder' in check_diagnosis or 'gad' in check_diagnosis or 'general anxiety disorder' in check_diagnosis or 'anxious' in check_diagnosis:
            anxiety=1
            cleaned_diagnosis='Yes'
        if 'adhd' in check_diagnosis:
            adhd=1
            cleaned_diagnosis='Yes'
        if 'autism' in check_diagnosis or 'asc' in check_diagnosis or 'asd' in check_diagnosis:
            autism=1
            cleaned_diagnosis='Yes'
        if 'ptsd' in check_diagnosis or 'post traumatic stress disorder' in check_diagnosis or 'post-traumatic stress disorder' in check_diagnosis:
            ptsd=1
            cleaned_diagnosis='Yes'
        if 'eating disorder' in check_diagnosis or 'anorexia' in check_diagnosis or 'body dismorphia' in check_diagnosis or 'body dysmorphia' in check_diagnosis or 'body dysmorphic' in check_diagnosis:
            eating_disorder=1
            cleaned_diagnosis='Yes'
        if 'ocd' in check_diagnosis:
            ocd=1
            cleaned_diagnosis='Yes'
        if 'bpd' in check_diagnosis or 'borderline personality disorder' in check_diagnosis or 'border personality disorder' in check_diagnosis or 'anti social personality disorder' in check_diagnosis or 'eupd' in check_diagnosis or 'emotionally unstable personality disorder' in check_diagnosis:
            personality_disorder=1
            cleaned_diagnosis='Yes'
        if 'bipolar' in check_diagnosis:
            bipolar=1
            cleaned_diagnosis='Yes'
        if 'fnd' in check_diagnosis or 'functional neurological disorder' in check_diagnosis or 'non-epileptic attack disorder' in check_diagnosis or 'nead' in check_diagnosis:
            fnd=1
            cleaned_diagnosis='Yes'
        if 'no ' in check_diagnosis or 'none' in check_diagnosis or check_diagnosis == ' no' or 'no, ' in check_diagnosis or "no\n" in check_diagnosis or check_diagnosis == 'mo' or 'na ' in check_diagnosis or ' na' in check_diagnosis or check_diagnosis == 'na' or 'n/a' in check_diagnosis or check_diagnosis == '' or check_diagnosis == 'no':
            cleaned_diagnosis='No'
        if 'a few years ago' in check_diagnosis or 'past' in check_diagnosis or 'years ago' in check_diagnosis or 'as a child' in check_diagnosis or 'nothing since' in check_diagnosis:
            cleaned_diagnosis='No'
            historical='Yes'
        if check_diagnosis =='yes' or check_diagnosis == 'yess' or check_diagnosis == 'yes, am currently having councelling. cbt':
            cleaned_diagnosis='Yes'
            unspecified=1
        ##catch exceptions
        if 'by a gp' in check_diagnosis or 'currently medicated with' in check_diagnosis or 'from a dr' in check_diagnosis: 
            ##to catch someone who says they were diagnosed, not by a psychiatrist but a gp
            ##to catch someone who says they were diagnosed years ago but still take medication
            ##someone says diagnosed from a dr not psychiatrist
            cleaned_diagnosis='Yes'

        temp_diagnosis['cleaned_diagnosis']=cleaned_diagnosis
        temp_diagnosis['depression']=depression
        temp_diagnosis['anxiety']=anxiety
        temp_diagnosis['adhd']=adhd
        temp_diagnosis['autism']=autism
        temp_diagnosis['ptsd']=ptsd
        temp_diagnosis['eating_disorder']=eating_disorder
        temp_diagnosis['ocd']=ocd
        temp_diagnosis['bpd']=personality_disorder
        temp_diagnosis['bipolar']=bipolar
        temp_diagnosis['fnd']=fnd
        temp_diagnosis['historical']=historical
        temp_diagnosis['unspecified']=unspecified

        diagnosis=pd.concat([diagnosis, temp_diagnosis])
    return diagnosis


##extract digit_span task performance
def make_digit_span(df):
    digit_span=pd.DataFrame()
    for i in set(df.participant_no): 
        sub_df=df[df.participant_no==float(i)]
        digit_span_df=sub_df.dropna(subset=['digit_span']).reset_index()
        if len(digit_span_df.index)==0:
            temp_digit_span=pd.DataFrame({'digit_span': "task failed", 'participant_no': [sub_df.participant_no.iloc[0]] })
        else:
            temp_digit_span=pd.DataFrame({'digit_span': [digit_span_df.digit_span[0]], 'participant_no': [digit_span_df.participant_no[0]]})
        digit_span=pd.concat([digit_span, temp_digit_span])
    return digit_span

#extract video ratings
def vid_ratings(df, participant_no, to_do):  
    sub_df=df[df.participant_no==float(participant_no)]
    rating_vids_df=sub_df[sub_df.trial_var=="rate_stim"]
    rating_vids_df.replace({'response': '  '}, np.nan, inplace=True)
    rating_vids_df= rating_vids_df.dropna(subset=['response'])
    rating_vids_df.sort_values(by=["stimulus", "trial_index"], inplace=True) ##groups dataframe by video type - allows you to extract 1st, 2nd and 3rd presentation of each video
    
    #create a dataframe with all ratings (for one participant)
    rating_vids_a=[]
    rating_vids_b=[]

    vals = range(len(rating_vids_df.index))

    for val in vals: #loop through the dataframe
        stim = str(rating_vids_df.iloc[val].stimulus)
        trial_type=rating_vids_df.iloc[val].type
        response = ast.literal_eval(rating_vids_df.iloc[val].response) #makes it a dictionary
        unpleasant=response['Q0']
        arousing=response['Q1']
        disgusting=response['Q2']
        frightening=response['Q3']

        #extract which video it is
        if "0888" in stim:
            vid="0888"
        elif "1414" in stim:
            vid="1414"
        elif "1765" in stim:
            vid="1765"
        elif "1987" in stim:
            vid="1987"
        elif "2106" in stim:
            vid="2106"
        elif "0046" in stim:
            vid = "0046"
        elif "0374" in stim:
            vid = "0374"
        elif "0548" in stim:
            vid = "0548"
        elif "0877" in stim:
            vid = "0877"
        elif "1202" in stim:
            vid = "1202"
        else:
            vid = "ERROR"

        if val in range(0, 30, 2):
            rating_vids_a.append({
                'Vid' : vid,
                'trial_type': trial_type,
                'unpleasant_1': unpleasant,
                'arousing_1': arousing,
                'disgusting_1': disgusting,
                'frightening_1': frightening,
                'disgust_stim': 0,
                'fear_stim': 0,
            })
        elif val in range(1,30,2):
            rating_vids_b.append({
                'Vid' : vid,
                'trial_type': trial_type,
                'unpleasant_2': unpleasant,
                'arousing_2': arousing,
                'disgusting_2': disgusting,
                'frightening_2': frightening,
                'disgust_stim': 0,
                'fear_stim': 0,
            })

    rating_vids_a=pd.DataFrame(rating_vids_a)
    rating_vids_b=pd.DataFrame(rating_vids_b)
    rating_vids=rating_vids_a.merge(rating_vids_b, on=['Vid', 'trial_type', 'disgust_stim', 'fear_stim'])
    rating_vids=rating_vids[['Vid', 'trial_type','unpleasant_1', 'unpleasant_2','arousing_1', 'arousing_2','disgusting_1', 'disgusting_2', 'frightening_1', 'frightening_2', 'disgust_stim', 'fear_stim']]

    #add which video was chosen for disgust and fear stim
    fear_stim=str(sub_df.fear_stimulus.dropna())
    disgust_stim=str(sub_df.disgust_stimulus.dropna())

    if "0888" in disgust_stim:
        rating_vids.loc[rating_vids['Vid']=="0888", ['disgust_stim']]=1
    elif "1414" in disgust_stim:
        rating_vids.loc[rating_vids['Vid']=="1414", ['disgust_stim']]=1
    elif "1765" in disgust_stim:
        rating_vids.loc[rating_vids['Vid']=="1765", ['disgust_stim']]=1
    elif "1987" in disgust_stim:
        rating_vids.loc[rating_vids['Vid']=="1987", ['disgust_stim']]=1
    elif "2106" in disgust_stim:
        rating_vids.loc[rating_vids['Vid']=="2106", ['disgust_stim']]=1
    else:
        print("error")

    if "0046" in fear_stim:
        rating_vids.loc[rating_vids['Vid']=="0046", ['fear_stim']]=1
    elif "0374" in fear_stim:
        rating_vids.loc[rating_vids['Vid']=="0374", ['fear_stim']]=1
    elif "0548" in fear_stim:
        rating_vids.loc[rating_vids['Vid']=="0548", ['fear_stim']]=1
    elif "0877" in fear_stim:
        rating_vids.loc[rating_vids['Vid']=="0877", ['fear_stim']]=1
    elif "1202" in fear_stim:
        rating_vids.loc[rating_vids['Vid']=="1202", ['fear_stim']]=1
    else:
        print("error")

    #add participant number
    rating_vids['participant_no']=sub_df.reset_index().participant_no[0]
    #create dataframe with just the chosen stimuli for each subject
    chosen_stim=pd.concat([rating_vids[rating_vids.disgust_stim==1], rating_vids[rating_vids.fear_stim==1]])
    chosen_stim['participant_no']=sub_df.reset_index().participant_no[0]
    
    #extract ratings of points loss (after points block)
    points_rating=sub_df[sub_df.trial_var=="points_rate_stim"]
    #extract ratings of points loss (after points block)
    ##due to glitch - 1/3rd of participants don't have points ratings
        #happened when the points block was last
    if len(sub_df[sub_df.trial_var=="points_rate_stim"])==0:
        response = {'Q0': np.nan, 'Q1': np.nan, 'Q2': np.nan, 'Q3': np.nan}
    else:
        points_rating=sub_df[sub_df.trial_var=="points_rate_stim"]
        response = ast.literal_eval(points_rating.iloc[0].response) #makes it a dictionary
    #response = {'Q0': 4, 'Q1': 3, 'Q2': 2, 'Q3': 1} ###REMOVE ONCE HAVE ACTUAL DATA
    points_rating=pd.DataFrame({
        'participant_no': [participant_no],
        'Vid' : ["points"],
        'trial_type': ["points"],
        'unpleasant_1': [response['Q0']],
        'arousing_1': [response['Q1']],
        'disgusting_1': [response['Q2']],
        'frightening_1': [response['Q3']],
        'disgust_stim': [0],
        'fear_stim': [0],
        'points_stim': [1],
    }) 
    
    if to_do == "plot":
        #Checking chosen stim
        fig, ax = plt.subplots(nrows=2,ncols=2, sharey=False)
        fig.tight_layout(pad=4)

        ax[0,0].bar(['Disgust', 'Fear'], [np.mean(chosen_stim[chosen_stim.trial_type=="disgust"].unpleasant_2), np.mean(chosen_stim[chosen_stim.trial_type=="fear"].unpleasant_2)])
        ax[0,0].set_title("Valence ratings")

        ax[0,1].bar(['Disgust', 'Fear'], [np.mean(chosen_stim[chosen_stim.trial_type=="disgust"].arousing_2), np.mean(chosen_stim[chosen_stim.trial_type=="fear"].arousing_2)])
        ax[0,1].set_title("Arousal ratings")

        ax[1,0].bar(['Disgust', 'Fear'], [np.mean(chosen_stim[chosen_stim.trial_type=="disgust"].disgusting_2), np.mean(chosen_stim[chosen_stim.trial_type=="fear"].disgusting_2)])
        ax[1,0].set_title("Disgust ratings")

        ax[1,1].bar(['Disgust', 'Fear'], [np.mean(chosen_stim[chosen_stim.trial_type=="disgust"].frightening_2), np.mean(chosen_stim[chosen_stim.trial_type=="fear"].frightening_2)])
        ax[1,1].set_title("Fear ratings")
        fig.suptitle("Subject number "+str(int(sub_df.reset_index().participant_no[0])), size=16)
    elif to_do == "rating_vids":
        return rating_vids
    elif to_do == "chosen_stim":
        return chosen_stim
    elif to_do == "points_rating":
        return points_rating
    else:
        return "ERROR"


##REVERSAL LEARNING TASK
#create_block_df - creates a dataframe with all needed data from one experimental block
def create_block_df(df, block_name, participant_no):
    sub_df=df[df.participant_no==participant_no].reset_index()
    task_df=sub_df[sub_df.task=="main_task"]
    block_df=pd.DataFrame(columns=['n_trial', 'rt', 'stim_selected', 'correct_stim', 'correct', 'feedback', 'feedback_congruent', 'correct_count', 'trial_till_correct', 'reversal', 'block_no', 'participant_no', 'timed_out'])
    block=task_df[task_df.block_type==block_name]
    block.reset_index(inplace=True)
    block.drop(['level_0', 'index'], axis=1, inplace=True)

    #extract fractal pair used:
    fractal_pair=pd.DataFrame(set(block.correct_stim)).dropna()
    fractal_pair.sort_values(by=[0], inplace=True)
    fractals=list(fractal_pair[0])
    fractal_val={'F000', 'F009', 'F010', 'F012', 'F014', 'F015', 'F018', 'F020'}
    fractal_vals = [next((val for val in fractal_val if val in item), item) for item in fractals]

    #extract punishment colour
    first_punishment = task_df.loc[task_df['stimulus'].str.contains(block_name, case=False, na=False), 'stimulus'].iat[0]
    if first_punishment.find('YELLOW') != -1:
        punish_colour='YELLOW'
    elif first_punishment.find('BLUE') != -1:
        punish_colour='BLUE'
    else:
        punish_colour='ERROR'

    for i in set(block.n_trial):
        trial=block[block.n_trial==i]
        trial.reset_index(inplace=True)

        row = []
        if punish_colour in trial.feedback[2]:
            feedback='incorrect'
        else:
            feedback='correct'

        row.append({
            'n_trial': trial.n_trial[0], #
            'stim_selected': trial.stim_selected[0],#
            'correct_stim': trial.correct_stim[0],#
            'correct': trial.correct[0],#
            'feedback': feedback,
            'feedback_congruent': trial.feedback_congruent[2],
            'correct_count': trial.correct_count[0],#
            'trial_till_correct': trial.trial_till_correct[0],#
            'rt': trial.rt[0],
            'reversal': trial.reversal[0],#
            'block_no': trial.block_no[0],#
            'block_type': trial.block_type[0],
            'participant_no': trial.participant_no[0],#
            'timed_out': 0,
            'time_taken': (block.time_elapsed.iloc[-1]-block.time_elapsed[0])/60000, ##in minutes
            'fractals': fractal_vals,
            'punish_colour': punish_colour
        })
        block_df=pd.concat([block_df, pd.DataFrame(row)])
    block_df.reset_index(inplace=True)


    #replace stimuli with 0 and 1 (for plotting)
    stim=list(set(block_df.correct_stim.to_list()))
    stim0="<img src='"+str(stim[0])+"'</img>"
    stim0b="  <img src='"+str(stim[0])+"'</img>"
    stim1="<img src='"+str(stim[1])+"'</img>"
    stim1b="  <img src='"+str(stim[1])+"'</img>"

    block_df.replace([stim[0], stim[1]], [0, 1], inplace=True)
    block_df.replace([stim0, stim1], [0,1], inplace=True)
    block_df.replace([stim0b, stim1b], [0,1], inplace=True)

    #did they time out before reaching 7 reversals
    short_block=block[block.trial_till_correct.notna()] ##removes trials after they timed out (if they did)
    if short_block.iloc[-1].reversal==7.0 and short_block.iloc[-1].correct_count>=5:
        block_df.timed_out=0
    else:
        block_df.timed_out=1
    
    ##did reach reversal criteria for inclusion
    criteria=5
    if short_block.iloc[-1].reversal>=criteria:
        block_df['criteria']=0
    else:
        block_df['criteria']=1

    return block_df

#uses create_block_df to create a 'task' dataframe with all data across all three blocks
def create_task_df(df, to_do):
    task_df=pd.DataFrame()
    for participant_no in set(df.participant_no):
        if to_do == "plot":
            fig, ax = plt.subplots(3,1, sharey=True)
            fig.tight_layout(pad=4)

            disgust_df=create_block_df(df, "Disgust", participant_no)
            ax[0].plot(disgust_df.stim_selected, 'o')
            ax[0].plot(disgust_df.correct_stim)
            if disgust_df.timed_out[0] == 0:
                timed_out="false"
            else:
                timed_out="true"

            if disgust_df.criteria[0] == 0:
                criteria="false"
            else:
                criteria="true"           

            ax[0].set_title("DISGUST block number: "+str(int(disgust_df.block_no[0]+1))+", timed out: "+timed_out+", criteria:"+criteria)

            fear_df=create_block_df(df, "Fear", participant_no)
            ax[1].plot(fear_df.stim_selected, 'o')
            ax[1].plot(fear_df.correct_stim)
            if fear_df.timed_out[0] == 0:
                timed_out="false"
            else:
                timed_out="true"

            if fear_df.criteria[0] == 0:
                criteria="false"
            else:
                criteria="true"  

            ax[1].set_title("FEAR block number: "+str(int(fear_df.block_no[0]+1))+", timed out: "+timed_out+", criteria:"+criteria )

            points_df=create_block_df(df, "Points", participant_no)
            ax[2].plot(points_df.stim_selected, 'o')
            ax[2].plot(points_df.correct_stim)
            if points_df.timed_out[0] == 0:
                timed_out="false"
            else:
                timed_out="true"

            if points_df.criteria[0] == 0:
                criteria="false"
            else:
                criteria="true"  

            ax[2].set_title("POINTS block number: "+str(int(points_df.block_no[0]+1))+", timed out: "+timed_out+", criteria:"+criteria)

            fig.suptitle("Subject number "+str(participant_no), size=16)
        temp_task=pd.concat([create_block_df(df, "Disgust", participant_no), create_block_df(df, "Fear", participant_no), create_block_df(df, "Points", participant_no)])
        task_df=pd.concat([task_df, temp_task])
    return task_df

#checks if the participant understood the task (using number of reversals achieved and attention checks)
def make_task_understood(df, complete_task_df, to_do):
    task_understood=pd.DataFrame()
    for i in set(df.participant_no):
        task_understood_temp=pd.DataFrame({'participant_no': [i]})
        sub_df=df[df.participant_no==float(i)].reset_index()

         #attention checks 
        attention=sub_df[sub_df.trial_var=="attention_check"].reset_index()
        if ast.literal_eval(attention.loc[0].response) == "{'Q0': ['Apple', 'Banana']}" or attention.loc[0].response == "{'Q0': ['Apple', 'Banana']}":
                    block1=2
        elif 'Apple' in ast.literal_eval(attention.loc[0].response)['Q0'] or 'Banana' in ast.literal_eval(attention.loc[0].response)['Q0']:
                    block1=1
        else:
                    block1=0
                
        if attention.loc[1].response== "{'Q0': ['Bowl', 'Spoon']}":
                    block2=2
        elif 'Bowl' in ast.literal_eval(attention.loc[1].response)['Q0'] or 'Spoon' in ast.literal_eval(attention.loc[1].response)['Q0']:
                    block2=1
        else:
                    block2=0
        if attention.loc[2].response== "{'Q0': ['River', 'Mountain']}":
                    block3=2
        elif 'River' in ast.literal_eval(attention.loc[2].response)['Q0'] or 'Mountain' in ast.literal_eval(attention.loc[2].response)['Q0']:
                    block3=1
        else:
                    block3=0

        attention_checks = pd.DataFrame({
            'block': ['block 1', 'block 2', 'block 3'],
            'correct' :[block1, block2, block3]
        })
        task_understood_temp['attention_checks']=np.sum(attention_checks.correct)

        #timings - breaks and total time elapsed
        task=sub_df[sub_df.task=="main_task"]
        if len(task[task.rt/60000>10].index) ==0:
            task_understood_temp['long_breaks']="No"
        else:
            task_understood_temp['long_breaks']="Yes"
            task_understood_temp['breaks_details']=[task[task.rt/60000>10].reset_index().rt[0]/60000]
        
        task_understood_temp['total_time']=task.time_elapsed.iloc[-1]/60000

        ##did they reach the right number of reversals
        task_df=complete_task_df[complete_task_df.participant_no==i]
        disgust=task_df[task_df.block_type=="Disgust"]
        task_understood_temp['timed_out_d']=disgust.reset_index().timed_out[0]
        task_understood_temp['criteria_d']=disgust.reset_index().criteria[0]

        fear=task_df[task_df.block_type=="Fear"]
        task_understood_temp['timed_out_f']=fear.reset_index().timed_out[0]
        task_understood_temp['criteria_f']=fear.reset_index().criteria[0]

        points=task_df[task_df.block_type=="Points"]
        task_understood_temp['timed_out_p']=points.reset_index().timed_out[0]
        task_understood_temp['criteria_p']=points.reset_index().criteria[0]

        task_understood_temp['timed_out_total']=task_understood_temp[['timed_out_f', 'timed_out_d', 'timed_out_p']].sum(axis=1)
        task_understood_temp['criteria_total']=task_understood_temp[['criteria_f', 'criteria_d', 'criteria_p']].sum(axis=1)

        ##Checking they learnt the task correctly
        if task_understood_temp.attention_checks[0]>=4 and task_understood_temp.criteria_total[0]<3 and task_understood_temp.long_breaks[0]=="No":
            task_understood_temp['task_understood']="Yes"
        else:
            task_understood_temp['task_understood']="No"
        task_understood=pd.concat([task_understood, task_understood_temp])

    if to_do == "plot_exclusions":
        for participant_no in set(task_understood[task_understood.task_understood=="No"].participant_no):
            fig, ax = plt.subplots(3,1, sharey=True)
            fig.tight_layout(pad=4)

            disgust_df=create_block_df(df, "Disgust", participant_no)
            ax[0].plot(disgust_df.stim_selected, 'o')
            ax[0].plot(disgust_df.correct_stim)
            if disgust_df.timed_out[0] == 0:
                timed_out="false"
            else:
                timed_out="true"

            if disgust_df.criteria[0] == 0:
                criteria="false"
            else:
                criteria="true"           

            ax[0].set_title("DISGUST block number: "+str(int(disgust_df.block_no[0]+1))+", timed out: "+timed_out+", criteria:"+criteria)

            fear_df=create_block_df(df, "Fear", participant_no)
            ax[1].plot(fear_df.stim_selected, 'o')
            ax[1].plot(fear_df.correct_stim)
            if fear_df.timed_out[0] == 0:
                timed_out="false"
            else:
                timed_out="true"

            if fear_df.criteria[0] == 0:
                criteria="false"
            else:
                criteria="true"  

            ax[1].set_title("FEAR block number: "+str(int(fear_df.block_no[0]+1))+", timed out: "+timed_out+", criteria:"+criteria )

            points_df=create_block_df(df, "Points", participant_no)
            ax[2].plot(points_df.stim_selected, 'o')
            ax[2].plot(points_df.correct_stim)
            if points_df.timed_out[0] == 0:
                timed_out="false"
            else:
                timed_out="true"

            if points_df.criteria[0] == 0:
                criteria="false"
            else:
                criteria="true"  

            ax[2].set_title("POINTS block number: "+str(int(points_df.block_no[0]+1))+", timed out: "+timed_out+", criteria:"+criteria)

            fig.suptitle("Subject number "+str(participant_no), size=16)
    return task_understood

#create a dataframe with all outcomes from the task
def make_task_outcomes(df):
    blocks=["Disgust", "Fear", "Points"]
    task_summary=pd.DataFrame()

    for participant in list(set(df.participant_no)):
        row=pd.DataFrame(data=[participant], columns=['participant_no'])
        for block_type in blocks:
            sub_df=df[df.participant_no==participant]
            block_df=sub_df[sub_df.block_type==block_type].reset_index()
            block_df=block_df[block_df.trial_till_correct.notna()] ##removes trials after they timed out (if they did)
            percentage_correct=block_df['correct'].value_counts(normalize=True)[True]
            fractals=str(block_df.fractals.iloc[-1])

            #perseverative and regressive errors
            n_reversal=int(block_df.iloc[-1].reversal)+1

            first_correct=0
            perseverative_er=0
            regressive_er=0
            n_till_correct=[]

            block_perseverative_er=0
            for n in range(n_reversal):
                reversal_df=block_df[block_df.reversal==n].reset_index()
                reversal_length=np.shape(reversal_df)[0]+1
                n_till_correct.append(reversal_df.trial_till_correct.iloc[-1]) 

                #find first correct response
                for i in range(reversal_length):
                    if len(reversal_df)<=i: ##if the reversal is only 1 trial long break
                            first_correct=0
                            break
                    elif reversal_df.loc[i].correct == True:
                        first_correct=i
                        break
                perseverative_er+=first_correct ##when they first get the correct response is an index of perseveration
                #count perseverative errors with this (first trial doesn't count as an error)
                if first_correct == 0:
                    reversal_perseverative_er=0
                else:
                    reversal_perseverative_er=first_correct-1 #as first trial doesn't count
                block_perseverative_er+=reversal_perseverative_er
                mean_perseverative_er=block_perseverative_er/(n_reversal-1) ##mean perseverative errors per reversal

                #count number of errors past this point --> gives you regressive errors
                if np.shape(reversal_df.iloc[first_correct: reversal_length].correct.value_counts()) ==(1,):
                    regressive_er=regressive_er
                else:
                    regressive_er+=reversal_df.iloc[first_correct: reversal_length].correct.value_counts()[False]
                mean_regressive_er=regressive_er/(n_reversal-1) ##mean regressive errors per reversal

            total_error=block_df['correct'].value_counts()[False]
            if regressive_er+perseverative_er != total_error: ##catches error in the code
                prop_regressive_er="ERROR"
                prop_perseverative_er="ERROR"

            median_till_correct = statistics.median(n_till_correct)
            mean_till_correct=np.mean(n_till_correct)

            #win-stay-lose-shift - NB based on actual feedback not on actually whether they were correct
            #win-stay
            win_stay_count=0
            win_shift_count=0
            win_index=block_df[block_df.feedback=="correct"].index
            if win_index[-1]+1 > block_df.index[-1]: ## to prevent an error in for loop by indexing more than is in the block
                win_index=win_index.drop(win_index[-1])#if last trial is a win, drop from win index because can't test what happened on next trial
            for i in win_index:
                if block_df.loc[i].stim_selected==block_df.loc[i+1].stim_selected: ##did you choose the same when you won and the trial after
                    win_stay_count+=1
                else:
                    win_shift_count+=1
            win_stay_ratio=win_stay_count/(win_stay_count+win_shift_count)

            #lose-shift
            lose_stay_count=0
            lose_shift_count=0
            lose_index=block_df[block_df.feedback=="incorrect"].index
            if lose_index[-1]+1 > block_df.index[-1]: ## to prevent an error in for loop by indexing more than is in the block
                lose_index=lose_index.drop(lose_index[-1]) #if last trial is a loss, drop from lose index because can't test what happened on next trial
            for i in lose_index:
                if block_df.loc[i].stim_selected==block_df.loc[i+1].stim_selected: ##did you choose the same when you lost and the trial after
                    lose_stay_count+=1
                else:
                    lose_shift_count+=1
            lose_shift_ratio=lose_shift_count/(lose_stay_count+lose_shift_count)

            if block_type=="Disgust":
                disgust=1
                fear=0
                points=0
            elif block_type=="Fear":
                fear=1
                disgust=0
                points=0
            elif block_type=="Points":
                points=1
                fear=0
                disgust=0
            else:
                print("ERROR")

            #write into a dataframe
            row=pd.DataFrame({
                'percentage_correct': [percentage_correct],
                'mean_perseverative_er': [mean_perseverative_er],
                'mean_regressive_er': [mean_regressive_er],
                'median_till_correct': [median_till_correct],
                'mean_till_correct': [mean_till_correct],
                'win_stay': [win_stay_ratio],
                'lose_shift': [lose_shift_ratio],
                'timed_out': [block_df.timed_out[0]],
                'block_no': [block_df.block_no[0]+1],
                'block_type': block_type,
                'disgust_block': disgust,
                'fear_block': fear,
                'points_block': points,
                'participant_no': participant,
                'fractals': fractals
            })
            task_summary=pd.concat([task_summary, row])
    return task_summary


def replace_outliers_with_nan(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1- 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column]=df[column].apply(lambda x: np.nan if x<lower_bound or x>upper_bound else x)
    return df

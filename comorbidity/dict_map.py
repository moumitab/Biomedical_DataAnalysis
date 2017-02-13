#In this code we input a csv file and make a dict where key is SNOMED code and freq is the value

import pandas as pd
import numpy as np
#fileoutput = '/Users/moumitabhattacharya/Desktop/Research_Phase2/Diagnosis_Analysis/Gender_Comorbidity_study/Male/cond_freq_Male.csv'
fileinput1 = '/Users/moumitabhattacharya/Desktop/Research_Phase2/KidneyData/data/updatedData/Office_20160303_sorted_Run2_Frequency_conditons.csv'
data = pd.read_csv(fileinput1,delimiter=',')


dict_con = dict()
for i, row in data.iterrows():
    cond = row['conditions']
    print cond
    print row['Frequency']
    cond = float(cond)
    if cond not in dict_con:
        dict_con[cond] = row['Frequency']
    else:
        dict_con[cond] = dict_con[cond] + row['Frequency']

w1 = open('/Users/moumitabhattacharya/Desktop/Research_Phase2/KidneyData/data/updatedData/Office_20160303_sorted_Run2_cond.csv','w')
print dict_con
for each in dict_con.items():
    w1.write(str(each))
    w1.write('\n')


import pandas as pd
from pandas import DataFrame
import datetime
import pandas.io.data
import csv
import numpy as np
import matplotlib.pyplot as plt

#We will create a new csv for hospital records with each of the  206 frequent conditions as columns
#we remove the codes not found in the master list of SNOMED codes
# Value in each of these column will be set to 1 if the conditon is present in the condition list
# else set to 0 if the condition is not present in the list

#Read hospital records and 206commonConditions file
fileinput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\Hospital_20151104_sorted_2.csv'
fileinputConditions = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\conditionsList.csv'
fileoutput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\Hospital_ConditonFeatures_2.csv'

#Use Pandas to read CSV
data = pd.read_csv(fileinput,index_col = 'visit_occurrence_id', parse_dates = True)
dataConditions = pd.read_csv(fileinputConditions)
print(data.head())

dict_con = dict()
for i, row in dataConditions.iterrows():
    cond = row['conditions']
    if cond not in dict_con:
        dict_con[cond] = 1
    else:
        dict_con[cond] = dict_con[cond] + 1

#Read eachline of the hospita records and go through the conditions list,
# if the condition is one among the frequency list conditions, fill the corresponding condition column with 1
#with open('C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\Hospital_20151104_sorted.csv','wb') as f:

sum1 = 0
sum2 = 0
sum3 = 0
for i, row in data.iterrows():
    CondRow = row['conditions']
    sum1 = sum1 + 1
    if(str(CondRow).lower() != 'nan'):

        listCondition = CondRow.split(',')
        for eachCond in listCondition:
            sum2 = sum2 +1
            eachCond = int(eachCond)
            if eachCond in dict_con.keys():
                sum3 = sum3 +1
                data.loc[i, eachCond] = 1

print(sum3)
print(sum2)
#Output
data.to_csv(fileoutput)


#Code commented and can be used only to generate the 188 unique condition set for features
# with open('C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\commonConditionListforColumns.csv','wb') as f:
#     w = csv.writer(f)
#     w.writerow(dict_con.keys())
#     w.writerow(dict_con.values())
#     w.writerow('\n')
#






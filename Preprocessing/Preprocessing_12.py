import pandas as pd
from pandas import DataFrame
import datetime
import pandas.io.data
import csv
import numpy as np
import matplotlib.pyplot as plt

#We will create a new csv for office records or hospital records with each of the 40/50 frequent conditions
# from hospital or office records added to the record-set as an indvidual features
#we remove the codes not found in the master list of SNOMED codes
# Value in each of these column will be set to 1 if the conditon is present in the condition list
# else set to 0 if the condition is not present in the list

#Input file: office records, HospitalRecordsTop30conditionsList
#Output file: New office records with all other fields intact along with additional 30 columns populated with values 1 or 0

fileinput1 = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\records_date\\Hospital_20151104_sorted_date.csv'
fileinput2 = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\HospitalRecordsTop105conditionsList.csv'
fileoutput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\records_date\\HospitalRecords_105TopConditionFeatures.csv'

data = pd.read_csv(fileinput1,delimiter=',')
conditionList = pd.read_csv(fileinput2,delimiter=',')

#Create a diction containing all the 30 conditions. This is done in order to access the conditions in O(1) runtime
dict_con = dict()
for i, row in conditionList.iterrows():
    cond = row['conditions']
    if cond not in dict_con:
        dict_con[cond] = 1
    else:
        dict_con[cond] = dict_con[cond] + 1

#Read eachline of the hospita records and go through the conditions list,
# if the condition is one among the frequency list conditions, fill the corresponding condition column with 1
print(len(dict_con))
sum1 = 0
sum2 = 0
sum3 = 0
for i, row in data.iterrows():
    CondRow = row['conditions']
    sum1 = sum1 + 1
    if(str(CondRow).lower() != 'nan' ):
        listCondition = CondRow.split(',')
        for eachCond in listCondition:
            if(eachCond != ' No matching concept'):
                sum2 = sum2 +1
                eachCond = float(eachCond)
                if eachCond in dict_con.keys():
                    sum3 = sum3 +1
                    print("visit id",row['visit_occurrence_id'])
                    print("condition SNOMED code",eachCond)
                    data.loc[i, eachCond] = 1

print(sum3)
print(sum2)
#Output
data.to_csv(fileoutput)

#creating frequency table of conditions in Hospital data
import numpy
import pandas
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import re

#f = open('C:\\Users\\Moumita\\Google Drive\\Research_Phase2\\KidneyData\\Data_iteration3\\Hospital_Cond_feature.csv','r')
#f = open('/Users/moumitabhattacharya/Desktop/Research_Phase2/Diagnosis_Analysis/Gender_Comorbidity_study/Male/Office_Cond_feature_Male.csv','r')
f = open('/Users/moumitabhattacharya/Desktop/Research_Phase2/KidneyData/data/updatedData/Office_20160303_sorted_Run2_conditions.csv','r')
#w1 = open('/Users/moumitabhattacharya/Desktop/Research_Phase2/Diagnosis_Analysis/Gender_Comorbidity_study/Male/Office_Frequency_conditons_Male.csv','w')
w1 = open('/Users/moumitabhattacharya/Desktop/Research_Phase2/KidneyData/data/updatedData/Office_20160303_sorted_Run2_Frequency_conditons.csv','w')
#w1 = open('C:\\Users\\Moumita\\Google Drive\\Research_Phase2\\Diagnosis_Analysis\\Analysis_hospitalRecordSet\\FrequencyListOfConditionsInHospitalRecordSet\\Hospital_Frequency_conditons_round2.csv','w')
data = f.readlines()
dict_x = dict()
for dataFirstLine in data:
    dataFirstLine1 = dataFirstLine.strip("'").strip('"').strip('\n').strip('"')
    dataFirstLineEach = dataFirstLine1.split(',')
    for eachItem in dataFirstLineEach:
        if eachItem not in dict_x:
            dict_x[eachItem] = 1
        else:
            dict_x[eachItem] = dict_x[eachItem] + 1

for each in dict_x.items():
    w1.write(str(each))
    w1.write('\n')




#lengthOfDict = len(dict_x)
#plt.bar(range(lengthOfDict),dict_x.values(),align='center')
#plt.xticks(range(lengthOfDict), dict_x.keys())
#plt.show()



#dataFirstLine2 = dataFirstLine1.lstrip('').rstrip('"')
#dataFirstLine3 = str(dataFirstLine1)

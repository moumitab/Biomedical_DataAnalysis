#STEP 1 in comorbidity preprocessing

import pandas as pd
#We will create a new csv for hospital records with each of the  206 frequent conditions as columns
#we remove the codes not found in the master list of SNOMED codes
# Value in each of these column will be set to 1 if the conditon is present in the condition list
# else set to 0 if the condition is not present in the list

#Read hospital records and 206commonConditions file

#fileinput = '/Users/moumitabhattacharya/Google Drive/Research_Phase2/KidneyData/data/Hospital_Records.csv'Office_20160303_sorted_date_Male_Conditions.csv
fileinput = '/Users/moumitabhattacharya/Desktop/Research_Phase2/KidneyData/data/updatedData/Office_20160303_sorted_Run2_date.csv'
#/Users/moumitabhattacharya/Google Drive/Research_Phase2/Diagnosis_Analysis/Analysis_hospitalRecordSet/FrequencyListOfConditionsInHospitalRecordSet
fileinputConditions = '/Users/moumitabhattacharya/Desktop/Research_Phase2/KidneyData/data/updatedData/Office_20160303_sorted_Run2_SNOMED_450.csv'
fileoutput = '/Users/moumitabhattacharya/Desktop/Research_Phase2/KidneyData/data/updatedData/Office_20160303_sorted_Run2_SNOMED_450_Cond_Features.csv'

#Use Pandas to read CSV
data = pd.read_csv(fileinput)
dataConditions = pd.read_csv(fileinputConditions)
print(data.head())

dict_con = dict()
for i, row in dataConditions.iterrows():
    cond = row['conditions']
    cond = float(cond)
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
            if(eachCond != ' No matching concept'):
                sum2 = sum2 +1
                print(sum2)
                eachCond = float(eachCond)
                if eachCond in dict_con.keys():
                    sum3 = sum3 +1
                    data.loc[i, eachCond] = 1
print(sum3)
print("Out of the loop")
#Output
data.to_csv(fileoutput)


#Code commented and can be used only to generate the 188 unique condition set for features
# with open('C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\commonConditionListforColumns.csv','wb') as f:
#     w = csv.writer(f)
#     w.writerow(dict_con.keys())
#     w.writerow(dict_con.values())
#     w.writerow('\n')
#






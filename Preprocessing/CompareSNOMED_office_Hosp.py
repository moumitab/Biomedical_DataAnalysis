#We want to compare the list of SNOMED  codes in used for Topic Modelling in the Office
# Record set and in the hospital record set

#Insert the SNOMED codes in office record set and insert the SNOMED code in the Lab record set
import pandas as pd
import numpy as np
#/Users/moumitabhattacharya/Google Drive/Research_Phase2/Diagnosis_Analysis

fileHosp  = 'Users\moumitabhattacharya\Google Drive\Research_Phase2\Diagnosis_Analysis\Analysis_hospitalRecordSet\FrequencyListOfConditionsInHospitalRecordSet\HospitalCondList_400_conditions.csv'

fileOffice = 'Users\moumitabhattacharya\Google Drive\Research_Phase2\Diagnosis_Analysis\Analysis_hospitalRecordSet\FrequencyListOfConditionsInHospitalRecordSet\HOfficeConditionsList_680.csv'

fileOut = 'Users\moumitabhattacharya\Google Drive\Research_Phase2\Diagnosis_Analysis\Analysis_hospitalRecordSet' \
            '\FrequencyListOfConditionsInHospitalRecordSet\CommonCond_680_400.csv'

df_office = pd.read_csv(fileOffice,delimiter=',')
df_Hosp = pd.read_csv(fileHosp,delimiter=',')

office_list = df_office['SNOMED']
Hosp_list = df_Hosp['SNOMED']

officeDict = dict()
HospDict = dict()

for element in office_list:
    if(officeDict.has_key(element)):
        pass
    else:
        officeDict[element] = 'N'

for ele in Hosp_list:
    if(HospDict.has_key(ele)):
        pass
    else:
        HospDict[ele] = 'N'

count_Hosp = 0
count_Office = 0
commonConditionList = list()

for item in officeDict.keys():
    if(HospDict.has_key(item)):
        HospDict[item] = 'Y'
        count_Hosp = count_Hosp +1
        commonConditionList.append(item)

for items in HospDict.keys():
    if(officeDict.has_key(items)):
        officeDict[items] = 'Y'
        count_Office = count_Office+1

import csv
# Open File
resultFyle = open('C:\\Users\\Moumita\\Google Drive\\Research_Phase2\\Diagnosis_Analysis\\Analysis_hospitalRecordSet\\FrequencyListOfConditionsInHospitalRecordSet\\CommonCond.csv','wb')

# Create Writer Object
wr = csv.writer(resultFyle, dialect='excel')

# Write Data to File
for item in commonConditionList:
    wr.writerow([item,])

print("length of Hospital Dictionary",len(HospDict),"Number of common elements with office records",count_Hosp)
print("length of Office Dictionary",len(officeDict),"Number of common elements with office records",count_Office)




# for each patient ID create a dictionary of diseases
# And enter the counts for each disease
#Add the column entries for each disease for each patient
#Finally output this file with patient ID as the row names
# and Diagnosis codes as the column names
#STEP 4

import numpy as np
import pandas as pd

#fileinput1 = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\Diagnosis_Analysis\\RawData\\OfficeCondition_bow_noAllZero.csv'
fileinput1 = '/Users/moumitabhattacharya/Desktop/Research_Phase2/Diagnosis_Analysis/Gender_Comorbidity_study/Male/Male_426/Office_20160303_sorted_date_Male_Conditions_426_Features_no_NA.csv'
#fileoutput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\Diagnosis_Analysis\\RawData\\DTM.csv'
fileoutput = '/Users/moumitabhattacharya/Desktop/Research_Phase2/Diagnosis_Analysis/Gender_Comorbidity_study/Male/Male_426/Office_20160303__Conditions_426_Features_Male_BOW.csv'
df = pd.read_csv(fileinput1,delimiter=',')
patientID = df['person_id']
patientID = np.array(patientID)
patient_ID_unique = np.unique(patientID)
columns = list(df.columns)
columns = [c for c in columns if c not in ["person_id"]]

# for eachPatientID in patient_ID_unique:
#     for i, row in df.iterrows():
#         if(row['person_id'] == eachPatientID):
#             print('Hello')
#Groups the entire data w.r.t person_id and sums up all the columns
newData = df.groupby('person_id')[columns].sum()[0:]
newData.to_csv(fileoutput)
print(newData.shape)




#In this code we will subset the office and lab records into new record-sets with the
# all the patients present in the hospital records
import pandas as pd
import csv
import numpy as np

#Input the hospital unique patient list
file1 = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\uniquePatient_id_hospitalRecords.csv'
file2 = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\Lab_20151104_sorted.csv'
file2Out = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\Lab_20151104_sorted_patientHospitalized.csv'

#Dataframe for each input
uniquePatientIDdata = pd.read_csv(file1)
labRecordSetData = pd.read_csv(file2)

colnamesLab = list(labRecordSetData.columns.values)
print("colnamesLab",colnamesLab)
#columns = list(data.columns[2:])
print("colnamesLab",len(colnamesLab))
#Convert the unique patient list into a dictionary
uniqueID_dict = {}
for i, row in uniquePatientIDdata.iterrows():
    key = row['person_id']
    if key in uniqueID_dict:
        pass
    uniqueID_dict[key] = row['Frequency']
print("result.has_key(3)",uniqueID_dict.has_key(3))

#Subset the office Data set to contain only those patients who have been hospitalized atleast once or more

newLabRecordSet = pd.DataFrame(columns=colnamesLab)

j = 0
for i, row in labRecordSetData.iterrows():
    personID = row['\xef\xbb\xbfperson_id']
    #print("newOfficeRecordSet.loc[[j],'\xef\xbb\xbfperson_id']",newOfficeRecordSet.loc[j,'\xef\xbb\xbfperson_id'])
    if(uniqueID_dict.has_key(personID)):
        labRecordSetData.loc[j,'\xef\xbb\xbfperson_id'] = row['\xef\xbb\xbfperson_id']
        labRecordSetData.loc[j,'visit_occurrence_id'] = row['visit_occurrence_id']
        labRecordSetData.loc[j,'lab_report_date'] = row['lab_report_date']
        labRecordSetData.loc[j,'Gender'] = row['Gender']
        labRecordSetData.loc[j,'Race'] = row['Race']
        labRecordSetData.loc[j,'Ethnicity'] = row['Ethnicity']
        labRecordSetData.loc[j,'Age'] = row['Age']
        labRecordSetData.loc[j,'Albumin'] = row['Albumin']
        labRecordSetData.loc[j,'Alanine aminotransferase'] = row['Alanine aminotransferase']

        labRecordSetData.loc[j,'Alkaline phosphatase'] = row['Alkaline phosphatase']
        labRecordSetData.loc[j,'Aspartate aminotransferase'] = row['Aspartate aminotransferase']
        labRecordSetData.loc[j,'Bicarbonate'] = row['Bicarbonate']
        labRecordSetData.loc[j,'Bilirubin.direct'] = row['Bilirubin.direct']
        labRecordSetData.loc[j,'Bilirubin.indirect'] = row['Bilirubin.indirect']
        labRecordSetData.loc[j,'Bilirubin.total'] = row['Bilirubin.total']
        labRecordSetData.loc[j,'C reactive protein'] = row['C reactive protein']
        labRecordSetData.loc[j,'Calcium'] = row['Calcium']
        labRecordSetData.loc[j,'Calcium.ionized'] = row['Calcium.ionized']
        labRecordSetData.loc[j,'Carbon dioxide, total'] = row['Carbon dioxide, total']
        labRecordSetData.loc[j,'Chloride'] = row['Chloride']
        labRecordSetData.loc[j,'Cholesterol'] = row['Cholesterol']
        labRecordSetData.loc[j,'Cholesterol in HDL'] = row['Cholesterol in HDL']
        labRecordSetData.loc[j,'Creatinine'] = row['Creatinine']
        labRecordSetData.loc[j,'Erythrocyte'] = row['Erythrocyte']
        labRecordSetData.loc[j,'Fasting glucose'] = row['Fasting glucose']
        labRecordSetData.loc[j,'Ferritin'] = row['Ferritin']
        labRecordSetData.loc[j,'Gamma glutamyl transferase'] = row['Gamma glutamyl transferase']
        labRecordSetData.loc[j,'Glucose'] = row['Glucose']
        labRecordSetData.loc[j,'Hematocrit'] = row['Hematocrit']
        labRecordSetData.loc[j,'Hemoglobin'] = row['Hemoglobin']
        labRecordSetData.loc[j,'Hemoglobin A1c'] = row['Hemoglobin A1c']
        labRecordSetData.loc[j,'INR'] = row['INR']
        labRecordSetData.loc[j,'Iron'] = row['Iron']
        labRecordSetData.loc[j,'Iron binding capacity'] = row['Iron binding capacity']
        labRecordSetData.loc[j,'Lactate'] = row['Lactate']
        labRecordSetData.loc[j,'Lactate dehydrogenase 1'] = row['Lactate dehydrogenase 1']
        labRecordSetData.loc[j,'Leukocytes'] = row['Leukocytes']
        labRecordSetData.loc[j,'Lipoprotein.beta'] = row['Lipoprotein.beta']
        labRecordSetData.loc[j,'Parathyrin.intact'] = row['Parathyrin.intact']
        labRecordSetData.loc[j,'Phosphate'] = row['Phosphate']
        labRecordSetData.loc[j,'Platelets'] = row['Platelets']
        labRecordSetData.loc[j,'Potassium'] = row['Potassium']
        labRecordSetData.loc[j,'Prealbumin'] = row['Prealbumin']
        labRecordSetData.loc[j,'Sodium'] = row['Sodium']
        labRecordSetData.loc[j,'Transferrin.carbohydrate'] = row['Transferrin.carbohydrate']
        labRecordSetData.loc[j,'Triglyceride'] = row['Triglyceride']
        labRecordSetData.loc[j,'Urea nitrogen'] = row['Urea nitrogen']

        j = j +1
        print("j",j)
        #print("newOfficeRecordSet",newOfficeRecordSet)

#print("newOfficeRecordSet",newOfficeRecordSet)
labRecordSetData.to_csv(file2Out)



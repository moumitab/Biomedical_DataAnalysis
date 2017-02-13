#Remove rows will all zeros from the CSV file with leftmost column as patient ID and first row as conditions
import pandas as pd

fileinput1 = '/Users/moumitabhattacharya/Desktop/Research_Phase2/Diagnosis_Analysis/Gender_Comorbidity_study/Male/Male_426/Office_20160303__Conditions_426_Features_Male_BOW.csv'
fileoutput = '/Users/moumitabhattacharya/Desktop/Research_Phase2/Diagnosis_Analysis/Gender_Comorbidity_study/Male/Male_426/Office_20160303__Conditions_426_Features_Male_BOW_noAllZero.csv'
data = pd.read_csv(fileinput1,delimiter=',')
data.index = data['person_id']
#data.set_index('person_id')

columns = list(data.columns)
columns = [c for c in columns if c not in ["person_id"]]
df = data[columns]
print(list(df.index))
print(df)
print(len(df))
df = df[(df.T != 0).any()]

print(len(df))
print(df)

df.to_csv(fileoutput)

#This file just maps the SNOMED codes to the description provided in the Data Dictionary
#Input files used are Office_Frequency_conditons.csv and SNOMED_CODE.csv
#STEP 3 in comorbidity pre-processing studies
import csv

f = open('/Users/moumitabhattacharya/Desktop/Research_Phase2/Data Dictionary/SNOMED_CODE.csv')
reader = csv.reader(f)
result = {}
for row  in reader:
    key = row[0]
    if key in result:
        pass
    result[key] = row[1:]
#/Users/moumitabhattacharya/Desktop/Research_Phase2/Diagnosis_Analysis/Gender_Comorbidity_study/Male/Male_426

with open('/Users/moumitabhattacharya/Desktop/Research_Phase2/KidneyData/data/updatedData/condtions_480.csv','r') as csvinput:
    with open('/Users/moumitabhattacharya/Desktop/Research_Phase2/KidneyData/data/updatedData/condtions_480_Desc.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput)
        for row in csv.reader(csvinput):
            if result.get(row[0]):
                description = result.get(row[0])
                writer.writerow(row + description)
            else:
                writer.writerow(row + ['code Not found'])

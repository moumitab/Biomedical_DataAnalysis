#Input the records with ones and empty cells and replace with 1 and 0
#STEP 2 in comorbidity preprocessing

import pandas as pd

#fileInput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\records_date\\HospitalRecords_105TopConditionFeatures.csv'

fileInput = '/Users/moumitabhattacharya/Desktop/Research_Phase2/KidneyData/data/updatedData/office_records_448_codes.csv'
#fileInput = '/Users/moumitabhattacharya/Desktop/Research_Phase2/Diagnosis_Analysis/Gender_Comorbidity_study/Male/Office_20160303_Male_BOW.csv'
fileOutput = '/Users/moumitabhattacharya/Desktop/Research_Phase2/KidneyData/data/updatedData/office_records_448_codes_no_NA.csv'

data = pd.read_csv(fileInput,delimiter=',')
features_train = list(data.columns)
X = data[features_train]

X.fillna(0, inplace=True)
data[features_train] = X

data.to_csv(fileOutput)

#Try out different code to fill in the empty cells of a .csv file with zeros
import csv
import sys

#1. Place each record of a file in a list.
#2. Iterate thru each element of the list and get its length.
#3. If the length is less than one replace with value x.


# reader = csv.reader(open(sys.argv[1], "rb"))
# for row in reader:
#     for x in row[:]:
#                 if len(x)< 1:
#                          x = 0
#                 print x
# print row

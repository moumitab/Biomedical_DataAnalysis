#Input the OfficeRecords_30TopConditionFeatures and for each of the condition column enter 0 if NULL/empty

import pandas as pd

#fileInput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\records_date\\HospitalRecords_105TopConditionFeatures.csv'
fileInput = 'C:\\Users\\Moumita\\Google Drive\\Research_Phase2\\Diagnosis_Analysis\\Analysis_hospitalRecordSet' \
            '\\FrequencyListOfConditionsInHospitalRecordSet\\HospitalRecordSetWithConditionFeatures.csv'
fileOutput = 'C:\\Users\\Moumita\\Google Drive\\Research_Phase2\\Diagnosis_Analysis\\Analysis_hospitalRecordSet\\FrequencyListOfConditionsInHospitalRecordSet\\HospitalRecordSetWithConditionFeatures_final.csv'

data = pd.read_csv(fileInput,delimiter=',')
features_train = list(data.columns[8:])
X = data[features_train]

X.fillna('', inplace=True)
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

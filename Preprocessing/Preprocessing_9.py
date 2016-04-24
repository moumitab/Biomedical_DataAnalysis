#Create a data structure for O(1) search of all the patient_id of all the patients hospitalized

import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier

fileinput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\Hospital_20151104_sorted.txt'
colnames = ['person_id','visit_occurrence_id','visit_start_date_time','visit_end_date_time','LengthOfStayHours','visit_type','Gender','Race',
            'Ethnicity','Age','conditions']
data = pd.read_csv(fileinput,delim_whitespace=True, names=colnames, header = None)
dataFrame = pd.DataFrame(data)
unique_id = pd.unique(dataFrame['person_id'].ravel())
print(len(unique_id))
dict = {}
for i in range(1, len(unique_id)):
    key = unique_id[i]
    print(key)
    dict.update({key:1})
    key = ''

print(dict.keys())

with open('C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\uniquePatient_id_hospitalRecords.csv','wb') as f:
    w = csv.writer(f)
    w.writerow(dict.keys())
    w.writerow(dict.values())
    w.writerow('\n')


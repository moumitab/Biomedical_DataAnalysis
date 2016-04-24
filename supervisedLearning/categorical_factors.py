from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
def encoder(df):

    var_mod = ['Gender','Race','Ethnicity','visit_type']
    le = LabelEncoder()
    for i in var_mod:
        df[i] = le.fit_transform(df[i])


    print(df.dtypes)
    return df

fileIutput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\records_date\\HospitalRecords_105TopConditionFeatures.csv'
data = pd.read_csv(fileIutput,delimiter=',',na_values = '')
df = encoder(data)
fileout = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\records_date\\HospitalRecords_105TopConditionFeatures_factored.csv'
df.to_csv(fileout)
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
def encoder(df):

    var_mod = ['Gender','Race','Ethnicity']
    le = LabelEncoder()
    for i in var_mod:
        df[i] = le.fit_transform(df[i])
    df.dtypes
    return df

fileIutput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\\updatedData\\OfficeRecords_139TopConditionFeatures.csv'
data = pd.read_csv(fileIutput,delimiter=',',na_values = '')
df = encoder(data)
fileout = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\\updatedData\\OfficeRecords_139TopConditionFeatures_factored.csv'
df.to_csv(fileout)
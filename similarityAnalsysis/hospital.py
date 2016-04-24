
import pandas as pd
import numpy as np
inputFile = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\Hospital_20151104_sorted_2.csv'
df = pd.read_csv(inputFile)
print df.describe()
#df['age'].hist(bins = 10)
conditions = df['conditions']



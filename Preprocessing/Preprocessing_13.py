import pandas as pd
fileinput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\Hospital_ConditonFeatures.csv'
data = pd.read_csv(fileinput,index_col = 'visit_occurrence_id', parse_dates = True)
print(data.columns)


#Reordering the colums to have the class lable (visit_type)as the left most column
import pandas as pd
import numpy as np

fileInput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\preprocessed\\Hospital_ConditonFeatures_2.csv'
data = pd.read_csv(fileInput,index_col = 'visit_occurrence_id', parse_dates = True)
df = pd.DataFrame(data)
sequence = ['visit_type','visit_occurrence_id','person_id','visit_start_date_time','visit_end_date_time','LengthOfStayHours'
    'Gender','Race','Ethnicity','Age','conditions','59621000','44054006','55822004','194774006','42343007','235595009'
    ,'49436004','40930008','235595009','44054006','59621000','55822004','194774006','42343007','49436004','40930008']
your_dataframe = df.reindex(columns=sequence)
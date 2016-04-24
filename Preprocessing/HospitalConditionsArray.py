import pandas as pd
import numpy as np

def getData():
    fileinput = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\KidneyData\\data\\Hospital_20151104_sorted_2.csv'
    conditionList =  'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\conditions_clustering_analysis\\197Condtions_hospitalRecords.csv'

    data = pd.read_csv(fileinput)
    dataConditions = pd.read_csv(conditionList)

    print("col names",data.columns.values)
    return(data, dataConditions)

def conditionDict(dataConditions):
    dict_con = dict()
    Conditions = dataConditions.iloc[0,1]
    for i, row in dataConditions.iterrows():
        cond = row['SNOMED ']
        if cond not in dict_con:
            dict_con[cond] = 1
        else:
            dict_con[cond] = dict_con[cond] + 1

    return dict_con

def makeConditionVectors(dict_con,data,dataConditions):
    dataSub = data[["\xef\xbb\xbfperson_id", 'visit_occurrence_id','conditions']]
    for i, row in  data.iterrows():
        conditionList = row['conditions']
        if(str(conditionList).lower() != 'nan'):
            conditionList = conditionList.split(',')
            for eachCondition in conditionList:
                eachCondition = int(eachCondition)
                if eachCondition in dict_con.keys():
                    data.loc[i,eachCondition] = 1
    print data.head
    writeCSV(data)


def writeCSV(data):
    conditionVec =  'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\conditions_clustering_analysis\\HospitalCondtionsVectors.csv'
    data.to_csv(conditionVec)



if __name__ == "__main__":
    data, conditionList = getData()
    dict = conditionDict(conditionList)
    makeConditionVectors(dict,data,conditionList)

import pandas as pd
import numpy as np

#This code ranks cosine similarity complete matrix to provide ranks in ascending  order for
#all the conditions
#This will help us recognize which conditions co-occurs most with which other conditions
#based on the hospital record set

def minInEachCol(data):
    columns = list(data.columns)
    print(columns[0])
    print(columns[1])
    newRankMat = pd.DataFrame()
    for i, row in data.iterrows():
        newColName = i + "_rank"
        print(newColName)
        newRankMat[newColName] = row.rank(ascending=1)
    output(newRankMat)


def output(newRankMat):
    outputFile  = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\conditions_clustering_analysis\\SimilarityRankMatrix.csv'
    newRankMat.to_csv(outputFile)



inputfile = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\conditions_clustering_analysis\\cosineSimilarityCompleteMatrix.csv'
data = pd.read_csv(inputfile, index_col='SNOMED')
minInEachCol(data)

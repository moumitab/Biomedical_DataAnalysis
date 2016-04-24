import numpy as np
import pandas as pd

inputFile = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\conditions_clustering_analysis\\cosineSimilarity_179_run8.csv'
df = pd.read_csv(inputFile,index_col='SNOMED')
print(df.shape)
data = df.as_matrix()
dataTranspose = np.transpose(data)
similarityMatrix = data + dataTranspose
print(similarityMatrix)
import csv
with open('C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\conditions_clustering_analysis\\cosineSimilarityCompleteMatrix2.csv','w') as csvoutput:
    for i in range(len(similarityMatrix)):
        writer = csv.writer(csvoutput)
        writer.writerow(similarityMatrix[i])

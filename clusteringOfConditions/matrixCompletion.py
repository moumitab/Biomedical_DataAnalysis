import numpy as np
import pandas as pd

inputFile = '/Users/moumitabhattacharya/Google Drive/Research_Phase2/conditions_clustering_analysis/Office_Condition/ofc_179_code_cosineSimilarity.csv'
df = pd.read_csv(inputFile,index_col='SNOMED')
print(df.shape)
data = df.as_matrix()
dataTranspose = np.transpose(data)
similarityMatrix = data + dataTranspose
print(similarityMatrix)
import csv
with open('/Users/moumitabhattacharya/Google Drive/Research_Phase2/conditions_clustering_analysis/Office_Condition/ofc_179_code_cosineSimilarity_matrix_completion.csv','w') as csvoutput:
    for i in range(len(similarityMatrix)):
        writer = csv.writer(csvoutput)
        writer.writerow(similarityMatrix[i])

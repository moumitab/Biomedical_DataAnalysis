import pandas as pd
import numpy as np

#import numpy.linalg.norm as norm

def getData():
    #conditionVec =  'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\conditions_clustering_analysis\\Condtions_Vectors_Hoss_Input.csv'

    conditionVec = '/Users/moumitabhattacharya/Google Drive/Research_Phase2/conditions_clustering_analysis/Office_Condition/OfficeConditionOutputVec.csv'
    #conditionVec = 'C:\\Users\\Moumita\\Dropbox\\Transcript_RNA\\CAPNS1_tissue_normalizeddata_2.csv'
    df = pd.read_csv(conditionVec)
    data = df.as_matrix()
    data = np.transpose(data)
    n_samples, n_features = data.shape
    print("n_samples",n_samples)
    print("n_features",n_features)
    return data

def _validate_vector(u, dtype=None):
    # XXX Is order='c' really necessary?
    u = np.asarray(u, dtype=dtype, order='c').squeeze()
    # Ensure values such as u=1 and u=[1] still return 1-D arrays.
    u = np.atleast_1d(u)
    if u.ndim > 1:
        raise ValueError("Input vector should be 1-D.")
    return u


def cosine(u, v):
    """
    Computes the Cosine distance between 1-D arrays.
    The Cosine distance between `u` and `v`, is defined as
    .. math::
       1 - \\frac{u \\cdot v}
                {||u||_2 ||v||_2}.
    where :math:`u \\cdot v` is the dot product of :math:`u` and
    :math:`v`.
    Parameters
    ----------
    u : (N,) array_like
        Input array.
    v : (N,) array_like
        Input array.
    Returns
    -------
    cosine : double
        The Cosine distance between vectors `u` and `v`.
    """
    u = _validate_vector(u)
    v = _validate_vector(v)
    dist = round(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)),3)
    return dist



data = getData()
similarityMatrix = np.zeros((179, 179))

print data[0,:]
print len(data[0,:])
print(len(data))
for i in range (len(data)):
    u = data[i,:]
    for j in range(i,len(data)):
        v = data[j]
        print(i,j)
        dist = cosine(u,v)
        similarityMatrix[i,j] = dist

print("similarityMatrix.shape",similarityMatrix.shape)
#
# similarityMatrix.to_csv(outputfile)


#similarityMatrixOut =  'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\conditions_clustering_analysis\\cosineSimilarity_179_run5.csv'
##This piece works just using the csv write option
# np.savetxt(similarityMatrixOut,
#            similarityMatrix,
#            delimiter=',',
#            fmt='%10.5f',
#            header='Created by Numpy')

import csv

with open('/Users/moumitabhattacharya/Google Drive/Research_Phase2/conditions_clustering_analysis/Office_Condition/ofc_179_code_cosineSimilarity.csv','w') as csvoutput:
    for i in range(len(similarityMatrix)):
        writer = csv.writer(csvoutput)
        writer.writerow(similarityMatrix[i])



#similarityMatrix.tofile(similarityMatrixOut,sep=',',format='%10.5f')
    # restOfData = data - eachData
    # print restOfData
    # for eachColumn in restOfData:
    #     print(eachColumn)

#dist = cosine(u,v)
#print(dist)
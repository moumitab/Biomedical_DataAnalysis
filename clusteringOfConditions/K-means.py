#Here we implement simple K-means clustering to cluster the conditions vectors
import pandas as pd
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist,pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

def getData():
    conditionVec =  'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\conditions_clustering_analysis\\Condtions_Vectors_Hoss_Input.csv'
    data = pd.read_csv(conditionVec)
    columns = list(data.columns)
    X = data[columns]
    n_samples, n_features = data.shape
    print("n_samples",n_samples)
    print("n_features",n_features)
    return X

def bench_k_means(data):
    #Determine the range of K
    k_range = (5,6)

    #Fit the k-means model for each n_clusters = k
    k_means_var = [KMeans(n_clusters=k).fit(data) for k in k_range]

    #Pull out the cluster center for each model
    centroids = [X.cluster_centers_ for X in k_means_var]

    labels = [Y.labels_ for Y in k_means_var]
    print("labels",labels)
    print(len(labels))
    #Calculate the silhouette scores
    sh_scores = [silhouette_score(data,lab) for lab in labels]

    #Calculate the euclidean distance from each point to each cluster center
    k_euclid = [cdist(data,cent,'euclidean') for cent in centroids]
    dist = [np.min(ke,axis=1) for ke in k_euclid ]

    #Total within-cluster sum of squares
    wcss = [sum(d**2)for d in dist]
    print("Total within-cluster sum of squares",wcss)

    print("silhouette_score",sh_scores)


    #The total sum of squares
    #tss = sum(pdist(data)**2)/data.shape[0]

    #The between clusters sum of squares
    #bss = tss - wcss

    #print(bss)

if __name__ == '__main__':
    data = getData()
    bench_k_means(data)





import pandas as pd
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def getData():
    conditionVec =  'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\conditions_clustering_analysis\\Condtions_Vectors_Hoss_Input_2.csv'
    data = pd.read_csv(conditionVec)
    columns = list(data.columns)
    X = data[columns]
    n_samples, n_features = data.shape
    print("n_samples",n_samples)
    print("n_features",n_features)
    return X

def form_clusters(x,k):
    """
    Build clusters
    """
    #K = required number of clusters
    no_clusters = k
    model = KMeans(n_clusters=no_clusters,init='random')
    model.fit(x)
    labels = model.labels_
    #Pull out the cluster center for each model
    centroids = model.cluster_centers_

    print(labels)
    #Calculate the silhouette score
    sh_score = model.scores(x,labels)
    #silhouette_score(x,labels,metric='euclidean',sample_size=300)
    return labels,centroids,sh_score

def plotSilhouette(sh_scores):

    no_clusters = [i+1 for i in range(1,5)]
    plt.figure(2)
    plt.plot(no_clusters, sh_scores)
    plt.title("cluster quality")
    plt.xlabel("No of clusters k")
    plt.ylabel("silhouette coefficient")
    plt.show()

if __name__ == '__main__':
     data = getData()
     labels = []
     centroids = []
     sh_scores = []
     for i in range(10,11):
         label,centroid,sh_score = form_clusters(data,i+1)
         sh_scores.append(sh_score)
         labels.append(label)
         centroids.append(centroid)

     fileout = 'C:\\Users\\Moumita\\Dropbox\\Research_Phase2\\conditions_clustering_analysis\\clusterCentroidLabels.csv'
     labels = np.array(labels)
     centroids = np.array(centroids)
     plotSilhouette(sh_scores)
     print("lables",labels)
     print("centroid",centroids)
     plt.plot(centroids)
     plt.show()
     #labels = pd.DataFrame(labels)
     #centroids = pd.DataFrame(centroids)



from time import time

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering

missing_value=['?']
data= pd.read_csv("wiki4HE.csv", na_values=missing_value)
print(data.isnull().sum())

data.dropna(inplace=True)
print(data.shape)
X=data.values
y=X[:,2]
X_edit=np.delete(X,[2],1)

from sklearn import metrics
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans_model.labels_

KMeans_Sil = metrics.silhouette_score(X, labels, metric='euclidean')
print(KMeans_Sil)



for linkage in ('ward', 'average', 'complete', 'single'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=6)
    t0 = time()
    clustering.fit(X)
    print("%s :\t%.2fs" % (linkage, time() - t0))
    labels = clustering.labels_
    Agglo_Sil = metrics.silhouette_score(X, labels, metric='euclidean')
    print(Agglo_Sil)


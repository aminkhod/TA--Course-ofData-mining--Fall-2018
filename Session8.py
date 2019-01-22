
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
# from scipy.cluster.hierarchy import dendrogram, linkage
# from matplotlib import pyplot as plt
#
# linked = linkage(X_edit, 'ward')
#
# # labelList = range(1, 11)
#
# plt.figure(figsize=(10, 7))
# plt.title('Hierarchical Clustering Dendrogram (truncated)')
# dendrogram(linked,
#             orientation='top',
#             labels=y,
#             distance_sort='descending',
#             show_leaf_counts=True)
# plt.show()


########Silhouette Coefficient
from sklearn import metrics
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans_model.labels_

KMeans_Sil = metrics.silhouette_score(X, labels, metric='euclidean')
print(KMeans_Sil)




linkage = "single"
n_clusters = 4
model = AgglomerativeClustering(linkage=linkage, n_clusters=n_clusters).fit(X)
labels = model.labels_

Agglo_Sil = metrics.silhouette_score(X, labels, metric='euclidean')
print(Agglo_Sil)







################Visualisation for Agglomerative Clustering

n_samples, n_features = X_edit.shape


digits = datasets.load_digits(n_class=10)
X = digits.data
y = digits.target

# np.random.seed(0)
# def nudge_images(X, y):
#     # Having a larger dataset shows more clearly the behavior of the
#     # methods, but we multiply the size of the dataset only by 2, as the
#     # cost of the hierarchical clustering methods are strongly
#     # super-linear in n_samples
#     shift = lambda x: ndimage.shift(x.reshape((8, 8)),
#                                   .3 * np.random.normal(size=2),
#                                   mode='constant',
#                                   ).ravel()
#     X = np.concatenate([X, np.apply_along_axis(shift, 1, X)])
#     Y = np.concatenate([y, y], axis=0)
#     return X, Y
#
#
# X_edit, y = nudge_images(X_edit, y)



#----------------------------------------------------------------------
###### Visualize the clustering

def plot_clustering(X_red, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                 color=plt.cm.nipy_spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

#----------------------------------------------------------------------
# 2D embedding of dataset
print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X_edit)
print("Done.")



for linkage in ('ward', 'average', 'complete', 'single'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=6)
    t0 = time()
    clustering.fit(X)
    print("%s :\t%.2fs" % (linkage, time() - t0))
    labels = clustering.labels_
    Agglo_Sil = metrics.silhouette_score(X, labels, metric='euclidean')
    print(Agglo_Sil)

plt.show()


1+1
import numpy as np

#######Feature selection#########

#Removing features with low variance
from sklearn.feature_selection import VarianceThreshold
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
newX= sel.fit_transform(X)


#Univariate feature selection
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest,SelectPercentile
from sklearn.feature_selection import chi2
iris = load_iris()
X, y = iris.data, iris.target
print(X.shape)
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
print(X_new.shape)

X_newpercent = SelectPercentile(chi2, percentile=70).fit_transform(X, y)
print(X_newpercent.shape)



## Feature Extraction with RFE(recursive feature elimination)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 2)
fit = rfe.fit(X, y)
print("Num Features: ",  int(fit.n_features_))
print("Selected Features: ",fit.support_)
print("Feature Ranking: ",  fit.ranking_)


# Feature Importance with Extra Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier
# feature extraction
model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_)


# Tree-based feature selection

from sklearn.feature_selection import SelectFromModel
print(X.shape)
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)
print(clf.feature_importances_)
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(X)
print(X_new.shape)



from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_fit=pca.fit(X)
print(pca_fit.components_)
print("The amount of variance explained by each of the selected components : \n", pca.explained_variance_ratio_)
print("*" * 50)
print("The summation amount of variance explained by each of the selected components : \n", np.sum(pca.explained_variance_ratio_))
pca_x = pca.transform(X)

from sklearn import linear_model
lm = linear_model.LinearRegression(normalize=True)
lm.fit(pca_x, y)
predicted = lm.predict(pca_x)

from sklearn.metrics import r2_score, mean_squared_error
print("MSE: ", mean_squared_error(y, predicted))
print("R2 score: ", r2_score(y, predicted))

B = lm.coef_
B0  = lm.intercept_
model = lambda X: B0 + np.dot(X, B)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
if len(pca_x[0]) == 2 :
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pca_x[:, 0], pca_x[:, 1], y, c='green')
    ax.plot3D(pca_x[:, 0], pca_x[:, 1], model(pca_x), 'r')
    plt.show()


1+1
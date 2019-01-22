import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
# from sklearn import svm
# from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt



################################################
############ Cross Validation ###########
################################################

########### train_test_split

from sklearn.model_selection import train_test_split
boston = datasets.load_boston()
print(boston.data.shape, boston.target.shape)

X_train, X_test, y_train, y_test = train_test_split(
    boston.data, boston.target, test_size=0.1, random_state=0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

regr = LinearRegression()
regr.fit(X_train,y_train)
print(regr.score(X_train,y_train))
predictions= regr.predict(X_test)


##plot
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")

# model = svm.SVC(kernel='linear', C=1)
# scores = cross_val_score(model , boston.data, boston.target, cv=10)
# print(scores)


############ KFold
from sklearn.model_selection import KFold
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
# y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=5)
kfsplit = kf.get_n_splits(boston.data)

print(kf)

kfoldregr = LinearRegression()
# kfoldregr.fit(X_train,y_train)
# print(kfoldregr.score(X_train,y_train))
# print(kfoldregr.predict(X_test))
KFoldScore=[]

for train_index, test_index in kf.split(boston.data):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = boston.data[train_index], boston.data[test_index]
    y_train, y_test = boston.target[train_index], boston.target[test_index]
    kfoldregr.fit(X_train, y_train)
    KFoldScore.append(kfoldregr.score(X_train, y_train))

print(KFoldScore)

######## Leave One Out Cross Validation
from sklearn.model_selection import LeaveOneOut
X = np.array([[1, 2], [3, 4]])
y = np.array([1, 2])
loo = LeaveOneOut()
loo.get_n_splits(X)


for train_index, test_index in loo.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   print(X_train, X_test, y_train, y_test)


##########  Little sample of cross validation
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
lm = LinearRegression()

scores = cross_val_score(lm, boston.data, boston.target, cv=6)
print("Cross-validated scores:", scores)

predictions = cross_val_predict(lm,boston.data, boston.target, cv=6)
plt.scatter(boston.target, predictions)

r2_score = metrics.r2_score(boston.target, predictions)
print("Cross-Predicted R^2",r2_score)


##########################################
############## preprocessing
##########################################
from sklearn import preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    boston.data, boston.target, test_size=0.4, random_state=0)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
clf = regr.fit(X_train_transformed,  y_train)
X_test_transformed = scaler.transform(X_test)
print(clf.score(X_train_transformed,y_train))
print(regr.predict(X_test_transformed))

5+2
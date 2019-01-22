import numpy as np
import pandas as pd


data1 = pd.read_csv("breast-cancer.csv")


X=data1.values[:,1:10]
y=data1.values[:,10]
y=np.array([1 if yinstance==4 else 0 for yinstance in y ])

missing_value=['?']
data2= pd.read_csv("breast-cancer.csv", na_values=missing_value)
print(data2.isnull().sum())

nanArray=data2['Bare Nuclei'].isnull()
nanind=[]
for ind in range( len(data2['Bare Nuclei'])):
		if (nanArray[ind]==True):
			nanind.append(ind)
print(nanind)

# ##removing missing values
# data2=data2.drop(nanind)


##replacing
bmedian = data2['Bare Nuclei'].median()
data2['Bare Nuclei'].fillna(bmedian,inplace=True)

X=data2.values[:,1:10]
y=data2.values[:,10]
y=np.array([1 if yinstance==4 else 0 for yinstance in y ])



### split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)




####LogisticRegression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
LogregModel = logreg.fit(X_train, y_train)
predicts=LogregModel.predict(X_train)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_train, predicts))
print("Error:",1-metrics.accuracy_score(y_train, predicts))


####naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnbModel =gnb.fit(X_train, y_train)




####Linear Regression
from sklearn.linear_model import LinearRegression
LR= LinearRegression()
LRmodel = LR.fit(X_train, y_train)
y_regpred = LRmodel.predict(X_test)
y_pred= [1 if x>=0.4 else 0 for x in y_regpred]

########Confusion matrix

cnf_matrix = metrics.confusion_matrix(y_pred , y_test)
print(cnf_matrix)

#####Metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test , y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))



###K_fold
#######LogisticRegression
from sklearn.model_selection import KFold
kf = KFold(n_splits=10,shuffle=True)
kfsplit = kf.get_n_splits(X)
kfoldlogreg =LogisticRegression()
KFoldACC = []
KFoldPREC = []
KFoldREC = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    kfoldlogreg.fit(X_train, y_train)
    logpredict=kfoldlogreg.predict(X_test)
    KFoldACC.append(metrics.accuracy_score(y_test,logpredict))
    KFoldPREC.append(metrics.precision_score(y_test,logpredict))
    KFoldREC.append(metrics.recall_score(y_test,logpredict))

print("Accuracy for logistic-10foldCV",np.mean(KFoldACC))
print("Precsion for logistic-10foldCV",np.mean(KFoldPREC))
print("Recall for logistic-10foldCV",np.mean(KFoldREC))


###LinearRegression
kfoldregr = LinearRegression()
acclist=[]
prelist=[]
reclist=[]
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    kfoldregrmodel = kfoldregr.fit(X_train, y_train)

    predictions = kfoldregr.predict(X_test)
    regrpredic = np.array([1 if x >= 0.5 else 0 for x in predictions])

    acc = metrics.accuracy_score(y_test, regrpredic)
    acclist.append(acc)

    recall = metrics.recall_score(y_test, regrpredic)
    reclist.append(recall)

    precession = metrics.precision_score(y_test, regrpredic)
    prelist.append(precession)


print('Kfold accuracy',np.mean(acclist))
print('Kfold precession',np.mean(prelist))
print('Kfold recall: ',np.mean(reclist))




####Leave One Out Cross Validation

##Logistic Regression
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo.get_n_splits(X)
loologreg = LogisticRegression()
predicts=[]
for train_index, test_index in loo.split(X):
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   loologregModel =loologreg.fit(X_train, y_train)
   predicts.append(loologreg.predict(X_test))

predict = np.array(predicts)
cnf_matrix = metrics.confusion_matrix(y, predict)

print(cnf_matrix)

acc = metrics.accuracy_score(y, predict)
print("Logistic Regression Accuracy by LOOCV", acc)

recall = metrics.recall_score(y, predict)
print("Logistic Regression Recall by LOOCV", recall)

precession = metrics.precision_score(y, predict)
print("Logistic Regression Precession by LOOCV", precession)



1+1


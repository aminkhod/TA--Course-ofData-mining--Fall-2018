import pandas as pd
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix

###reading data
missing_value=['?']
data2= pd.read_csv("breast-cancer.csv", na_values=missing_value)

##replacing
bmedian = data2['Bare Nuclei'].median()
data2['Bare Nuclei'].fillna(bmedian,inplace=True)

X=data2.values[:,1:10]
y=data2.values[:,10]
y=np.array([1 if yinstance==4 else 0 for yinstance in y ])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(15,), random_state=0)
clf.fit(X_train, y_train)

predict =clf.predict(X_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, predict))
print("Precision:",metrics.precision_score(y_test , predict))
print("Recall:",metrics.recall_score(y_test, predict))
print("F_measure:",metrics.f1_score(y_test, predict))


fpr, tpr, thresholds = metrics.roc_curve(y_test,predict)
mlpauc= metrics.auc(fpr, tpr)
print(mlpauc)
fpr, tpr, thresholds = metrics.precision_recall_curve(y_test,predict)
mlpaupr= metrics.auc(fpr, tpr)
print(mlpaupr)

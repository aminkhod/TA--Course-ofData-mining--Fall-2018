from sklearn import tree
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn import metrics

enb_frame = pd.read_csv('ENB2012_data.csv')
enb=enb_frame.values
X = enb[:768,:8]
y = enb[:768,8]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test= train_test_split(X, y,test_size=0.2 )

model = tree.DecisionTreeRegressor(max_depth=5)
tree_model = model.fit(X_train, y_train)
predict = tree_model.predict(X_train)

# metrics
from sklearn.metrics import r2_score, mean_squared_error
print("MSE error is :", mean_squared_error(y_train, predict))
print("r2 score is :", r2_score(y_train, predict))

from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(X_train, y_train)
predictReg = regr.predict(X_train)

print("MSE error is :", mean_squared_error(y_train, predictReg))
print("r2 score is :", r2_score(y_train, predictReg))


###reading data
Autism=pd.read_csv('Autism-Adult.csv')
nanind=[]
for ind in range( len(Autism['age numeric'])):
		if (Autism['age numeric'][ind]=='?'):
			nanind.append(ind)

Autism=Autism.drop(nanind)
X=Autism.values[:,:12]
y=Autism.values[:,12]
y_edit= np.array([1 if yinstance=='yes' else 0 for yinstance in y ])
X[:,11]=np.array([1 if xinstance=='f' else 0 for xinstance in X[:,11] ])
X_train,X_test,y_train,y_test=train_test_split(X,y_edit,test_size=0.2,random_state=0)


### tree based methods for classification
# criterion = 'gini' or 'entropy'
tree_classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=10)
tree_model = tree_classifier.fit(X_train, y_train)
predict = tree_model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, predict))
print("Precision:",metrics.precision_score(y_test, predict))
print("Recall:",metrics.recall_score(y_test, predict))
print("F_measure:",metrics.f1_score(y_test, predict))
print( "classification error is :", np.sum(predict != y_test) / len(y_test) )


########## Bagging method  ######
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
bagging = BaggingClassifier()
bagging_model = bagging.fit(X_train, y_train)

bagging_predict = bagging_model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, bagging_predict ))
print("Precision:",metrics.precision_score(y_test, bagging_predict ))
print("Recall:",metrics.recall_score(y_test, bagging_predict ))
print("F_measure:",metrics.f1_score(y_test, bagging_predict ))
print( "classification error is :", np.sum(bagging_predict  != y_test) / len(y_test) )


########### Random Forest  ###########

random_forest = RandomForestClassifier(min_samples_split=5, min_samples_leaf=2, max_depth=10)
random_forest_model = random_forest.fit(X_train, y_train)

random_forest_predict = random_forest_model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, random_forest_predict))
print("Precision:",metrics.precision_score(y_test, random_forest_predict ))
print("Recall:",metrics.recall_score(y_test, random_forest_predict ))
print("F_measure:",metrics.f1_score(y_test, random_forest_predict ))
print( "classification error is :", np.sum(random_forest_predict  != y_test) / len(y_test) )

svmachine = svm.SVC(gamma='auto',kernel='linear')
svm_model = svmachine.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F_measure:",metrics.f1_score(y_test, y_pred))
print( "classification error is :", np.sum(svm_model.predict(X_test) != y_test) / len(y_test) )

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logregmodel = logreg.fit(X_train, y_train)
y_pred = logregmodel.predict(X_test)



########Confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_pred , y_test )
print(cnf_matrix)

#####Metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))





##########Naive Bayes #########
from sklearn.naive_bayes import GaussianNB

GNB = GaussianNB()
GNBmodel = GNB.fit(X_train, y_train)
y_pred = GNBmodel.predict(X_test)

########Confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_pred,y_test )
print(cnf_matrix)

#####Metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F_measure:",metrics.f1_score(y_test, y_pred))



######QDA ###########
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
QDA= QuadraticDiscriminantAnalysis()
QDAmodel = QDA.fit(X_train, y_train)
y_pred = QDAmodel.predict(X_test)

########Confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)

#####Metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test , y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F_measure:",metrics.f1_score(y_test, y_pred))



######LDA ###########
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
LDA= LinearDiscriminantAnalysis()
LDAmodel = LDA.fit(X_train, y_train)
y_pred = LDAmodel.predict(X_test)

########Confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_pred ,y_test )
print(cnf_matrix)

#####Metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test , y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F_measure:",metrics.f1_score(y_test, y_pred))


###### Classification with Linear Regression######
from sklearn.linear_model import LinearRegression
LR= LinearRegression()
LRmodel = LR.fit(X_train, y_train)
y_regpred = LRmodel.predict(X_test)
y_pred= [1 if x>=0.4 else 0 for x in y_regpred]

########Confusion matrix
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_pred , y_test)
print(cnf_matrix)

#####Metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test , y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F_measure:",metrics.f1_score(y_test, y_pred))


1+1

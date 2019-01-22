####################Multi-layer Perceptron###################


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



###### Devide data to test and train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


############## Regression with MLP#############
#######sklearn.neural_network.MLPRegressor
from sklearn.neural_network import MLPRegressor

MLPRegressor(hidden_layer_sizes=(100, ),activation='relu',solver='adam',
				alpha=0.0001, batch_size='auto',
              learning_rate='constant',learning_rate_init=0.001,
                power_t=0.5, max_iter=200, shuffle=True,
              random_state=None, tol=0.0001, verbose=False,
				warm_start=False,momentum=0.9,
              nesterovs_momentum=True, early_stopping=False,
				validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
              epsilon=1e-08,	n_iter_no_change=10)


'''
hidden_layer_sizes=(100,12,45 )
activation=
'identity', no-op activation, useful to implement linear bottleneck, returns f(x) = x
'logistic', the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
'tanh', the hyperbolic tan function, returns f(x) = tanh(x).
'relu', the rectified linear unit function, returns f(x) = max(0, x)

solver=
'lbfgs' is an optimizer in the family of quasi-Newton methods.
'sgd' refers to stochastic gradient descent.
'adam' refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba

alpha: L2 penalty (regularization term) parameter.
batch_size='auto': When set to 'auto', batch_size=min(200, n_samples)
learning_rate: 'constant', 'invscaling', 'adaptive'    Only used when solver='sgd'.
max_iter : int, optional, default 200
warm_start : bool, optional, default False
momentum : float, default 0.9   only used when solver='sgd'.
validation_fraction : float, optional, default 0.1
beta_1 : float, optional, default 0.9   when solver='adam'
beta_2 : float, optional, default 0.999   when solver='adam'
'''

############## Classification with MLP#############
########sklearn.neural_network.MLPClassifier
from sklearn.neural_network import MLPClassifier
# MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
#               beta_1=0.9, beta_2=0.999, early_stopping=False,
#               epsilon=1e-08, hidden_layer_sizes=(5, 2),
#               learning_rate='constant', learning_rate_init=0.001,
#               max_iter=200, momentum=0.9, n_iter_no_change=10,
#               nesterovs_momentum=True, power_t=0.5, random_state=1,
#               shuffle=True, solver='lbfgs', tol=0.0001,
#               validation_fraction=0.1, verbose=False, warm_start=False)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=0)
clf.fit(X_train, y_train)

predict =clf.predict(X_test)

print([coef.shape for coef in clf.coefs_])
print(clf.predict_proba(X_test))
print(classification_report(y_test,predict))
print(confusion_matrix(y_test,predict))

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y,predict, pos_label=2)
mlpauc= metrics.auc(fpr, tpr)

fpr, tpr, thresholds = metrics.precision_recall_curve(y,predict, pos_label=2)
mlpaupr= metrics.auc(fpr, tpr)


#Currently, MLPClassifier supports only the Cross-Entropy loss function
#MLPClassifier supports multi-class classification by applying Softmax as the output function
clf = MLPClassifier(solver='adam',hidden_layer_sizes=(15,), alpha=1e-5, random_state=0)
mlp = clf.fit(X_train, y_train)

predictions = mlp.predict(X_test)
print([coef.shape for coef in clf.coefs_])
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print(len(mlp.coefs_))
print(len(mlp.coefs_[0]))
print(len(mlp.intercepts_[0]))


#############Visualization of MLP weights on MNIST##########
'''
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier


# Load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.

# # rescale the data, use the traditional train/test split
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

fig, axes = plt.subplots(4, 4)
# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
               vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()

'''

'''
####Classifier comparison#####
# Code source: Gael Varoquaux
#              Andreas Muller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()



'''
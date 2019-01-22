import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

####### Read Data
import pandas as pd
ENB= pd.read_csv("ENB2012_data.csv")

enb_values=ENB.values[:,0:10]
enb_feature= enb_values[0:768,0:8]
enb_target= enb_values[0:768,9]

##### Linear Regression
X, y = make_regression(n_features=2, random_state=0)
regr = LinearRegression()
regr.fit(X, y)
print(regr.residues_)
print(regr.coef_)
print(regr.intercept_)
a= [[4,5],[89,76]]
np.array(a)
print(regr.predict(a))
print(regr.score(X,y))
print(regr.get_params())

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
# print(fs.f_regression(X2,y))
# print(fs.f_classif(X2,y))
# print(sp.stats.f_oneway( y, np.transpose(X2)))
# print(sm.stats.anova_lm(est2, typ=2))
print(est2.summary())

data = datasets.load_boston()
df = pd.DataFrame(data.data, columns=data.feature_names)
array = df.values
X = array[:, 0:12]
Y = array[:, 12]
# Put the target (housing value -- MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])
X = df["RM"]
y = target["MEDV"]

X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
print(model.summary())

X = df[["RM", "LSTAT"]]
y = target["MEDV"]
model = sm.OLS(y, X).fit()
predictions = model.predict(X)
print(model.summary())


###### ElasticNet
from sklearn.linear_model import ElasticNet

X, y = make_regression(n_features=2, random_state=0)
regr = ElasticNet(random_state=0)
regr.fit(X, y)
print(regr.coef_)
print(regr.intercept_)
a= [[4,5],[89,76]]
np.array(a)
print(regr.predict(a))
print(regr.score(X,y))
print(regr.get_params())

from scipy import stats
# X = diabetes.data
# y = diabetes.target

ElNet = ElasticNet(random_state=0)
ElNet.fit(X, y)
params = np.append(ElNet.intercept_, ElNet.coef_)
predictions = ElNet.predict(X)

newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))

# Note if you don't want to use a DataFrame replace the two lines above with
# newX = np.append(np.ones((len(X),1)), X, axis=1)
# MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params/ sd_b

# from sklearn.feature_selection import chi2
# scores, pvalues = chi2(X, y)
p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

sd_b = np.round(sd_b,3)
ts_b = np.round(ts_b,3)
p_values = np.round(p_values,3)
params = np.round(params,4)

myDF3 = pd.DataFrame()
myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilites"] = [params,sd_b,ts_b,p_values]
print(myDF3)


diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
#
X2 = sm.add_constant(X)
est = OLS(y, X2)
est2 = est.fit_regularized()
print(est2.summary())


###### Ridge
from sklearn.linear_model import Ridge

# X, y = make_regression(n_features=2, random_state=0)
X=enb_feature
y=enb_target
regr = Ridge(alpha=5.0)
regr.fit(X, y)
print(regr.coef_)
print(regr.intercept_)
# a= [[4,5],[89,76]]
# np.array(a)
# print(regr.predict(a))
print(regr.score(X,y))
print(regr.get_params())


from scipy import stats
# X = diabetes.data
# y = diabetes.target

Ridgemodel = Ridge(alpha=5.0)
Ridgemodel.fit(X, y)
params = np.append(Ridgemodel.intercept_, Ridgemodel.coef_)
predictions = Ridgemodel.predict(X)

newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))

# Note if you don't want to use a DataFrame replace the two lines above with
# newX = np.append(np.ones((len(X),1)), X, axis=1)
# MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params/ sd_b

p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

sd_b = np.round(sd_b,3)
ts_b = np.round(ts_b,3)
p_values = np.round(p_values,3)
params = np.round(params,4)

myDF3 = pd.DataFrame()
myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilites"] = [params,sd_b,ts_b,p_values]
print(myDF3)

# diabetes = datasets.load_diabetes()
# X = diabetes.data
# y = diabetes.target
#
# X2 = sm.add_constant(X)
# est = sm.OLS(y, X2)
# est2 = est.fit()
# print(est2.summary())


from sklearn.linear_model import Lasso

# X, y = make_regression(n_features=2, random_state=0)
regr = Lasso(alpha=6)
regr.fit(X, y)
print(regr.coef_)
print(regr.intercept_)
# a= [[4,5],[89,76]]
# np.array(a)
# print(regr.predict(a))
print(regr.score(X,y))
print(regr.get_params())

LassoModel = Lasso(alpha=6)
LassoModel.fit(X, y)
params = np.append(LassoModel.intercept_, LassoModel.coef_)
predictions = LassoModel.predict(X)

newX = pd.DataFrame({"Constant":np.ones(len(X))}).join(pd.DataFrame(X))
MSE = (sum((y-predictions)**2))/(len(newX)-len(newX.columns))

# Note if you don't want to use a DataFrame replace the two lines above with
# newX = np.append(np.ones((len(X),1)), X, axis=1)
# MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
sd_b = np.sqrt(var_b)
ts_b = params/ sd_b

p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]

sd_b = np.round(sd_b,3)
ts_b = np.round(ts_b,3)
p_values = np.round(p_values,3)
params = np.round(params,4)

myDF3 = pd.DataFrame()
myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilites"] = [params,sd_b,ts_b,p_values]
print(myDF3)


# diabetes = datasets.load_diabetes()
# X = diabetes.data
# y = diabetes.target
#
# X2 = sm.add_constant(X)
# est = sm.OLS(y, X2)
# est2 = est.fit()
# print(est2.summary())

alpha=[0.1,0.01,1,2,3,5,8,10]
modelScore=[]
for al in alpha:
	regr=Ridge(alpha=al)
	regr.fit(X,y)
	modelScore.append([al,regr.score(X,y)])

modelScore=pd.DataFrame(modelScore)
plt.plot(modelScore.values[:,0],modelScore.values[:,1])
plt.xlabel("alpha")
plt.ylabel("R^2")
1+1

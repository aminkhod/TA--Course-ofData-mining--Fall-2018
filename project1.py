import pandas as pd

ENB2012= pd.read_csv("ENB2012_data.csv")
enb_value=ENB2012.values[:768,:10]

x_enb=enb_value[:,:8]
y_enb= enb_value[:,8]

from sklearn.linear_model import LinearRegression

enb_LM= LinearRegression()
enb_fit = enb_LM.fit(x_enb,y_enb)
print(enb_fit.coef_)
print(enb_fit.intercept_)
print(enb_fit.score(x_enb,y_enb))
y_hat_enb = enb_fit.predict(x_enb)

import numpy as np
newX = pd.DataFrame({"Constant":np.ones(len(x_enb))}).join(pd.DataFrame(x_enb))
MSE = (sum((y_enb - y_hat_enb)**2))/(len(newX)-len(newX.columns))
var_b = MSE * (np.linalg.inv(np.dot(newX.T,newX)).diagonal())
sd_b = np.sqrt(var_b)
params_LM = np.append(enb_fit.intercept_,enb_fit.coef_)
ts_b = params_LM/ sd_b
from scipy import stats
p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-1))) for i in ts_b]
myDF3 = pd.DataFrame()
myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilites"] = [params_LM,sd_b,ts_b,p_values]
print(myDF3)

1+1
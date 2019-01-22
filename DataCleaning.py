import pandas as pd
import numpy as np

#### Creating data
a,b, c=[12,23,np.nan,46,"",78],['==',"H&M","LCWAKIKI",12,"","Tomy Hilfigor"],[2,'n/a',1,np.nan,2,"na"]
MyData=[]
MyData=pd.DataFrame(MyData)


############### Missing values #############
### First type
MyData["a"],MyData["b"],MyData["c"]=a,b,c


print(MyData)

nullVector = MyData['a'].isnull()
nullIndexes= []
for index in range(len(nullVector)):
	if nullVector[index]==True:
		nullIndexes.append(index)

print(nullIndexes)


#### Second type
MyData.to_csv('mydata.csv',sep=',')
missing_value=['n/a',np.nan, "na","",'==']
editdata=pd.read_csv('mydata.csv', na_values=missing_value)
nullVector = editdata['c'].isnull()
nullIndexes= []
for index in range(len(nullVector)):
	if nullVector[index]==True:
		nullIndexes.append(index)

print(nullIndexes)

#######Third type
cnt=0
for row in editdata['b']:
	try:
		int(row)
		editdata.at[cnt,'b']=np.nan
	except ValueError:
		pass
	cnt+=1
editdata=editdata.drop(['Unnamed: 0'],axis=1)

nullVector = editdata['c'].isnull()
nullIndexes= []
for index in range(len(nullVector)):
	if nullVector[index]==True:
		nullIndexes.append(index)

print(nullIndexes)


### Some commands
print(editdata.isnull().sum())
# NA=MyData.isnull()
# print(NA.values().any())
print(MyData.isnull().sum().sum())


MyData['c'].fillna(12,inplace=True)

bmedian = editdata['c'].median()
editdata['c'].fillna(bmedian,inplace=True)

1+1

import pandas

brecan= pandas.read_csv('breast-cancer.csv')

import numpy as np
fg=56
print("sade")
v=[[1,1,1],[4,4,5]]
v=np.array(v)
v= np.asmatrix(v)

#Reading file and writing some variable
file = open("abalone.data.txt","r")
# file.write("Hello World")
f=file.read()
file.close()

#Reading data
with open("abalone.data.txt", "r", newline='\n') as f:
    data = f.read().split('\n')
    y=data.__len__()
vec=[]

for line in data:
    if (line!=''): #Last line has character of '' that we don't need
        word=line.split(',')
        vec.append(word)
vec=np.asarray(vec)
print(vec[1,2])
675+5

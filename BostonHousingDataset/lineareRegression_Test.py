# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 15:59:17 2021

@author: MiN0DE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.datasets import load_boston
from LineareRegression import extend_matrix
from LineareRegression import LR_fit
from LineareRegression import LR_predict
from LineareRegression import r2_score

# händischer schlichter Test, um zu sehen, ob 1en hinzugefuegt wurden
x = np.array([2.0,5.0,5.0,10.0])
s = extend_matrix(x)
print(type(x))
print(s)


#Aufgabe 2.2 a
#pd.DataFrame([]).to_numpy() --> https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_numpy.html
df_univariat = pd.read_csv("D:/MasterIKT/2. Semester/Special Engineering/Python/2LineareRegression/univariat.csv", sep=',')
x_univariat = df_univariat['x'].to_numpy()
y_univariat = df_univariat['y'].to_numpy()

#Aufgabe 2.2 b
#X_ext = extend_matrix(X) --> befindet sich schon bei meinem LR_fit()
theta_univariat = LR_fit(x_univariat, y_univariat)
print(theta_univariat)
#np.linspace() --> https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
yPredict_univariat = LR_predict(x_univariat,theta_univariat)
r2_univariat = r2_score(x_univariat, y_univariat, theta_univariat)

#Aufgabe 2.2 c
ausgleichsgerade_univariat = LR_predict(np.linspace(0,10,10), theta_univariat)




#Aufgabe 2.3 a
#.iloc[].to_numpy() --> https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html?highlight=pandas%20dataframe%20iloc
df_multivariat = pd.read_csv("D:/MasterIKT/2. Semester/Special Engineering/Python/2LineareRegression/multivariat.csv", sep=',')
x_multivariat = df_multivariat.iloc[:,0:2].to_numpy()
y_multivariat = df_multivariat['y'].to_numpy()


#Aufgabe 2.3 b
theta_multivariat = LR_fit(x_multivariat, y_multivariat)
print(theta_multivariat)
yPredict_multivariat = LR_predict(x_multivariat,theta_multivariat)
r2_multivariat = r2_score(x_multivariat, y_multivariat, theta_multivariat)

#Aufgabe 2.3 c
x1 = 3.5
x2 = 1.2
####GEHT NICHT!
#ausgleichgerade1_multivariat = np.linspace(x1,yPredict_multivariat,theta_multivariat)
#ausgleichsgerade2_multivariat = np.linspace(x2,yPredict_multivariat,theta_multivariat)

#Aufgabe 2.3 d
#t_eval = np.linspace(0,10,10)
#XX1,XX2 = np.meshgrid(t_eval,t_eval)
#X_eval = np.concatenate((XX1.reshape(-1,1),XX2.reshape(-1,1)),axis=1)
#y_eval = LR_predict(X_eval,theta)
#YY = y_eval.reshape(XX1.shape)
#fig = plt.figure()
#ax = Axes3D(fig)
#cset = ax.plot_surface(XX1,XX2,YY,cmap=cm.coolwarm)
#cset = ax.scatter(X[:,0],X[:,1],y, c=’red’)
#ax.clabel(cset, fontsize=9, inline=1)
#plt.show()

#Aufgabe 2.4 a
dataset = load_boston()
print(dataset.DESCR)

#Aufgabe 2.4 b + c
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
plt.hist(df) 

####noch kein Mittelwert berechnet --> 'MEDV'
df['target'] = dataset.target
plt.hist(df['target'])
plt.title("mittlere Hauspreise")
plt.xlabel("Wert")
plt.ylabel("Häufigket")




   
  

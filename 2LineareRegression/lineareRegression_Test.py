# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 15:59:17 2021

@author: MiN0DE
"""

import numpy as np
import pandas as pd
from LineareRegression import extend_matrix
#from LineareRegression import LR_fit
#from LineareRegression import LR_predict
#from LineareRegression import r2_score

# händischer schlichter Test, um zu sehen, ob 1en hinzugefuegt wurden
x = np.array([2.0,5.0,5.0,10.0])
s = extend_matrix(x)
print(type(x))
print(s)


#Aufgabe 2.2 a
df_univariat = pd.read_csv("D:/MasterIKT/2. Semester/Special Engineering/Python/2LineareRegression/univariat.csv")

#Aufgabe 2.2 b
X_ext = extend_matrix(X)
theta = LR_fit(X, y)
y = LR_predict(X,theta)
r2 = r2_score(X, y, theta)






#Aufgabe 2.3 a
df_multivariat = pd.read_csv("D:/MasterIKT/2. Semester/Special Engineering/Python/2LineareRegression/multivariat.csv")

#Aufgabe 2.3 d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
t_eval = np.linspace(0,10,10)
XX1,XX2 = np.meshgrid(t_eval,t_eval)
X_eval = np.concatenate((XX1.reshape(-1,1),XX2.reshape(-1,1)),axis=1)
y_eval = lineare_regression.LR_predict(X_eval,theta)
YY = y_eval.reshape(XX1.shape)
fig = plt.figure()
ax = Axes3D(fig)
cset = ax.plot_surface(XX1,XX2,YY,cmap=cm.coolwarm)
cset = ax.scatter(X[:,0],X[:,1],y, c=’red’)
ax.clabel(cset, fontsize=9, inline=1)
plt.show()
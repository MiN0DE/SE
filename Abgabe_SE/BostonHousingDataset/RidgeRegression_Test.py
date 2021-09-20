# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 12:56:22 2021

@author: Monique Golnik
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.datasets import load_boston
from RidgeRegression import train_test_split as split


#Aufgabe 3.2 a
df_univariat = pd.read_csv("D:/MasterIKT/2. Semester/Special Engineering/Python/2LineareRegression/univariat.csv", sep=',')
df_multivariat = pd.read_csv("D:/MasterIKT/2. Semester/Special Engineering/Python/2LineareRegression/multivariat.csv", sep=',')

#Aufgabe 3.2 b
#Teilen Sie die Daten in Trainings- (60%), Validierungs- (20%) und Testdaten (20%) auf. Verwen-
#den Sie dazu zweimal die bereitgestellte Routine train_test_split und initialisieren Sie den
#Zufallsgenerator mit 0.
x_univariat = df_univariat['x'].to_numpy()
y_univariat = df_univariat['y'].to_numpy()

x_multivariat = df_multivariat.iloc[:,0:2].to_numpy()
y_multivariat = df_multivariat['y'].to_numpy()

X_train, X_test, y_train, y_test = split(x_univariat, y_univariat, 0.6, 0)

X_train, X_test, y_train, y_test = split(x_multivariat, y_multivariat, 0.6, 0)
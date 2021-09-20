# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 17:34:34 2021

@author: Monique Golnik
"""
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_boston
from RidgeRegression import train_test_split
from GradientDecent import StandardScaler_fit
from RidgeRegression import Ridge_fit, Ridge_predict, QuadraticFeatures_fit_transform, mean_squared_error

#Aufgabe 1
#Importieren Sie den Datensatz (siehe 2.  ̈Ubungsblatt) und teilen Sie ihn in Trainings-, Validierungs-
#und Testdaten ein.

dataset = load_boston()
print(dataset.DESCR)
X, y = dataset['data'], dataset['target']
features = dataset.feature_names


# 60% - 20% -20% laut ÜbungsPDF RidgeRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, 0.6, 0)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, 0.5, 0)

#Aufgabe 2
#Erkunden Sie den Trainingsdatensatz. Visualisieren Sie zum Beispiel die verschiedenen Features und
#den Zusammenhang mit dem Preis. Beschreiben Sie Ihre Beobachtungen.
plt.hist(y_train)
plt.title("mittlere Hauspreise")
plt.xlabel("Wert von target")
plt.ylabel("Häufigkeit")
plt.show()

#TODO: das hier geht so nicht
#plt.hist(X_train)
#plt.title("CRIM")
#plt.xlabel("Wert von target")
#plt.ylabel("Häufigkeit")
#plt.show()


#Hilfestellung: We can iterate over all the columns in a lot of cool ways using this technique. Also remember that you can get the indices of all columns easily using:
#Quelle: https://stackoverflow.com/questions/28218698/how-to-iterate-over-columns-of-pandas-dataframe-to-run-regression
#for ind, column in enumerate(df.columns):
#    print(ind, column)

mean_train, std_train, coeffs = [], [], []
for ind, column in enumerate(X_train.T):
    mean, std = StandardScaler_fit(column)
    mean_train.append(mean)
    std_train.append(std)
    print(f'{features[ind]} ({ind}):')
    fig1 = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2)
    ax = plt.subplot(gs[0])
    ax.set_xlabel(features[ind])
    ax.set_ylabel("Häufigkeitsverteilung")
    ax.set_title("")
    plt.hist(column)
    ax2 = plt.subplot(gs[1])
    ax2.set_xlabel(features[ind])
    ax2.set_ylabel("mittlerer Preis")
    plt.scatter(column, y_train, alpha=0.4, )
   
    
    
#Aufgabe 3
#Bestimmen Sie fur polynomiale Features (p= 1 und p= 2) jeweils den optimalen Parameter f ur
#eine regularisierte lineare Regression und den entsprechenden Fehler. W ̈ahlen Sie das beste Modell
#aus und begr ̈unden Sie Ihre Entscheidung.
##--> ich brauche die Quadratische_fit Function 
#quadratisch_fit
X_train_2 = QuadraticFeatures_fit_transform(X_train)
X_val_2 = QuadraticFeatures_fit_transform(X_val)
#ridge-fit 
theta_1 = Ridge_fit(X_train, y_train, 0.1)
theta_2 = Ridge_fit(X_train_2, y_train, 0.1)
#ridge_predict
y_val_1 = Ridge_predict(X_val, theta_1)
y_val_2 = Ridge_predict(X_val_2, theta_2)
#mean-squaredError-func
mse_1 = mean_squared_error(y_val, y_val_1)
mse_2 = mean_squared_error(y_val_2, y_val_1)
#dann modelle vergleichen mithilfe von print 
print('mse für p=1'+str(mse_1))
print('mse für p=2'+str(mse_2))

#Aufgabe 4
#Erstellen Sie ein entsprechendes Modell auf den gesamten Trainings- und Validierungsdaten und
#evaluieren Sie es auf den Testdaten. Geben Sie Kennzahlen an und visualisieren Sie Ihr Ergebnis.
#Diskutieren Sie Ihr Resultat.

#Aufgabe 5
#Untersuchen Sie Ihr Modell mithilfe einer Lernkurve. Diskutieren Sie, wie Sie Ihr Modell verbessern
#k ̈onnten


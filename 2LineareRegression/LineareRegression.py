# -*- coding: utf-8 -*-
"""
Created on Mon May  3 09:45:18 2021

@author: Monique Golnik
"""

# -*- coding: utf-8 -*-

# lineare_regression
#
# Routinen zur Berechnung der multivariaten linearen Regression mit Modell-
# funktion
#
#   h_theta(x) = theta_0 + theta_1 * x_1 + ... + theta_n * x_n
#
# und Kostenfunktion
# 
#   J(theta) = 1/(2m) sum_i=1^m ( h_theta(x^(i)) - y^(i) )^2
#
# Der Vektor theta wird als
#
#   (theta_0, theta_1, ... , theta_n)
#
# gespeichert. Die Feature-Matrix mit m Daten und n Features als
#
#       [ - x^(1) - ]
#   X = [    .      ]    (m Zeilen und n Spalten)
#       [ - x^(m) - ]
#

import numpy as np

#%% extend_matrix

# Erweitert eine Matrix um eine erste Spalte mit Einsen
#
# X_ext = extend_matrix(X)
#
# Eingabe:
#   X      Matrix m x n (numpy.ndarray)
#
# Ausgabe
#   X_ext  Matrix m x (n+1) der Form [1 X] (numpy.ndarray)
#
def extend_matrix(X):
    # TODO: setze X_ext
    N = X.shape
    X_ext = np.c_[np.ones(N), X]
    return X_ext


    
#%% LR_fit

# Berechnung der optimalen Parameter der multivariaten linearen Regression 
# mithilfe der Normalengleichung.
#
# theta = LR_fit(X, y)
#
# Eingabe:
#   X      Matrix m x n mit m Datenpunkten und n Features (numpy.ndarray)
#   y      Vektor der Länge m der Zielwerte (numpy.ndarray)
#
# Ausgabe
#   theta  Vektor der  Länge n+1 der optimalen Parameter (numpy.ndarray)
#
# Hinweis: Benutzen Sie extend_matrix und np.linalg.solve zur Lösung des 
#   linearen Gleichungssystems
#
def LR_fit(X, y):
    # TODO: berechne theta --> entspricht dem Block "maschinelles Lernen" in der PDF S. 18
    
    #Hinweis nutzen s.o.
    X_ext = extend_matrix(X)
    #Formel PDF S. 35 + Hinweis 
    #np.linalg.solve() --> https://stackabuse.com/solving-systems-of-linear-equations-with-pythons-numpy/
    #.dot = https://www.javatpoint.com/numpy-dot
    theta = np.linalg.solve(X_ext.T.dot(X_ext),X_ext.T.dot(y))
    return theta

    
#%% LR_predict

# Berechnung der Vorhersage der der multivariaten linearen Regression.
#
# y = LR_predict(X,theta)
#
# Eingabe:
#   X      Matrix m x n mit m Datenpunkten und n Features (numpy.ndarray)
#   theta  Vektor der  Länge n+1 der Parameter (numpy.ndarray)
#
# Ausgabe
#   y      Vektor der Länge m der Vorhersagewerte (numpy.ndarray)
#
# Hinweis: Benutzen Sie extend_matrix.
#
def LR_predict(X, theta):
    # TODO: berechne y --> entspricht dem Block "Vorhersage" in der PDF S. 18
    
    #Hinweis nutzen s.o.
    X_ext = extend_matrix(X)
    y = X_ext.dot(theta)
    return y
    

#%% r2_score

# Berechnung des Bestimmtheitsmaßes R2
#
# y = r2_score(X, y, theta)
#
# Eingabe:
#   X      Matrix m x n mit m Datenpunkten und n Features (numpy.ndarray)
#   y      Vektor der Länge m der Zielwerte (numpy.ndarray)
#   theta  Vektor der  Länge n+1 der Parameter (numpy.ndarray)
#
# Ausgabe
#   r2     Bestimmtheitsmaß R2 (Skalar)
#
# Hinweis: Benutzen Sie LR_predict
#
def r2_score(X, y, theta):
    # TODO: berechne r2 --> dient zur Berechnung der Güte (wenn 1 dann super gut, wenn 0 dann schlecht)
    prediction = LR_predict(X,theta)
    
    #Formel aus Aufgaben-PDF S.21
    #auseinanderklamüsern, lässt sich leichter  Fehler finden und ist lesbarer
    oben = np.sum((y-prediction)**2)
    unten = np.sum((y - np.mean(y))**2)
 
    #sicher gehen, dass keine Variable 0 ist und die Division verhindert
#zu sehr an Java gewöhnt...
    #if oben != 0 and unten != 0 
    #r2 = 1- (oben/unten) 
    #else 
    #r2 = 0
#https://www.python-kurs.eu/bedingte_anweisungen.php    
    r2 = 1 - (oben/unten) if oben and unten !=0 else 0
    return r2

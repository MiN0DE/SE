# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 12:50:52 2021

@author: Monique Golnik
"""

# -*- coding: utf-8 -*-

# ridge_fit
#
# Routinen zur Berechnung der regularisierten multivariaten linearen Regression
# mit polynomialen Features und Modellfunktion
#
#   h_theta(x) = theta_0 + theta_1 * x_1 + ... + theta_n * x_n
#
# und Kostenfunktion
# 
#   J(theta) = 1/(2m) sum_(i=1)^m ( h_theta(x^(i)) - y^(i) )^2 
#              + alpha/(2m) sum_(k=1)^n theta_k^2 
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


#%% extend_matrix (vom letzten Mal verwenden, wird nicht geprüft)

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
    N = np.size(X,0)
    X_ext = np.c_[np.ones((N,1)), X]
    return X_ext


#%% train_test_split (wird bereit gestellt)

# Teilt den Datensatz in Training (Anteil frac) und Test (Rest)
#
# [Xtrain, Xtest, ytrain, ytest] = train_test_split(X)
#
# Eingabe:
#   X      Matrix m x n (numpy.ndarray)
#   y      Vektor der Länge m der Zielwerte (numpy.ndarray)
#   frac   Anteil im Trainingsset 0 <= frac <= 1 
#
# Ausgabe
#   Xtrain Featurematrix Trainingsset 
#          mtrain x n mit mtrain = frac * m (numpy.ndarray)
#   Xtest  Featurematrix Testset 
#          mtest = m - mtrain (numpy.ndarray)
#   ytrain Vektor Zielwerte Trainingsset Länge mtrain (numpy.ndarray)
#   ytest  Vektor Zielwerte Testset Länge mtest (numpy.ndarray)
#
def train_test_split(X, y, frac, seed):
    m = X.shape[0]
    np.random.seed(seed)
    index = np.arange(m)
    np.random.shuffle(index)
    cut = int(m*frac)
    return X[index[:cut],:], X[index[cut:],:], y[index[:cut]], y[index[cut:]]


#%% QuadraticFeatures_fit_transform

# Fügt der Featurematrix quadratische und gemischte Features hinzu
#
# Xq = QuadraticFeatures_fit_transform(X)
#
#        [ |        |    |        |            |       |       |          |   ]
#   Xq = [x_1, ... x_n, x_1^2, x_1*x_2, ... x_1*x_n, x_2^2, x_2*x_3, ... x_n^2] 
#        [ |        |    |        |            |       |       |          |   ]
#
# Eingabe:
#   X      Featurematrix m x n (numpy.ndarray)
#
# Ausgabe
#   Xq     Featurematrix m x (n+n*(n+1)/2) (numpy.ndarray)
#
def QuadraticFeatures_fit_transform(X):
    # TODO: berechne Xq
    
    
    return Xq


#%% mean_squared_error

# Berechnung des mittleren Fehlerquadrats
#
# mse = mean_squared_error(y_true, y_pred)
#
# Eingabe:
#   y_true  Vektor der Länge m der wahren Zielwerte (numpy.ndarray)
#   y_pred  Vektor der Länge m der vorhergesagten Zielwerte (numpy.ndarray)
#
# Ausgabe
#   mse    Mittleres Fehlerquadrat mse = 1/m sum_(i=1)^m (y_true_i-y_pred_i)^2
#
def mean_squared_error(y_true, y_pred):
    # TODO: berechne mse
    mse = np.mean((y_true - y_pred) **2)
    return mse

    
#%% Ridge_fit

# Berechnung der optimalen Parameter der multivariaten regularisierten linearen
# Regression mithilfe der Normalengleichung.
#
# theta = Ridge_fit(X, y, alpha)
#
# Eingabe:
#   X      Matrix m x n mit m Datenpunkten und n Features (numpy.ndarray)
#   y      Vektor der Länge m der Zielwerte (numpy.ndarray)
#   alpha  Regularisierungsparameter (Skalar)
#
# Ausgabe
#   theta  Vektor der  Länge n+1 der optimalen Parameter (numpy.ndarray)
#
# Hinweis: Benutzen Sie extend_matrix und np.linalg.solve zur Lösung des 
#   linearen Gleichungssystems
#
def Ridge_fit(X, y, alpha):
    # TODO: berechne theta --> Ansätze s. PDF Lineare Regression2 S. 31
    X_ext=extend_matrix(X)
    X_T = X_ext.T
    ident=(np.identity(np.size(X_T @ X,1)))
    ident[0,0] = 0
    summ = X_T @ X + ident*alpha
    temp = np.linalg.inv(summ)
    temp2 = X_T.dot(y)
    
    theta = np.dot(temp, temp2)
    return theta

    
#%% Ridge_predict

# Berechnung der Vorhersage der der multivariaten regularisierten linearen 
# Regression.
#
# y = Ridge_predict(X,theta)
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
def Ridge_predict(X, theta):
    # TODO: berechne y
    #Hinweis nutzen s.o.
    X_ext = extend_matrix(X)
    y = X_ext.dot(theta)
    return y
   
    

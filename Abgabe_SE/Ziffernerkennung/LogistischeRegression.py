# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:04:44 2021

@author: Monique Golnik
"""

# -*- coding: utf-8 -*-

# logistische_regression
#
# Routinen zur Berechnung der multivariaten logistischen Regression
# mit Modellfunktion
#
#   h_theta(x) = sigma(theta_0 + theta_1 * x_1 + ... + theta_n * x_n)
#
# mit
#
#   sigma(t) = 1/(1+exp(-t))
#
# und Kostenfunktion
# 
#   J(theta) = -1/m sum_(i=1)^m (y^(i) log(h_theta(x^(i))) 
#                               + (1-y^(i)) log(1 - h_theta(x^(i))) 
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



#####Einführung: https://www.inwt-statistics.de/blog-artikel-lesen/Logistische_Regression.html
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



#%% logistic_cost_function

# Berechnung der Kostenfunktion der logistischen Regression und deren 
# Gradienten
#
# J, Jgrad = logistic_cost_function(X,y, theta)
#
# Eingabe:
#   X      Matrix m x n mit m Datenpunkten und n Features (numpy.ndarray)
#   y      Vektor der Länge m der Zielwerte (numpy.ndarray)
#   theta  Vektor der  Länge n+1 der Parameter (numpy.ndarray)
#
# Ausgabe
#   J      Wert der Kostenfunktion (Skalar)
#   Jgrad  Gradient der Kostenfunktion (numpy.ndarray)
#

#Auslagerung für mehr Übersichtlichkeit in der Fehlerfindung
#Idee von PDF5 S. 8
#sg. sigmoide Funktion
def sigma(t):
    return 1 / (1+np.exp(-t))
## für t = 0 sollte 1/2 herauskommen


def logistic_cost_function(X,y, theta):
    # TODO: berechne J und Jgrad
    #Ansatz PDF3 ab Seite 15 und PDF5 Seite 8 + 13
    
    #s. wie  h(theta)= sigma(xT*theta)
    h = sigma(extend_matrix(X).dot(theta))
    print(h)
    #PDF Seite 17
    #J = -1/len(y)*(y @ np.log(h) + (1 - y) @ np.log(1 - h)) -->dividiert durch 0 und hängt sich auf!:(
    J=(1/len(y))*((-1)*np.transpose(y)@np.log(h) -np.transpose(1-y)@np.log(1-h)  )
    #Ansatz aus GradientDecent Übung: gradientvector = 1/len(y) * extend_matrix(X).T.dot(diff)
    #PDF S.22
    Jgrad = 1/len(y) * extend_matrix(X).T.dot(h - y)
    return J, Jgrad    


#%% LogisticRegression_fit

# Berechnung der optimalen Parameter der multivariaten logistischen Regression 
# mithilfe des Gradientenabstiegsverfahrens
#
# theta, J = LogisticRegression_fit(X,y,eta,tol)
#
# Die Iteration soll abgebrochen werden, falls 
#
#   || grad J || < tol
#
# gilt, wobei ||.|| die (euklidsche) Länge eines Vektors ist. Die Iteration
# soll abbrechen (mittels raise), falls die Kostenfunktion nicht fällt. Als
# Startvektor soll der Nullvektor gewählt werden.
#
# Eingabe:
#   X      Matrix m x n mit m Datenpunkten und n Features (numpy.ndarray)
#   y      Vektor der Länge m der Zielwerte (numpy.ndarray)
#   eta    Learning rate (Skalar)
#   tol    Toleranz der Abbruchbedingung
#
# Ausgabe
#   theta   Aktueller Vektor der Länge n+1 der optimalen Parameter (numpy.ndarray)
#   J       Aktueller Wert der Kostenfunktion (float)
#
def LogisticRegression_fit(X,y,eta,tol):
    # TODO: berechne theta und J
    X_ext = extend_matrix(X)
    #Vorbedingungen schaffen für Iteration
    iteration_theta = []
    iteration_J = []
    #und da kein Theta sonst vorhanden, ich aber J will und bei Vektor 0 beginnen soll
    theta = np.zeros(X_ext.shape[1])
    J, Jgrad = logistic_cost_function(X,y,theta)
    
    while(np.linalg.norm(Jgrad) >=tol):
        theta = theta - eta * Jgrad
        J, Jgrad = logistic_cost_function(X, y, theta)
        iteration_theta.append(theta)
        iteration_J.append(J)
        
        if len(iteration_theta) > 2:
            if np.all(iteration_J[-2:] <= J):
                raise 
               
    return theta, J


#%% LogisticRegression_predict

# Berechnung der Vorhersage der multivariaten logistischen Regression
#
# y = LogisticRegression_predict(X,theta)
#
# Eingabe:
#   X      Matrix m x n mit m Datenpunkten und n Features (numpy.ndarray)
#   theta  Vektor der  Länge n+1 der Parameter (numpy.ndarray)
#
# Ausgabe
#   y      Vektor der Länge m der Vorhersagewerte (numpy.ndarray)
#
#
def LogisticRegression_predict(X, theta):
    # TODO: berechne y
    X_ext = extend_matrix(X)
      #s. h(theta)= sigma(xT*theta)
    h = sigma(X_ext.dot(theta))
    #PDF S.10/17, größer als 0.5 dann quasi aufrunden auf 1
    y = (h >= 0.5).astype(int)
    return y
    

#%% accuracy_score

# Berechnung der Genauigkeit
#
# acc = accuracy_score(y_true,y_pred)
#
# Eingabe:
#   y_true Vektor der Länge m der wahren Werte (numpy.ndarray)
#   y_pred Vektor der Länge m der Vorhersagewerte (numpy.ndarray)
#
# Ausgabe
#   acc    Genauigkeit (Skalar)
#
def accuracy_score(y_true,y_pred):
    # TODO: berechne acc
    #PDF5 Seite 25
    acc = y_true/y_pred
    return acc


#%% precision_score

# Berechnung der Präzision bzgl. der Klasse 1
#
# prec = precision_score(y_true,y_pred)
#
# Eingabe:
#   y_true Vektor der Länge m der wahren Werte (numpy.ndarray)
#   y_pred Vektor der Länge m der Vorhersagewerte (numpy.ndarray)
#
# Ausgabe
#   prec    Genauigkeit (Skalar)
#
def precision_score(y_true,y_pred):
    # TODO: berechne prec
    #sieh Konfusionsmatrix PDF5 s. 29
    #prec=tp/tp+fp
    tp = np.sum((y_true == y_pred) & (y_true == 1))
    fp = np.sum((y_true != y_pred) & (y_true == 0))

    prec = tp / (tp + fp)
    return prec

#%% recall_score

# Berechnung des Recalls bzgl. der Klasse 1
#
# recall = recall_score(y_true,y_pred)
#
# Eingabe:
#   y_true Vektor der Länge m der wahren Werte (numpy.ndarray)
#   y_pred Vektor der Länge m der Vorhersagewerte (numpy.ndarray)
#
# Ausgabe
#   recall Recall (Skalar)
#
def recall_score(y_true,y_pred):
    # TODO: berechne recall
    #sieh Konfusionsmatrix PDF5 s. 29
    #prec=tp/tp+fn
    tp = np.sum((y_true == y_pred) & (y_true == 1))
    fn = np.sum((y_true != y_pred) & (y_true == 0))
    
    recall = tp/(tp+fn)
    return recall# -*- coding: utf-8 -*-


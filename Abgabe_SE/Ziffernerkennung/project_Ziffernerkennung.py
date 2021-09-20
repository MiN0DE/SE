# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 19:19:26 2021

@author: Monique Golnik
"""

import numpy as np
import matplotlib.pyplot as plt
from LogistischeRegression_Test import train_test_split  as split
#from logistischeregression import LogisticRegression_fit as lrf
#from logistischeregression import LogisticRegression_predict as lrp
import LogistischeRegression as lr
#Aufgabe 1 Importieren Sie den Datensatz fur handgeschriebene Ziffern mithilfe folgender Anweisungen.
from sklearn.datasets import load_digits
digits = load_digits()

data_x = digits.data
data_y = digits.target
target = digits.target_names


#Aufgabe 1.2 Erstellen Sie einen geeigneten Zielvektor. Teilen Sie die Daten in Traings- (80%) und Testdaten
#(20%) auf
X_train, X_test, y_train, y_test = split(data_x, data_y, 0.8, 0)
#Überprüfung
print(np.shape(X_train),np.shape(y_train),np.shape(X_test),np.shape(y_test))

#Aufgabe 2 Visualisieren Sie verschiedene Ziffern mittels plt.matshow und beschreiben Sie Ihre Beobachtungen.
#Hinweise: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.matshow.html
plt.matshow(digits.images[20])
plt.colorbar()
plt.show()

plt.matshow(digits.images[1000])
plt.colorbar()
plt.show()

plt.matshow(digits.images[658])
plt.colorbar()
plt.show()


#Aufgabe 3 Erstellen Sie auf den Trainingsdaten eine logistische Regression
## TODO: die cost_function hängt sich auf - FEHLERSUCHE BETREIBEN!! 
theta, J = lr.LogisticRegression_fit(X_train, y_train, 0.01, 3e-4)
#Ausgeben lassen
print(theta, J)
#Vorhersage treffen
#y_pred = lr.LogisticRegression_predict(X_test, theta)



#Aufgabe 4  Evaluieren Sie Ihr Modell auf den Trainings- und Testdaten. Geben sie die Genauigkeit aus, erstellen
#Sie eine Konfusionsmatrix und geben Sie geeignete Kennzahlen an. Visualisieren Sie falsch klassifizierte Ziffern des Testdatensatzes. 
#Diskutieren Sie Ihre Ergebnisse.
#Aus Übungsblatt: Genauigkeit, Präzision, Recall
y_pred = 2
print(y_pred)
print(y_test)

ac = lr.accuracy_score(y_test, y_pred)
prec = lr.precision_score(y_test, y_pred)
rec = lr.recall_score(y_test,y_pred)
#Genauigkeit
print(ac)
#
f1= 2*(prec *rec)/prec + rec

#Hilfestellung zur Visualisierung meiner Konfusionsmatrix
def confusion_matrix(y_true, y_pred):
    tn = np.sum((y_true == y_pred) & (y_true == 0))
    tp = np.sum((y_true == y_pred) & (y_true == 1))
    fp = np.sum((y_true != y_pred) & (y_true == 0))
    fn = np.sum((y_true != y_pred) & (y_true == 1))
    
    return np.array([[tp, fp],[fn, tn]])


k_matrix=confusion_matrix(y_test, y_test)
k_matrix = k_matrix/np.max(np.abs(k_matrix), axis=1)
plt.matshow(k_matrix)

#besser hohe Präzision oder hoher Recall? Frage in Abgabe PDF beantworten


#Aufgabe5  Untersuchen Sie Ihr Modell mithilfe einer Lernkurve. Diskutieren Sie, wie Sie Ihr Modell verbessern
#konnten. Wurde eine Regularisierung helfen?

#Bonus Aufgabe:  Erweitern Sie die Erkennung mittels One-vs-Rest-Methode auf alle Ziffern
#und werten Sie Ihr Modell geeignet aus
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
data_y_9 = np.copy(data_y)

for i in range(len(data_y)):
    if data_y[i] == 9:
        data_y_9[i] =1
    else:
        data_y_9[i] =0

#Aufgabe 1.2 Erstellen Sie einen geeigneten Zielvektor. Teilen Sie die Daten in Traings- (80%) und Testdaten
#(20%) auf
X_train, X_test, y_train, y_test = split(data_x, data_y_9, 0.8, 0)
#Überprüfung
print("Aufgabe 1.2",np.shape(X_train),np.shape(y_train),np.shape(X_test),np.shape(y_test), np.shape(X_val), np.shape(y_val))

#Aufgabe 2 Visualisieren Sie verschiedene Ziffern mittels plt.matshow und beschreiben Sie Ihre Beobachtungen.
#Hinweise: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.matshow.html
plt.matshow(digits.images[0])
plt.colorbar()
plt.show()

plt.matshow(digits.images[8])
plt.colorbar()
plt.show()

plt.matshow(digits.images[5])
plt.colorbar()
plt.show()


#Aufgabe 3 Erstellen Sie auf den Trainingsdaten eine logistische Regression
theta, J = lr.LogisticRegression_fit(X_train, y_train, 0.001,0.1)
#Ausgeben lassen
print(theta, J)

#Vorhersage treffen
y_pred = lr.LogisticRegression_predict(X_test,theta)



#Aufgabe 4  Evaluieren Sie Ihr Modell auf den Trainings- und Testdaten. Geben sie die Genauigkeit aus, erstellen
#Sie eine Konfusionsmatrix und geben Sie geeignete Kennzahlen an. Visualisieren Sie falsch klassifizierte Ziffern des Testdatensatzes. 
#Diskutieren Sie Ihre Ergebnisse.
#Aus Übungsblatt: Genauigkeit, Präzision, Recall
#y_pred=2
#print("y_pred",y_pred)
#print("y_test", y_test)
#Kennzahlen Genauigkeit, Präzision, Recall, F1
ac = lr.accuracy_score(y_test, y_pred)
print("Kennzahl Genauigkeit \n",ac)

prec = lr.precision_score(y_test, y_pred)
print("Kennzahl Präzision \n", prec)

rec = lr.recall_score(y_test,y_pred)
print("Kennzahl Recall \n",rec)

f1= 2*(prec *rec)/(prec + rec)
print("Kennzahl F1 \n",f1)

#besser hohe Präzision oder hoher Recall? Frage in Abgabe PDF beantworten

#Hilfestellung zur Visualisierung meiner Konfusionsmatrix
def confusion_matrix(y_true, y_pred):
    tn = np.sum((y_true == y_pred) & (y_true == 0))
    tp = np.sum((y_true == y_pred) & (y_true == 1))
    fp = np.sum((y_true != y_pred) & (y_true == 0))
    fn = np.sum((y_true != y_pred) & (y_true == 1))
    
    return np.array([[tp, fp],[fn, tn]])


k_matrix=confusion_matrix(y_pred, y_test)
print("Konfusionsmatrix \n", k_matrix)





#Aufgabe5  Untersuchen Sie Ihr Modell mithilfe einer Lernkurve. Diskutieren Sie, wie Sie Ihr Modell verbessern
#konnten. Wurde eine Regularisierung helfen?

#Bonus Aufgabe:  Erweitern Sie die Erkennung mittels One-vs-Rest-Methode auf alle Ziffern
#und werten Sie Ihr Modell geeignet aus

for i in [0,1,2,3,4,5,6,7,8,9]:
    plt.matshow(digits.images[i])
    print("Ziffernindex", i)
    plt.show()
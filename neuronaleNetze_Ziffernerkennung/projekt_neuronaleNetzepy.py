# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 18:21:19 2021

@author: Monique Golnik
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid as parGrid
from sklearn.model_selection import learning_curve 
# keras für Neuronale Netze
import tensorflow 
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from keras.layers import Dense, LSTM, Dropout 
from keras.utils import plot_model 
# Für das Darstellen von Bildern im SVG-Format
import graphviz as gv
import pydot
from keras.utils import model_to_dot
from IPython.display import SVG
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger



#Aufgabe 1. Importieren Sie den Datensatz fur handgeschriebene Ziffern mithilfe folgender Anweisungen. ¨
from sklearn.datasets import load_digits
digits = load_digits()

#Aufgabe 1.2 und erstellen Sie einen geeigneten Zielvektor. Teilen Sie ihn in Trainings- (60%) Validierungs- (20%)
#und Testdaten (20%) ein.

def train_test_split(X, y, frac, seed):
    m = X.shape[0]
    np.random.seed(seed)
    index = np.arange(m)
    np.random.shuffle(index)
    cut = int(m*frac)
    return X[index[:cut],:], X[index[cut:],:], y[index[:cut]], y[index[cut:]]

X, y = digits['data'], digits['target']
features = digits.feature_names
targets = digits.target_names
images = digits.images

X_train, X_test, y_train, y_test = train_test_split(X, y, 0.6, 0)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, 0.5, 0)

print("Aufgabe 1.2 Trainingsdaten",np.shape(X_train),np.shape(y_train))
print("Aufgabe 1.2 Testdaten",np.shape(X_test),np.shape(y_test))
print("Aufgabe 1.2 Validierungsdaten", np.shape(X_val), np.shape(y_val))

#Aufgabe 2 Erstellen Sie ein geeignetes neuronales Netz und werten Sie die Genauigkeit auf den Validierungsdaten aus. 
#Verwenden Sie die Early Stopping Strategie bei der Berechnung.

#Quelle: elab2go.de/demp-py5 
#Model erstellen
#Konfigurationsparameter
TIMESTEPS = 7 # Länge eines gleitenden Zeitfensters
UNITS = 10 # Anzahl von Ausgängen bei einer LSTM-Schicht
N_LAYER = 4 # Anzahl LSTM-Schichten
model = Sequential(name='sequential')  # Erstelle ein sequentielles Modell
# Füge Schichten hinzu
for i in range(N_LAYER):
    lstm_layer = LSTM(units = UNITS, input_shape=(TIMESTEPS,1),
                      return_sequences=True, 
                      name = 'lstm_' + str(i+1))
    model.add(lstm_layer)
model.add(LSTM(units = UNITS, input_shape=(TIMESTEPS,1), name = 'lstm_' + str(N_LAYER+1)))
model.add(Dense(units = 1, name='dense_1'))
# Konfiguriere das Modell für die Trainingsphase
model.compile(optimizer = "adam", loss = "mse", metrics=['mean_squared_error'])
# Zusammenfassung und Visualisierung des Modells
model.summary()
plot_model(model, show_shapes=True, show_layer_names=True)



#Modell trainieren
cb_stop = EarlyStopping(monitor='val_loss', mode='min', 
                        verbose=1, patience=200)
log_file = 'demo-py5-log.csv'
cb_logger = CSVLogger(log_file, append=False, separator=';')

# X_Train erhält eine zusätzliche Dimension
X_train = np.reshape(X_train, 
                     (X_train.shape[0], X_train.shape[1], 1))
# Trainiere das Modell mit Hilfe der Funktion fit()
BATCH_SIZE = 64
#Wie oft will ich mein Neuron trainieren
EPOCHS = 100
#20% Validierungsdaten
VALIDATION_SPLIT = 0.2
history = model.fit(X_train, y_train, 
                    epochs=EPOCHS, batch_size=BATCH_SIZE, 
                    validation_split=VALIDATION_SPLIT, verbose=2, 
                    callbacks=[cb_logger, cb_stop])
# Speichere das Modell im Format HDF5
model.save("model_adam.h5")
print("History");print(history.history.keys());

plt.plot(history.history['mean_squared_error'], label='MSE (Trainingsdaten)')
plt.plot(history.history['val_mean_squared_error'], label='MSE (Testdaten)')

plt.title('Training: Entwicklung des Fehlers')
plt.ylabel('MSE-Fehler')
plt.xlabel('Epochen')
plt.legend()

#Aufgabe 3. Optimieren Sie Ihr Neuronales Netz, z.B. hinsichtlich Standardisierung, Anzahl der Neuronen pro
#Schicht, Anzahl der Schichten, Batch Normalization und Dropout anhand der Genauigkeit auf den
#Validierungsdaten. Sie konnen z.B. die Klasse ¨ sklearn.model_selection.ParameterGrid
#verwenden.

#Aufgabe 4 . Geben Sie fur das optimale Modell die Genauigkeit, die Konfusionsmatrix und Kennzahlen f ¨ ur jede ¨
#Klasse an. Diskutieren Sie Ihre Ergebnisse.

#Aufgabe 5 Untersuchen Sie Ihr Modell mithilfe einer Lernkurve. Diskutieren Sie, wie Sie Ihr Modell verbessern
#konnten.

#Aufgabe 6 Werten Sie das optimale Modell auf den Testdaten aus (Genauigkeit, Konfusionsmatrix, Kennzahlen
#fur jede Ziffer, etc.). Visualisieren Sie falsch klassifizierte Ziffern. Diskutieren Sie Ihr Modell


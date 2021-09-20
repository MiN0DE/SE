# -*- coding: utf-8 -*-
#
# Summe aller positiven Einträge eines Vektors
#
#   s = summe_pos(x)
#
#   Eingabe
#       x       Vektor (numpy.ndarray mit einer Dimension)
# 
#   Ausgabe
#       s       Summe aller positiven Einträge von x (Skalar)
# 
import numpy as np

def summe_pos(x):
    # TODO: berechne s
    x[x<0] = 0
    s = sum(x)
    return s

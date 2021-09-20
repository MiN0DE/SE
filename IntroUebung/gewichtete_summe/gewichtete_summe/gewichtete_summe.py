# -*- coding: utf-8 -*-

# gewichtete_summe berechnet
#
#   s = sum_(i=1)^n i * x_i
# 
# Eingabe:
#   x   Vektor (numpy.ndarray)
# Ausgabe:
#   s   Skalar
#
def gewichtete_summe(x):
    n = len(x);
    s = 0.0;
    for i in range(n):
        s += (i+1)*x[i];
    return s

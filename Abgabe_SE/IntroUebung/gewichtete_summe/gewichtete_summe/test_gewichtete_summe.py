# -*- coding: utf-8 -*-
import numpy as np 
from gewichtete_summe import gewichtete_summe
tol = 1e-12;

# händisch berechnet
x = np.array([-3.0,4.0,1.0])
s_ref = 8.0
s = gewichtete_summe(x)
assert( abs(s-s_ref) < tol)

# analytische tests
nn = [4, 10, 53]
for n in nn:
    x = np.ones(n);
    s_ref = n*(n+1)/2;
    s = gewichtete_summe(x)
    assert(abs(s-s_ref) < tol)

# extreme eingaben
x = np.array([])
s_ref = 0.0
s = gewichtete_summe(x)
assert( abs(s-s_ref) < tol)

# zufällige eingaben
n = 100;
x = np.random.randn(n);
xtilde = x / np.arange(1,n+1)
s_ref = np.sum(x);
s = gewichtete_summe(xtilde)
assert( abs(s-s_ref) < tol)

print('Alle Tests bestanden.')

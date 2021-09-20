# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:58:20 2021

@author: MiN0DE
"""

#tests für summe_pos()

import numpy as np
from summe_pos import summe_pos


# händisch berechnet
x = np.array([2.0,5.0,5.0,10.0])
s = summe_pos(x)
print(type(x))
print(s)
assert s == 22
print('1 Test bestanden')

x = np.array([2.0,5.0,-5.0,10.0])
s = summe_pos(x)
print(type(x))
print(s)
assert s == 17
print('2 Tests bestanden')
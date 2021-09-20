# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 19:42:49 2021

@author: Monique Golnik
"""

import numpy as np
#aus RidgeRegression übernommen
def train_test_split(X, y, frac, seed):
    m = X.shape[0]
    np.random.seed(seed)
    index = np.arange(m)
    np.random.shuffle(index)
    cut = int(m*frac)
    return X[index[:cut],:], X[index[cut:],:], y[index[:cut]], y[index[cut:]]
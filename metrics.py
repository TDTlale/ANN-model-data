# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:11:28 2024

@author: Dell
"""
import numpy as np
from sklearn.metrics import r2_score

#absolute percentage error
def ape(measurement,prediction):
    
    return (abs(measurement - prediction)*100)/measurement

#mean squared logarithmic error
def msle(measurement,prediction):

    return sum((np.log(measurement + 1) - np.log(prediction + 1))**2)/len(measurement)

#mean squared error
def mse(measurement,prediction):
    
    return sum((measurement - prediction)**2)/len(prediction)

#coefficient of determination
def R2_score(measurement, prediction):
    
    return r2_score(measurement, prediction, sample_weight = None, multioutput='raw_values', force_finite=True)
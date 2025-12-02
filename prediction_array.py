# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 12:50:32 2025

@author: tlale
"""

import numpy as np
import pandas as pd
import os
import itertools

def generate_factorial_array(params):
    """
    Generates a full factorial array of all parameter combinations.
    
    Parameters:
    - params: list of dicts, each with 'min', 'max', and 'levels' keys
    
    Returns:
    - numpy.ndarray with all combinations
    """
    levels_list = []
    
    for p in params:
        if p['levels'] < 2:
            raise ValueError("Each parameter must have at least 2 levels.")
        levels = np.linspace(p['min'], p['max'], p['levels'])
        levels_list.append(levels)
    
    # Generate all combinations (Cartesian product)
    factorial_combinations = list(itertools.product(*levels_list))
    
    return np.array(factorial_combinations)


params = [
    {'min': 1000, 'max': 1500, 'levels': 21},
    {'min': 500, 'max': 1000, 'levels': 21}, 
    {'min': 1, 'max': 7, 'levels': 21}
]

array = generate_factorial_array(params)
#print(array)


prediction_array = pd.DataFrame(array)

prediction_array.describe()

with pd.ExcelWriter(r"{}\prediction_array.xlsx".format(os.path.dirname(os.getcwd()))) as writer:

        
        prediction_array.to_excel(writer, sheet_name = 'prediction_array', index=False)
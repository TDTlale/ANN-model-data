# -*- coding: utf-8 -*-
"""
Created on Thu May  8 16:46:07 2025

@author: tlale
"""

def label_loc(cs, rx, ry,prop_dict, prop):
    fmt = {}
    manual_labels = []
    indx = []
    count = 0
    for level, path in zip(cs.levels, cs.get_paths()):
        
        
        if level !=0:
            
            bbox = path.get_extents() 
            y_max = bbox.max[1]
            x_max = bbox.max[0]
            
            if y_max != float("inf"):
                count = count + 1
                manual_labels.append((rx*x_max, ry*y_max))
                fmt[level] = f'{prop_dict[prop]} = {round(level,2)}'
                indx.append(count)
    return fmt, manual_labels, indx
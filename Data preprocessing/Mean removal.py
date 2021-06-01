# -*- coding: utf-8 -*-
"""
Created on Mon May 31 20:37:20 2021

@author: JuanCHerreraPro
"""

#Mean removal

"""  It involves removing the mean from each feature so that it is centered on 
    zero. Mean removal helps in removing any bias from the features 
    
 """

import numpy as np
from sklearn import preprocessing

input_data = np.array([[5.1, -2.9, 3.3],
 [-1.2, 7.8, -6.1],
 [3.9, 0.4, 2.1],
 [7.3, -9.9, -4.5]])

# Print mean and standard deviation
print("BEFORE remove mean:")
print("Mean =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

# Remove mean
data_scaled = preprocessing.scale(input_data)
print("\nAFTER:")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))
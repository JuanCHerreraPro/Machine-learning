# -*- coding: utf-8 -*-
"""
Created on Mon May 31 20:48:29 2021

@author: JuanCHerreraPro
"""

#Scaling

"""
    Since the range of values of raw data varies widely, in some machine learning
    algorithms, objective functions will not work properly without normalization. 
    
    For example, many classifiers calculate the distance between two points by the Euclidean distance.
    If one of the features has a broad range of values, the distance will 
    be governed by this particular feature.

"""

import numpy as np
from sklearn import preprocessing

input_data = np.array([[5.1, -2.9, 3.3],
 [-1.2, 7.8, -6.1],
 [3.9, 0.4, 2.1],
 [7.3, -9.9, -4.5]])

# Min max scaling
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nMin max scaled data:\n", data_scaled_minmax)



#Normalization

"""  
    We use the process of normalization to modify the values in the feature vector so
    that we can measure them on a common scale. In machine learning, we use many
    different forms of normalization. Some of the most common forms of normalization
    aim to modify the values so that they sum up to 1. L1 normalization, which refers
    to Least Absolute Deviations, works by making sure that the sum of absolute
    values is 1 in each row. L2 normalization, which refers to least squares, works by
    making sure that the sum of squares is 1.
"""


# Normalize data
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nL1 normalized data:\n", data_normalized_l1)
print("\nL2 normalized data:\n", data_normalized_l2)
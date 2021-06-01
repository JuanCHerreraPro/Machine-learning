# -*- coding: utf-8 -*-
"""
Created on Mon May 31 20:29:35 2021

@author: JuanCHerreraPro

"""

#Binarization

"""   
Binarization is used to convert numerical values into Boolean values. Let's use an
inbuilt method to binarize input_data and using 2.1 as the threshold value.

"""

import numpy as np
from sklearn import preprocessing

input_data = np.array([[5.1, -2.9, 3.3],
 [-1.2, 7.8, -6.1],
 [3.9, 0.4, 2.1],
 [7.3, -9.9, -4.5]])

# Binarize data
data_binarized = preprocessing.Binarizer(threshold=2.0).transform(input_data)
print("\nBinarized data:\n", data_binarized)
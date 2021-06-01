# -*- coding: utf-8 -*-
"""
Created on Mon May 31 21:14:42 2021

@author: JuanCHerreraPro
"""

#Label encoding

"""  
    When performing classification, we usually deal with lots of labels
    Labels are normally words, because words can be understood by humans. 
    To convert word labels into numbers, a label encoder can be used. Label
    encoding refers to the process of transforming word labels into numbers
"""

import numpy as np
from sklearn import preprocessing

# Sample input labels
input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']

#Create label encoder and fit the labels
encoder = preprocessing.LabelEncoder();
encoder.fit(input_labels);

#Print the mapping
for i, item in enumerate(encoder.classes_):
    print(item, '--->', i)
    
# Encode a set of labels using the encoder
test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))

# Decode a set of values using the encoder
encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values =", encoded_values)
print("Decoded labels =", list(decoded_list))

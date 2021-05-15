# Correlation heatmaps
"""
    A correlation exists between two features when there is a relationship between
    the different values of the features. 
    If a feature changes consistently in relation to another feature, these features are said to be highly correlated.

    Correlation can be positive (an increase in one value of a feature increases the value
    of the target variable) or negative (an increase in one value of a feature decreases the
    value of the target variable).
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("train.csv")
X = data.iloc[:,0:20] #Columnas de valores independiente
y = data.iloc[:,-1] # Columna de valor dependiente

#Obtenetemos las coorrelaciones de cada carácteristica del dataset
correlation_matrix = data.corr()
top_corr_features = correlation_matrix.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
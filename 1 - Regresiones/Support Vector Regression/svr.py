# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:48:49 2021

@author: windows
"""

#SVR



# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1))
y = np.ravel(y)

# Ajustar la regresión con el dataset
from sklearn.svm import SVR 
regression = SVR(kernel = "rbf")
regression.fit(X, y)

# Crear aquí nuestro modelo de regresión


# Predicción de nuestros modelos
y_pred = sc_y.inverse_transform(regression.predict(sc_X.transform([[6.5]])))


# Visualización de los resultados del Modelo SRV
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X, regression.predict(X), color = "blue")
plt.title("Modelo de Regresión (SVR)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

# Visualización de los resultados del SVR Reescaladas

Xinverse=sc_X.inverse_transform(X)
yinverse=sc_y.inverse_transform(y)
X_grid = np.arange(min(Xinverse), max(Xinverse), 0.1)
X_grid = X_grid.reshape(-1, 1)
plt.ticklabel_format(style = "plain")
plt.scatter(Xinverse, yinverse, color = "red")
plt.plot(X_grid, sc_y.inverse_transform(regression.predict(sc_X.transform(X_grid))), color = "blue")
plt.title("Modelo de Regresión (SVR reescalado)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()


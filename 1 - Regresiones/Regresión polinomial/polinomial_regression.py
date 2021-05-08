# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:50:04 2021

@author: windows
"""

#Regresión polinómica 
#Carga de librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #Agregamos el ":2" para hacerlo matriz
y = dataset.iloc[:, 2].values


#Dividir el dataset en entrenamiento y testing 
#(En esta ocación, no es necesario por no tener datos repetidos, por lo que al dividir los conjuntos de datos, perderemos información)

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Ajustar la regresión lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Ajustar la regresión polinomica con el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 9)
X_poly = poly_reg.fit_transform(X) #Unicamente se creó la matriz de caracteristicas polinómica
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualización del modelo lineal
plt.scatter(X, y, color ="red")
plt.plot(X, lin_reg.predict(X), color  = "blue")
plt.title("Modelo de Regresión lineal")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

#Visualización del modelo polinomico
plt.scatter(X, y, color ="red")
plt.plot(X, lin_reg_2.predict(X_poly), color  = "blue")
plt.title("Modelo de Regresión polinómica")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()


# Predicción de nuestros modelos
# Se ha añadido la sintaxis de doble corchete necesaria para hacer la predicción en las últimas versiones de Python (3.7+)
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
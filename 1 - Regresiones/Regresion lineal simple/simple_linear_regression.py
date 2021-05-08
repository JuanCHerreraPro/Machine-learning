# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:40:04 2021

@author: windows
"""

#Regresión lineal simple

#Carga de librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar el dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


#Dividir el dataset en entrenamiento y testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Crear el modelo de regresión lineal simple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression 
regression = LinearRegression();
regression.fit(X_train, y_train);

#Predecir el conjunto de test
y_pred = regression.predict(X_test);

#Visualizar el conjunto de entrenamiento 
plt.scatter(X_train, y_train, color="red");
plt.plot(X_train, regression.predict(X_train), color="blue");
plt.title("Sueldo vs Años de experiencia (Conjunto de entrenamiento)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (en dolares)")
plt.show();

#Visualizar el conjunto de test
plt.scatter(X_test, y_test, color="red");
plt.plot(X_train, regression.predict(X_train), color="blue");
plt.title("Sueldo vs Años de experiencia (Conjunto de test)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (en dolares)")
plt.show();
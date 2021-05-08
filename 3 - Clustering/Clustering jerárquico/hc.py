# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:51:43 2021

@author: windows
"""

#Clustering Jerárquico 

#Importar librerías de trabajo 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importar conjunto de datos
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values


#Utilizar el dendrograma para encontrar el número óptimo de clusters 
import scipy.cluster.hierarchy as sch
dendrograma = sch.dendrogram(sch.linkage(X, method = "ward" ))
plt.title("Dendrograma")
plt.xlabel("Clientes")
plt.ylabel("Distancia Euclidea")
plt.show()

#Ajsutar el clustering jerárquico 
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
y_hc = hc.fit_predict(X)

# Visualización de los clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = "red", label = "Cautos")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = "blue", label = "Estandard")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = "green", label = "Objetivo")
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = "cyan", label = "Descuidados")
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = "magenta", label = "Conservadores")
plt.title("Cluster de clientes")
plt.xlabel("Ingresos anuales (en miles de $)")
plt.ylabel("Puntuación de Gastos (1-100)")
plt.legend()
plt.show()
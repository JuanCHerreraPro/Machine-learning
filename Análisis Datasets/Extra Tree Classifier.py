#Feature importance


import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("50_Startups.csv")

#Separado de datos
X = data.iloc[:,0:3]
y = data.iloc[:,-1]

#Cargamos el modelo
model = ExtraTreesClassifier();

model.fit(X, y);

#Importancia de las caracteristicas de Tree based classifiers
print(model.feature_importances_)



#Gráfica de la importancia de las direferentes caracteristicas
feat_importances = pd.Series(model.feature_importances_, index=X.
columns)

feat_importances.nlargest(5).plot(kind='barh')
plt.show()

"""Recuerda que es importante considerar la naturaleza del problema, 
el clásificador utulizado es para valores categoricos y no valores continuos"""

""" A este dataset 50_Startups.csv, se le realizó una modificación en su ultima columna
 para complir con los volores cátegiricos para el vector y"""
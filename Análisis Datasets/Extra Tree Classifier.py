#Feature importance
"""Feature importance provides a score for each feature in a dataset. A higher score
means the feature has more importance or relevancy in relation to the output feature.

Feature importance is normally an inbuilt class that comes with Tree-Based Classifiers.
In the following example, we use the Extra Tree Classifier to determine the top five
features in a dataset:

"""

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_csv("train.csv")

#Separado el vector dependiente de datos
X = data.iloc[:,0:20] #Columnas de valores independientes
y = data.iloc[:,-1] #Columna de valor dependinte

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
el clásificador utulizado es para valores categoricos y no valores continuos """

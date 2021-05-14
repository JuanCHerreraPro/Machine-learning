# Univariate selection
"""
Statistical tests can be used to determine which features have the strongest
correlation to the output variable. The scikit-learn library has a class called
SelectKBest that provides a set of statistical tests to select the K "best" features
in a dataset.
"""

import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data = pd.read_csv("train.csv")

X = data.iloc[:,0:20] #Columnas de valores independientes
y = data.iloc[:,-1] #Columna de valor dependinte

#aplicar SelectKBest para extraer las mejores caracteristicas
bestFeatures = SelectKBest(score_func=chi2, k=5)
fit = bestFeatures.fit(X, y)
dfscores = pd.DataFrame(fit.scores_)


dfcolumns = pd.DataFrame(X.columns)
scores = pd.concat([dfcolumns,dfscores],axis=1)
scores.columns = ['specs','score']
print(scores.nlargest(5,'score')) #Imprimos el orden de las caracteristicas
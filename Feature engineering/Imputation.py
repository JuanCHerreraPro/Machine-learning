
"""
Rows with missing
values is a common problem. There might be many reasons why values are missing:
    • Inconsistent datasets
    • Clerical error
    • Privacy issues
Regardless of the reason, having missing values can affect the performance of
a model, and in some cases, it can bring things to a screeching halt since some
algorithms do not take kindly to missing values. There are multiple techniques to
handle missing values. They include:
"""

#Removing the row with missing values

import pandas as pd

data = pd.read_csv("train_modify.csv")

#Definimos un umbral
threshold = 0.6 #umbral


print(data.isnull().mean())

#Eliminar columnas con una tasa de valor faltante superior al umbral
data = data[data.columns[data.isnull().mean() < threshold]]

#Eliminar filas con una tasa de valor faltante superior al umbral
data = data.loc[data.isnull().mean(axis=1) < threshold]

print(data)


#---------------------------------#
#Numerical Imputation

"""
Numerical Imputation – Imputation is another method to deal with missing values.
Imputation simply means replacing the missing value with another value that
"makes sense"

In the case of numerical variables, these are common replacements:
    • Using zero as the replacement value is one option
    • Calculate the mean for the full dataset and replace the missing value with the
    mean
    • Calculate the average for the full dataset and replace the missing value with
    the average
"""

#Priema opción llenar los vaolores faltantes con 0's
data = data.fillna(0)
print(data)

#Segunda opción llenar los valores faltantes con la media de la columna
data = data.fillna(data.median())
print(data)
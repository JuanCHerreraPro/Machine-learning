# Transformación logaritmica
"""
Logarithm transformation (or log transform) is a common feature engineering
transformation. Log transform helps to flatten highly skewed values. After the log
transformation is applied, the data distribution is normalized.
"""

import pandas as pd
import numpy as np


data = pd.DataFrame({'value':[3,67, -17, 44, 37, 3, 31, -38]})

data['log+1'] = (data['value']+1).transform(np.log)

#Tratamiento de valores negativos
#Notar que los valores son diferentes
data['log'] = (data['value']-data['value'].min()+1) .transform(np.log)

print(data)
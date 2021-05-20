
#Dropping the outlier rows with standard deviation
import pandas as pd

data = pd.read_csv("train.csv")

#Eliminando filas outliers con desviación estandar formula
factor = 2
print(data['battery_power'].mean())
upper_lim = data['battery_power'].mean() + data['battery_power'].std() * factor

lower_lim = data['battery_power'].mean() - data['battery_power'].std() * factor

data2 = data[(data['battery_power'] < upper_lim) & (data['battery_power'] > lower_lim)]

print(data2.shape)


"""
Another method to detect and remove outliers is to use percentiles. With this
method, we simply assume that a certain percentage of the values for a feature are
outliers. What percentage of values to drop is again subjective and it is going to be
domain-dependent.
"""

#Eliminando fulas outliers con porcentajes


upper_lim = data['battery_power'].quantile(.99)
lower_lim = data['battery_power'].quantile(.01)
data3 = data[(data['battery_power'] < upper_lim) & (data['battery_power'] > lower_lim)]
print(data3.shape)
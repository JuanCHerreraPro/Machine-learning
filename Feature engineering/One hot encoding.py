# -*- coding: utf-8 -*-
"""
Some machine learning algorithms cannot handle categorical features,
so one-hot encoding is a way to convert these categorical features into numerical
features. Let's say that you have a feature labeled "status" that can take one of
three values (red, green, or yellow). Because these values are categorical, there
is no concept of which value is higher or lower.
"""

import pandas as pd

data = pd.read_csv("dataset.csv")

encoded_columns = pd.get_dummies(data['color'])

data = data.join(encoded_columns).drop('color', axis=1)

print(data)
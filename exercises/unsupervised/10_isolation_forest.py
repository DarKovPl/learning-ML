import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

sns.set()

data = pd.read_csv('../data/factory.csv')
print(data.head())
print(f'\n{data.describe()}')
px.scatter(
    data,
    x=data.item_length,
    y=data.item_width,
    width=1900,
    height=1000,
    template='plotly_dark',
    title='Isolation Forest'
).show()

outlier = IsolationForest(n_estimators=100, contamination=0.03)
y_pred = outlier.fit_predict(data)
data['outlier_flag'] = y_pred

px.scatter(
    data,
    x='item_length',
    y='item_width',
    color='outlier_flag',
    width=1900,
    height=1000,
    template='plotly_dark',
    color_continuous_midpoint=-1,
    title='Isolation Forest'
).show()

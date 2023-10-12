import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

data = pd.read_csv('../data/OnlineRetail.csv', encoding='latin', parse_dates=['InvoiceDate'])
copied_data = data.copy()
print(f'\n{copied_data.head()}')
print(f'\n{copied_data.info()}')
print(f'\n{copied_data.describe()}')
print(f"\n{copied_data.describe(include=['object'])}")
print(f"\n{copied_data.describe(include=['datetime'])}")
print(f'\n{copied_data.isnull().sum()}')

copied_data = copied_data.dropna()
print(f'\n{copied_data.isnull().sum()}')

print(f'\n{copied_data["Country"].value_counts()}')
tmp = copied_data['Country'].value_counts()
tmp = tmp[tmp > 200].reset_index()
px.bar(
    tmp,
    x='Country',
    y='count',
    template='plotly_dark',
    color_discrete_sequence=['#ff5b03'],
    title='Frequency of purchases by country'
)

df_UK = copied_data.query("Country == 'United Kingdom'").copy()
print(df_UK.head())
print(df_UK.describe())

df_UK['Sales'] = df_UK['Quantity'] * df_UK['UnitPrice']
print(df_UK.head())

tmp = df_UK.groupby(df_UK['InvoiceDate'].dt.date)['CustomerID'].count().reset_index().rename(
    columns=dict(CustomerID='Count')
)
print(f'\n{tmp.head()}')

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

trace1 = px.line(
    tmp, x='InvoiceDate', y='Count', template='plotly_dark', color_discrete_sequence=['#ff5b03'])['data'][0]
trace2 = px.scatter(
    tmp, x='InvoiceDate', y='Count', template='plotly_dark', color_discrete_sequence=['#ff5b03'])['data'][0]

fig.add_trace(trace1, row=1, col=1)
fig.add_trace(trace2, row=2, col=1)
fig.update_layout(template='plotly_dark', title='Purchase frequency by date', width=1900, height=1000)

print("**************************************************************")

tmp = df_UK.groupby(df_UK['InvoiceDate'].dt.date)['Sales'].sum().reset_index()
print(f'\n{tmp.head()}')

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

trace1 = px.line(
    tmp, x='InvoiceDate', y='Sales', template='plotly_dark', color_discrete_sequence=['#ff5b03'])['data'][0]
trace2 = px.scatter(
    tmp, x='InvoiceDate', y='Sales', template='plotly_dark', color_discrete_sequence=['#ff5b03'])['data'][0]

fig.add_trace(trace1, row=1, col=1)
fig.add_trace(trace2, row=2, col=1)
fig.update_layout(template='plotly_dark', title='Total sales by date', width=1900, height=1000)

print("**************************************************************")

data_user = pd.DataFrame(copied_data['CustomerID'].unique(), columns=['CustomerID'])
print(f'\n{data_user}')

last_purchase = df_UK.groupby('CustomerID')['InvoiceDate'].max().reset_index().rename(
    columns=dict(InvoiceDate='LastPurchaseDate')
)
print(f'\n{last_purchase}')

last_purchase['Retention'] = (last_purchase['LastPurchaseDate'].max() - last_purchase['LastPurchaseDate']).dt.days
print(last_purchase.head())
print(last_purchase.Retention.value_counts())

px.histogram(
    last_purchase,
    x='Retention',
    template='plotly_dark',
    width=1900,
    height=1000,
    title='Retention',
    nbins=100,
    color_discrete_sequence=['#ff5b03']
)

data_user = pd.merge(data_user, last_purchase, on='CustomerID')
data_user = data_user[['CustomerID', 'Retention']]
print(data_user.head())

px.scatter(
    data_user,
    x='CustomerID',
    y='Retention',
    template='plotly_dark',
    width=1900,
    height=1000,
    color_discrete_sequence=['#ff5b03']
)
print("**************************************************************")

scaler = StandardScaler()
data_user['RetentionScaled'] = scaler.fit_transform(data_user.Retention.values.reshape(-1, 1))
print(f'\n{data_user.head()}')

px.scatter(
    data_user,
    x='CustomerID',
    y='RetentionScaled',
    template='plotly_dark',
    width=1900,
    height=1000,
    color_discrete_sequence=['#ff5b03']
)

wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, max_iter=1000, n_init='auto')
    kmeans.fit(data_user.RetentionScaled.values.reshape(-1, 1))
    wcss.append(kmeans.inertia_)

wcss = pd.DataFrame(data=np.c_[range(1, 10), wcss], columns=['NumberOfClusters', 'WCSS'])
print(wcss)

px.line(
    wcss,
    x='NumberOfClusters',
    y='WCSS',
    template='plotly_dark',
    title='WCSS',
    width=1900,
    height=1000,
    color_discrete_sequence=['#ff5b03']
)

kmeans = KMeans(n_clusters=3, max_iter=1000, n_init=10)
kmeans.fit(data_user.RetentionScaled.values.reshape(-1, 1))
data_user['Cluster'] = kmeans.labels_
print(f'\n{data_user.head()}')

tmp = data_user.groupby('Cluster')['Retention'].describe()['mean'].reset_index().rename(
    columns=dict(mean='MeanRetention')
)
print(f'\n{tmp}')
px.bar(
    tmp,
    x='Cluster',
    y='MeanRetention',
    template='plotly_dark',
    width=1900,
    height=1000,
    color_discrete_sequence=['#ff5b03']
)
px.scatter(
    data_user,
    x='CustomerID',
    y='Retention',
    color='Cluster',
    template='plotly_dark',
    width=1900,
    height=1000,
    title='Clusters visualisation K-MEAN'
)
print("**************************************************************")

dbscan = DBSCAN(eps=0.03, min_samples=5)
dbscan.fit(data_user.RetentionScaled.values.reshape(-1, 1))
data_user['Cluster'] = dbscan.labels_
print(f'\n{data_user.head()}')

px.scatter(
    data_user,
    x='CustomerID',
    y='Retention',
    color='Cluster',
    template='plotly_dark',
    width=1900,
    height=1000,
    title='Clusters visualisation DBSCAN'
)
print("**************************************************************")

data_sales = df_UK.groupby('CustomerID')['Sales'].sum().reset_index()
print(f'\n{data_sales.head()}')

data_user = pd.merge(data_user, data_sales, on='CustomerID')
print(f'\n{data_user.head()}')

scaler = StandardScaler()
data_user['SalesScaled'] = scaler.fit_transform(data_user.Sales.values.reshape(-1, 1))
print(f'\n{data_user.head()}')

px.scatter(
    data_user,
    x='CustomerID',
    y='Sales',
    template='plotly_dark',
    color_discrete_sequence=['#ff5b03'],
    title='Sales broken down by customer'
)

wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, max_iter=1000, n_init='auto')
    kmeans.fit(data_user.SalesScaled.values.reshape(-1, 1))
    wcss.append((i, kmeans.inertia_))

wcss = pd.DataFrame(data=wcss, columns=['NumberOfClusters', 'WCSS'])
print(f'\n{wcss}')

px.line(
    wcss,
    x='NumberOfClusters',
    y='WCSS',
    template='plotly_dark',
    color_discrete_sequence=['#ff5b03'],
    width=1900,
    height=1000,
    title='WCSS'
)

kmeans = KMeans(n_clusters=3, max_iter=1000, n_init='auto')
kmeans.fit(data_user.SalesScaled.values.reshape(-1, 1))
data_user['Cluster'] = kmeans.labels_
data_user['Cluster'] = data_user.Cluster.astype(str)
print(f'\n{data_user}')
print("**************************************************************")

dbscan = DBSCAN(eps=0.5, min_samples=7)
dbscan.fit(data_user.SalesScaled.values.reshape(-1, 1))
data_user['Cluster'] = dbscan.labels_
data_user['Cluster'] = data_user['Cluster'].astype(str)
print(f'\n{data_user}')

px.scatter(
    data_user,
    x='CustomerID',
    y='Sales',
    color='Cluster',
    template='plotly_dark',
    width=1900,
    height=1000,
    title='DBSCAN - Cluster visualization'
).show()

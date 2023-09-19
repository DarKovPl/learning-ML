import numpy as np
import pandas as pd
import sklearn
import pandas_datareader.data as web_data

df_raw = web_data.DataReader(name='AMZN', data_source='stooq')
print(df_raw.head().to_string())


df = df_raw.copy()
df['day'] = df.index.day
df['month'] = df.index.month
df['year'] = df.index.year
print(df.head(), df.info())

"""
https://pl.wikipedia.org/wiki/Dyskretyzacja_(statystyka)
https://pandas.pydata.org/docs/reference/api/pandas.cut.html
"""

df = pd.DataFrame(data={'height': [175., 178.5, 185., 191., 184.5, 183., 168.]})
print(df)

df['height_cut'] = pd.cut(df.height, bins=5, labels=['very small', 'small', 'medium', 'high', 'very high'])
df['height_cut_amount'] = pd.cut(df.height, bins=5)
print(df)

df.drop(labels='height_cut_amount', axis=1, inplace=True)
df_dummies = pd.get_dummies(df, drop_first=True, prefix='height', dtype='float')
df_dummies.columns = df_dummies.columns.str.replace(' ', '_')

print(df_dummies.to_string())


df = pd.DataFrame(data={'lang': [['PL', 'ENG'], ['GER', 'ENG', 'PL', 'FRA'], ['RUS']]})
print(df)

df['lang_amount'] = df.lang.apply(len)
print(df)

print(type(df.lang[0]))

df['pl_flag'] = df.lang.apply(lambda x: 1 if 'PL' in x else 0)
df = df.assign(pl_flag_=df.lang.str.contains('PL', regex=False))
print(df)


df = pd.DataFrame(data={'website': ['wp.pl', 'onet.pl', 'google.com']})
print(df)

split_df = df.website.str.split('.', expand=True)
df['portal'] = split_df[0]
df['domain'] = split_df[1]
print(df)

print("ciohwhn")



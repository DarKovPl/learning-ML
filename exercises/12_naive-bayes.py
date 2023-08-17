import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

np.random.seed(42)
np.set_printoptions(precision=6, suppress=True)


pogoda = ['słonecznie', 'deszczowo', 'pochmurno', 'deszczowo', 'słonecznie', 'słonecznie',
          'pochmurno', 'pochmurno', 'słonecznie']
temperatura = ['ciepło', 'zimno', 'ciepło', 'ciepło', 'ciepło', 'umiarkowanie',
               'umiarkowanie', 'ciepło', 'zimno']

spacer = ['tak', 'nie', 'tak', 'nie', 'tak', 'tak', 'nie', 'tak', 'nie']

raw_df = pd.DataFrame(data={'pogoda': pogoda, 'temperatura': temperatura, 'spacer': spacer})
df = raw_df.copy()
print(df)

encoder = LabelEncoder()
df['spacer'] = encoder.fit_transform(spacer)
print(df)

df = pd.get_dummies(df, columns=df.columns[:2], drop_first=True, dtype='int')
print(df)

data = df.copy()
target = data.pop('spacer')
print(data)

classifier = GaussianNB()
classifier.fit(data, target)
print(classifier.score(data, target))


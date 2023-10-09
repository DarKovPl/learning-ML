import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

pd.set_option('display.float_format', lambda x: f'{x:.2f}')


products = pd.read_csv('../Data/products.csv', usecols=['product_id', 'product_name'])
orders = pd.read_csv('../Data/orders.csv', usecols=['order_id', 'product_id'])
print(products.head())
print(f'\n{orders.head()}')

data = pd.merge(orders, products, how='inner', on='product_id', sort=True).sort_values(by='order_id')
print(f'\n{data.head()}')
print(f'\n{data.describe()}')

transactions = data.groupby(by='order_id')['product_name'].apply(lambda name: ','.join(name))
print(f'\n{transactions.head().to_markdown()}')

transactions = transactions.str.split(',')
print(f'\n{transactions.head().to_markdown()}')
print("**************************************************************")

encoder = TransactionEncoder()
transactions_encoded = encoder.fit_transform(transactions, sparse=True)
print(f'\n{transactions_encoded}')

df_transactions_encoded = pd.DataFrame(transactions_encoded.toarray(), columns=encoder.columns_)
print(df_transactions_encoded.head())

supports = apriori(df_transactions_encoded, min_support=0.01, use_colnames=True)
supports = supports.sort_values(by='support', ascending=False)
print(f'\n{supports.head()}')

rules = association_rules(supports, metric='confidence', min_threshold=0)
rules = rules.iloc[:, [0, 1, 4, 5, 6]]
rules = rules.sort_values(by='lift', ascending=False)
print(rules.head(15))

import itertools

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, scale

data = {
    "size": ["XL", "L", "M", "L", "M"],
    "color": ["red", "green", "blue", "green", "red"],
    "gender": ["female", "male", "male", "female", "female"],
    "price": [199.0, 89.0, 99.0, 129.0, 79.0],
    "weight": [500, 450, 300, 380, 410],
    "bought": ["yes", "no", "yes", "no", "yes"],
}

df_raw = pd.DataFrame(data=data)
df = df_raw.copy()
print(df.info())

col_to_cat_change = [
    column
    for column in df.columns
    if (not df[column].dtype.name.__contains__("float"))
    and (not df[column].dtype.name.__contains__("int"))
]

for i in col_to_cat_change:
    df[i] = df[i].astype("category")

print(df.describe(include=["category"]).T)

"""
LabelEncoder - we use this for mapping target value. In this case it will be column 'bought'.
"""
le = LabelEncoder()
print(le.fit_transform(df.bought))
print(le.classes_)


"""
OneHotEncoder - we use this for mapping describing values. In this case it will be - 'size', 'color', 'gender', 'price', 'weight'
"""
encoder = OneHotEncoder(sparse_output=False)
print(encoder.fit_transform(df[["size"]]))
print(encoder.categories_)
print("-" * 20)

encoder = OneHotEncoder(drop="first", sparse_output=False)
print(encoder.fit_transform(df[["size"]]))
print(encoder.categories_)

"""
Standaryzacja - StandardScaler
std() - pandas nieobciążony
std() - numpy obciążony
"""

print(f"{df['price']}")
print(f"Średnia: {df['price'].mean()}")
print(f"Odchylenie standardowe: {df['price'].std():.4f}")

print("from sklearn.preprocessing import scale -> ", scale(df["price"]))
print("-" * 20)

scaler = StandardScaler()
print(scaler.fit_transform(df[["price"]]))
print("-" * 20)

scaler = StandardScaler()
df[["price", "weight"]] = scaler.fit_transform(df[["price", "weight"]])
print(df)
print("*" * 20)

df = df_raw.copy()
print(df)
print("|" * 20)

le = LabelEncoder()
df.bought = le.fit_transform(df.bought)
print(df)
print("|" * 20)

encoder = OneHotEncoder(sparse_output=False, drop="first").set_output(
    transform="pandas"
)
encodered = encoder.fit_transform(df)

index_to_drop_in_df = [
    index
    for index, _ in enumerate(encoder.categories_)
    if (not encoder.categories_[index].dtype.name.__contains__("float"))
    and (not encoder.categories_[index].dtype.name.__contains__("int"))
]
df.drop(df.columns[index_to_drop_in_df], axis=1, inplace=True)

col_nam_to_drop_in_encoded = [
    encodered.loc[:, encodered.columns.str.contains(col_name)].columns.to_list()
    for col_name in df.columns
]
encodered.drop(list(itertools.chain(*col_nam_to_drop_in_encoded)), axis=1, inplace=True)

df = pd.concat([df, encodered], axis=1)
df[["price", "weight"]] = scaler.fit_transform(df[["price", "weight"]])
print(df.to_string())


print("|" * 20)
df = df_raw
df = pd.get_dummies(data=df, drop_first=True, dtype="int")
df[["price", "weight"]] = scaler.fit_transform(df[["price", "weight"]])
print(df.to_string())

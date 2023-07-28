import numpy as np
from sklearn.linear_model import LinearRegression

"""
Równanie normalne
"""
X1 = np.array([1, 2, 3, 4, 5, 6])
Y = np.array([3000, 3250, 3500, 3750, 4000, 4250])
m = len(X1)
X1 = X1.reshape(m, 1)

print(f'Lata pracy: {X1}')
print(f'Wynagrodzenie: {Y}')
print(f'Liczba próbek: {m}')

regression = LinearRegression()
regression.fit(X1, Y)
print(regression.intercept_)
print(regression.coef_)
print('*|' * 40)

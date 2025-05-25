from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("score_updated.csv")

print(df.head())
print(df.describe())

plt.scatter(df['Hours'], df['Scores'])
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.show()

model = LinearRegression()
model.fit(df[['Hours']], df[['Scores']])

X = df[['Hours']]
y = df[['Scores']]

y_pred = model.predict(X)


hours = np.array([[5]])
print(model.predict(hours))
print(f"Slope (m): {model.coef_[0][0]}")
print(f"Intercept (b): {model.intercept_[0]}")
print(f"Mean squared error: {mean_squared_error(y, y_pred)}")
print(f"Coefficient of determination: {r2_score(y, y_pred)}")

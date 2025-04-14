import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv('data/triangles.csv')
X = df[['leg_a', 'leg_b']]
y = df['hypotenuse']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("True Hypotenuse")
plt.ylabel("Predicted Hypotenuse")
plt.title("True vs Predicted Hypotenuse")
plt.grid(True)
plt.savefig("src/hypotenuse_plot.png")
plt.show()


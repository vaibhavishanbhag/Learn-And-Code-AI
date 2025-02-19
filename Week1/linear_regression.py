import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data: Square footage (X) vs House Price (Y)
X = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)
Y = np.array([150000, 200000, 250000, 300000, 350000])

# Train Linear Regression model
model = LinearRegression()
model.fit(X, Y)

# Predict for a new house size
new_house = np.array([[2200]])
predicted_price = model.predict(new_house)
print("Predicted Price:", predicted_price[0])

# Plot the regression line
plt.scatter(X, Y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', label="Regression Line")
plt.xlabel("Square Footage")
plt.ylabel("House Price")
plt.legend()
plt.show()



# Get the learned parameters
w = model.coef_[0]
b = model.intercept_

print(f"Equation: Y = {w:.4f} * X + {b:.4f}")

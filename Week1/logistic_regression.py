from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

# Data: Study Hours (X) and whether student passed (Y: 1=Pass, 0=Fail)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # Pass/Fail

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X, Y)

# Predict for a new student who studied 4.5 hours
print("Predicted Probability of Passing:", model.predict_proba([[4.5]])[0][1])


# Generate values for smooth curve
X_test = np.linspace(0, 11, 100).reshape(-1, 1)  # Smooth range from 0 to 11
Y_prob = model.predict_proba(X_test)[:, 1]  # Get probability of passing (Y=1)

# Prediction for a student who studied 4.5 hours
X_new = np.array([[4.5]])
Y_new_prob = model.predict_proba(X_new)[0][1]

# Plot the logistic regression sigmoid curve
plt.plot(X_test, Y_prob, color='red', label="Logistic Regression Curve")

# Plot actual data points
plt.scatter(X, Y, color='blue', label="Actual Data", zorder=3)

# Highlight the new prediction (4.5 hours)
plt.scatter(4.5, Y_new_prob, color='green', s=100, label=f"Prediction for 4.5 hrs ({Y_new_prob:.2f})", edgecolors='black')

# Labels and title
plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.title("Logistic Regression - Probability of Passing vs Study Hours")
plt.legend()
plt.grid()
plt.show()


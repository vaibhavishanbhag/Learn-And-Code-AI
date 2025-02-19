# Questions and Answers

## 📌 Python Basics and Data Structures
### 1. Can we add elements of different data types in the same list?
**Answer**: Yes, Python lists can hold elements of different data types. For example, you can have a list that contains integers, strings, and other data types in the same list.

### 2. What is the meaning of `self` in Python?
**Answer**: `self` is a reference to the current instance of the class. It allows us to access variables and methods within the class. It's a way of referring to the instance within class methods.

### 3. What is the purpose of `__init__` in Python classes?
**Answer**: The `__init__` method is a special method in Python that is called when a new object of the class is created. It initializes the object's attributes.

### 4. How to make a class variable "private" using the `_` notation?
**Answer**: Using a single underscore (e.g., `_variable`) is a convention to indicate that a variable is intended to be private, though it is still accessible from outside the class. If you use double underscores (e.g., `__variable`), Python will name-mangle the variable, making it more challenging to access from outside.

### 5. What is `super()` in Python?
**Answer**: `super()` is used to call a method from a parent class within a child class. It allows you to call and extend the functionality of a method in the parent class.

### 6. If we want to print the method of the child class, why should we use inheritance in Python?
**Answer**: Inheritance is used when we want the child class to inherit properties and methods from the parent class, allowing code reuse and extension. If you want to override a method, you can do that in the child class using inheritance.

### 7. Can I define a class without inheritance but still use the same method as the parent?
**Answer**: Yes, you can define a class and have the same method, but without inheritance, the method will only exist in that specific class. Inheritance allows sharing and overriding methods across classes.

### 8. What is the purpose of the `in` operator in Python for dictionaries?
**Answer**: The `in` operator is used to check if a specific key exists in a dictionary.

### 9. How do we access the key-value pairs in a dictionary?
**Answer**: You can use a loop with the `items()` method to access all key-value pairs in a dictionary.

### 10. What is the meaning of Polymorphism in OOP?
**Answer**: Polymorphism allows objects of different classes to be treated as objects of a common base class. It also refers to the ability of a class to define methods that can be overridden in subclasses.

# Questions & Answers

## AI Basics

### 1. What is the equation for linear regression?
**Equation:**  
Linear regression predicts `Y` based on `X` using:
\`
Y = mX + b
\`
where:
- `m` = slope (coefficient),
- `b` = intercept.

### 2. What is the equation for logistic regression?
**Equation:**  
Logistic regression models probability using the sigmoid function:
\`
P(Y=1 | X) = 1 / (1 + e^(- (mX + b) ) )
\`
where `mX + b` is the linear combination of inputs.

### 3. How do I plot the sigmoid curve for logistic regression?
Use Matplotlib to visualize the sigmoid function:

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.linspace(-10, 10, 100)
Y = sigmoid(X)

plt.plot(X, Y, label="Sigmoid Curve", color='red')
plt.xlabel("X")
plt.ylabel("Probability")
plt.title("Sigmoid Function in Logistic Regression")
plt.legend()
plt.grid()
plt.show()
```

### 4. How do I plot the regression line using Matplotlib?
Use Matplotlib to visualize the linear regression line:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X, Y)

plt.scatter(X, Y, color='blue', label="Actual Data")
plt.plot(X, model.predict(X), color='red', label="Regression Line")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.legend()
plt.show()
```

## 📌 Machine Learning (ML)

### 5. In logistic regression, what does the following X represent?
```python
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # Pass/Fail
```
**Answer:**  
- `X` represents **study hours** of students.
- `Y` represents **pass (1) or fail (0)**.

### 6. How do I predict probabilities in logistic regression?
```python
from sklearn.linear_model import LogisticRegression

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X, Y)

# Predict probability for a student who studied 4.5 hours
prob = model.predict_proba([[4.5]])[0][1]
print("Predicted Probability of Passing:", prob)
```

### 7. What do the feature values in X represent in the following spam detection example?
```python
X = [[1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 1]]
Y = [1, 0, 1, 0]  # 1=Spam, 0=Not Spam
```
**Answer:**  
- Each **X row represents an email**, and the columns represent whether words like `["offer", "win", "free"]` appear.
- `1` means the word appears, `0` means it doesn’t.
- Example: `[1, 1, 1]` → Email contains all three words → More likely spam (`Y=1`).

## 📌 Clustering (K-Means)

### 8. What does the X array represent in the following K-Means clustering example?
```python
X = np.array([[500, 50], [520, 55], [700, 80], [710, 85], [300, 20]])
```
**Answer:**  
Each row in `X` represents **a customer's spending behavior**, where:
- First column = **Total amount spent ($)**
- Second column = **Number of transactions**

### 9. What does `Cluster: [0]` mean in K-Means clustering?
**Answer:**  
- The predicted **cluster label** tells which group the new data point belongs to.
- `0` means it belongs to the **first cluster** (similar spending behavior to customers in that group).

## 📌 Reinforcement Learning (Q-Learning)

### 10. How does Q-learning work in a 3x3 grid environment?
**Answer:**  
- The agent moves in a **3x3 grid**.
- It updates Q-values using the formula:
  \`
  Q(s,a) = Q(s,a) + α [reward + γ max(Q) - Q(s,a)]
  \`
- **Goal:** The agent learns to reach the goal at `(2,2)` efficiently.

### 11. How do I visualize an agent’s movement in a Q-learning environment?
Use Matplotlib to plot the **path taken by the agent**:

```python
import matplotlib.pyplot as plt
import numpy as np

grid_size = 3
path = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]  # Example path learned

grid = np.zeros((grid_size, grid_size))
for step in path:
    grid[step] = 0.5  # Mark path

grid[2, 2] = 1  # Goal

plt.imshow(grid, cmap="coolwarm", origin="upper")
plt.xticks(range(grid_size))
plt.yticks(range(grid_size))
plt.grid(True)
plt.title("Agent's Path to Goal")
plt.show()
```

---

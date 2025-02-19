from sklearn.tree import DecisionTreeClassifier

# Sample data: Email contains ("offer", "win", "free") -> Spam or Not Spam
X = [[1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 1]]
Y = [1, 0, 1, 0]  # 1=Spam, 0=Not Spam

# Train Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X, Y)

# Predict for a new email: ["offer", "win", "free"]
print("Spam Prediction:", model.predict([[1, 1, 1]]))

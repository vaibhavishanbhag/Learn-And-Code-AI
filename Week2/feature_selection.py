
# Continuation after add_visualize_feature.py

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define base features (before encoding)
features = ["Pclass", "Age", "Fare", "FamilySize", "IsAlone", "HasCabin"]

# One-hot encode categorical variables
train_df = pd.get_dummies(train_df, columns=["Sex", "Embarked", "Title"], drop_first=True)

# Add encoded columns to features
features += list(train_df.columns[train_df.columns.str.startswith(("Sex_", "Embarked_", "Title_"))])

# Train a Random Forest to check feature importance
X = train_df[features]
y = train_df["Survived"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Plot feature importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=feature_importance_df["Importance"], y=feature_importance_df["Feature"])
plt.title("Feature Importance for Survival Prediction")
plt.show()

# Define feature columns (exclude 'Survived' from features)
X = train_df.drop(columns=["Survived"])
y = train_df["Survived"]

# Split into training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Drop unnecessary categorical columns before training
X_train = X_train.drop(columns=["Name", "Ticket", "Cabin"], errors="ignore")
X_test = X_test.drop(columns=["Name", "Ticket", "Cabin"], errors="ignore")


# Train the model again after dropping invalid columns
model.fit(X_train, y_train)

# Ensure feature alignment
importance_values = model.feature_importances_
num_features = len(importance_values)

# Match feature names with importance values
feature_names = X_train.columns[:num_features]

# Create DataFrame
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance_values
}).sort_values(by="Importance", ascending=False)

# Display
print(feature_importance)


# Set importance threshold
threshold = 0.02  

# Select features with importance above threshold
selected_features = feature_importance[feature_importance["Importance"] > threshold]["Feature"].tolist()

# Keep only selected features in dataset
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Check selected features
print("Selected Features:", selected_features)

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Train XGBoost on selected features
xgb_model = XGBClassifier()
xgb_model.fit(X_train_selected, y_train)

# Predict on test data
y_pred_xgb = xgb_model.predict(X_test_selected)
xgb_acc = accuracy_score(y_test, y_pred_xgb)

print("XGBoost Accuracy:", round(xgb_acc * 100, 2), "%")


from sklearn.ensemble import RandomForestClassifier

# Train Random Forest on selected features
rf_model = RandomForestClassifier()
rf_model.fit(X_train_selected, y_train)

# Predict on test data
y_pred_rf = rf_model.predict(X_test_selected)
rf_acc = accuracy_score(y_test, y_pred_rf)

print("Random Forest Accuracy:", round(rf_acc * 100, 2), "%")

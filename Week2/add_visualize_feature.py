import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load train and test datasets
train_df = pd.read_csv("Week1/train.csv")

# Select only numeric columns
numeric_df = train_df.select_dtypes(include=["number"])

# Check feature correlation
corr = numeric_df.corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation with Survival")
# plt.show()


# Create FamilySize feature
train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1

# Extract Title from Name
train_df["Title"] = train_df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())

# Create IsAlone feature
train_df["IsAlone"] = (train_df["FamilySize"] == 1).astype(int)

# Create HasCabin feature
train_df["HasCabin"] = train_df["Cabin"].notnull().astype(int)

# Check new features
train_df[["FamilySize", "Title", "IsAlone", "HasCabin"]].head()

# Check updated table with new features
print(train_df.head())


import seaborn as sns
import matplotlib.pyplot as plt

# Plot Survival Rate by FamilySize
plt.figure(figsize=(6, 4))
sns.barplot(x=train_df["FamilySize"], y=train_df["Survived"], ci=None)
plt.title("Survival Rate by Family Size")
# plt.show()

# Plot Survival Rate by Title
plt.figure(figsize=(8, 4))
sns.barplot(x=train_df["Title"], y=train_df["Survived"], ci=None)
plt.xticks(rotation=45)
plt.title("Survival Rate by Title")
# plt.show()

# Plot Survival Rate by IsAlone
plt.figure(figsize=(6, 4))
sns.barplot(x=train_df["IsAlone"], y=train_df["Survived"], ci=None)
plt.xticks([0, 1], ["Not Alone", "Alone"])
plt.title("Survival Rate by Traveling Alone")
# plt.show()

# Plot Survival Rate by HasCabin
plt.figure(figsize=(6, 4))
sns.barplot(x=train_df["HasCabin"], y=train_df["Survived"], ci=None)
plt.xticks([0, 1], ["No Cabin", "Has Cabin"])
plt.title("Survival Rate by Cabin Availability")
# plt.show()


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
# plt.show()

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
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_selected, y_train)

# Predict on test data
y_pred_rf = rf_model.predict(X_test_selected)
rf_acc = accuracy_score(y_test, y_pred_rf)

print("Random Forest Accuracy:", round(rf_acc * 100, 2), "%")


from lightgbm import LGBMClassifier

# Train LightGBM on selected features
lgbm_model = LGBMClassifier()
lgbm_model.fit(X_train_selected, y_train)

# Predict on test data
y_pred_lgbm = lgbm_model.predict(X_test_selected)
lgbm_acc = accuracy_score(y_test, y_pred_lgbm)

print("LightGBM Accuracy:", round(lgbm_acc * 100, 2), "%")


from lightgbm import LGBMClassifier

# Manually adjusted parameters based on previous results
best_lgbm = LGBMClassifier(
    n_estimators=300,  # Increased from 100
    learning_rate=0.05,  
    max_depth=5,  # Increased from 3
    num_leaves=40,  # Slightly reduced from 50 to avoid overfitting
    min_child_samples=10,  # Reduced for better learning
    subsample=1.0,
    colsample_bytree=1.0
)

# Train model
best_lgbm.fit(X_train_selected, y_train)

# Evaluate model
y_pred_best_lgbm = best_lgbm.predict(X_test_selected)
best_lgbm_acc = accuracy_score(y_test, y_pred_best_lgbm)

print("Manually Tuned LightGBM Accuracy:", round(best_lgbm_acc * 100, 2), "%")

from sklearn.impute import SimpleImputer

# Initialize imputer
imputer = SimpleImputer(strategy="median")  # You can use "mean" or "most_frequent" if needed

# Fit and transform train & test sets
X_train_selected = pd.DataFrame(imputer.fit_transform(X_train_selected), columns=X_train_selected.columns)
X_test_selected = pd.DataFrame(imputer.transform(X_test_selected), columns=X_test_selected.columns)


from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Define base models
base_models = [
    ('xgb', xgb_model),
    ('rf', rf_model),
    ('lgbm', best_lgbm)  # Use the best LightGBM model
]

# Meta-model (Logistic Regression helps blend predictions)
stacking_model = StackingClassifier(
    estimators=base_models, 
    final_estimator=LogisticRegression(),
    passthrough=True  # Pass raw features + predictions to the meta-model
)

# Train stacking model
stacking_model.fit(X_train_selected, y_train)

# Predict on test set
y_pred_stack = stacking_model.predict(X_test_selected)

# Evaluate accuracy
stacking_acc = accuracy_score(y_test, y_pred_stack)
print(f"Stacking Model Accuracy: {stacking_acc * 100:.2f} %")


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Define the model
rf = RandomForestClassifier(random_state=42)

# Define hyperparameter grid
param_dist = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Randomized Search
random_search = RandomizedSearchCV(
    rf, param_dist, cv=5, n_iter=20, n_jobs=-1, verbose=2
)
random_search.fit(X_train_selected, y_train)

# Best Model
best_rf = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)

# Evaluate on Test Data
y_pred_best_rf = best_rf.predict(X_test_selected)
best_rf_acc = accuracy_score(y_test, y_pred_best_rf)
print(f"Tuned Random Forest Accuracy: {best_rf_acc * 100:.2f} %")

best_rf_manual = RandomForestClassifier(
    n_estimators=200,         # More trees for better stability
    max_depth=20,             # Prevent overfitting
    min_samples_split=5,      # Slightly higher for generalization
    min_samples_leaf=2,       # More flexibility than 4
    max_features='sqrt',      # Allow more features per split
    bootstrap=True,
    random_state=42
)

best_rf_manual.fit(X_train_selected, y_train)
y_pred_manual_rf = best_rf_manual.predict(X_test_selected)
manual_rf_acc = accuracy_score(y_test, y_pred_manual_rf)

print(f"Manually Tuned Random Forest Accuracy: {manual_rf_acc * 100:.2f} %")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Get feature importance from the best RF model
importances = best_rf_manual.feature_importances_
feature_names = X_train_selected.columns

# Create DataFrame for visualization
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x=feat_imp_df["Importance"], y=feat_imp_df["Feature"])
plt.title("Feature Importance - Random Forest")
# plt.show()

# Print the least important features
print(feat_imp_df.tail(5))


# Drop low-importance features
low_importance_features = ["Embarked_S", "Parch", "SibSp", "HasCabin", "Title_Miss"]
X_train_selected = X_train_selected.drop(columns=low_importance_features, errors="ignore")
X_test_selected = X_test_selected.drop(columns=low_importance_features, errors="ignore")

# Train Random Forest Again
rf_model = RandomForestClassifier()
rf_model.fit(X_train_selected, y_train)
y_pred_rf = rf_model.predict(X_test_selected)
rf_acc = accuracy_score(y_test, y_pred_rf)
print("Updated Random Forest Accuracy:", round(rf_acc * 100, 2), "%")

# Train LightGBM Again
xgb_model = XGBClassifier()
xgb_model.fit(X_train_selected, y_train)
y_pred_xgb = xgb_model.predict(X_test_selected)
xgb_acc = accuracy_score(y_test, y_pred_xgb)
print("Updated LightGBM Accuracy:", round(xgb_acc * 100, 2), "%")

from sklearn.ensemble import VotingClassifier

# Define the ensemble model
ensemble_model = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()), 
        ('lgbm', XGBClassifier())
    ], 
    voting='soft'  # Use 'hard' for majority vote, 'soft' for probability-based voting
)

# Train the ensemble model
ensemble_model.fit(X_train_selected, y_train)

# Predict and evaluate
y_pred_ensemble = ensemble_model.predict(X_test_selected)
ensemble_acc = accuracy_score(y_test, y_pred_ensemble)

print("Ensemble Model Accuracy:", round(ensemble_acc * 100, 2), "%")

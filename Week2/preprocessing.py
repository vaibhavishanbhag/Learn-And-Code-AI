import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Load train and test datasets
train_df = pd.read_csv("Week1/train.csv")
test_df = pd.read_csv("Week1/test.csv")

# Save PassengerId for test set (for final predictions later)
test_passenger_ids = test_df["PassengerId"]

# Drop unnecessary columns from both datasets
drop_cols = ["Name", "Ticket", "Cabin", "PassengerId"]
train_df.drop(columns=drop_cols, inplace=True)
test_df.drop(columns=drop_cols, inplace=True)

# Fill missing values using dictionary method
for df in [train_df, test_df]:
    df.fillna({"Age": df["Age"].median(), "Embarked": df["Embarked"].mode()[0], "Fare": df["Fare"].median()}, inplace=True)

    # Encode categorical variables
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = LabelEncoder().fit_transform(df["Embarked"])

    # Feature engineering: Create FamilySize feature
    df["FamilySize"] = df["SibSp"] + df["Parch"]

# Separate features and target from train set
X = train_df.drop(columns=["Survived"])
y = train_df["Survived"]

# Standardize features (important for PCA & t-SNE)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
test_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)

# Show processed data
print(X_scaled.head())



from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Apply PCA to reduce dimensions
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
X_pca = pca.fit_transform(X_scaled)

# Convert to DataFrame
X_pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

# Scatter plot of the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca_df["PC1"], X_pca_df["PC2"], c=y, cmap="coolwarm", alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA: Titanic Dataset (2D Projection)")
plt.colorbar(label="Survived")
plt.show()

# Check variance explained by components
explained_variance = pca.explained_variance_ratio_
print(f"Variance Explained by PC1: {explained_variance[0]*100:.2f}%")
print(f"Variance Explained by PC2: {explained_variance[1]*100:.2f}%")
print(f"Total Variance Explained: {np.sum(explained_variance)*100:.2f}%")


# Apply PCA with optimal number of components (6)
pca_optimal = PCA(n_components=6)
X_pca_optimal = pca_optimal.fit_transform(X_scaled)

# Check how much variance is retained
explained_variance_optimal = np.sum(pca_optimal.explained_variance_ratio_) * 100
print(f"Total Variance Retained with 6 Components: {explained_variance_optimal:.2f}%")

from sklearn.manifold import TSNE

# Apply t-SNE on PCA-reduced data (6 components)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
X_tsne = tsne.fit_transform(X_pca_optimal)

# Convert to DataFrame
X_tsne_df = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])

# Plot t-SNE result
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne_df["TSNE1"], X_tsne_df["TSNE2"], c=y, cmap="coolwarm", alpha=0.7)
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE: Titanic Dataset Clustering")
plt.colorbar(label="Survived")
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split the dataset into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_pca_optimal, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)

# Model Evaluation
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Model Evaluation
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Greens", xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()





from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Train XGBoost Model
xgb_model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Model Evaluation
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))

# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt="d", cmap="Blues", xticklabels=["Not Survived", "Survived"], yticklabels=["Not Survived", "Survived"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.show()


from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Define the XGBoost model
xgb_model = XGBClassifier(eval_metric='logloss')


# Hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0]
}

# Randomized search
random_search = RandomizedSearchCV(
    estimator=xgb_model, 
    param_distributions=param_grid, 
    n_iter=10, 
    scoring='accuracy', 
    cv=5, 
    verbose=1, 
    n_jobs=-1
)

# Train on your dataset (replace X_train, y_train with actual variables)
random_search.fit(X_train, y_train)

# Best parameters and accuracy
print("Best Parameters:", random_search.best_params_)
print("Best Accuracy:", random_search.best_score_)


final_xgb = XGBClassifier(
    subsample=0.8,
    n_estimators=100,
    min_child_weight=5,
    max_depth=5,
    learning_rate=0.1,
    eval_metric='logloss'
)

final_xgb.fit(X_train, y_train)

# Evaluate the model
y_pred = final_xgb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Final XGBoost Accuracy: {accuracy:.4f}")


import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Define the objective function for Bayesian Optimization
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }

    model = XGBClassifier(**params, eval_metric='logloss')
    
    # Perform cross-validation
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    return np.mean(score)

# Run Bayesian Optimization with 50 trials
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Print best parameters
print("Best Parameters:", study.best_params)
print("Best Accuracy:", study.best_value)

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Train the final model with the best parameters
final_xgb = XGBClassifier(
    n_estimators=321,
    max_depth=10,
    learning_rate=0.0104,
    min_child_weight=6,
    subsample=0.688,
    colsample_bytree=0.666,
    eval_metric='logloss'
)

final_xgb.fit(X_train, y_train)

# Predict on test set
y_pred = final_xgb.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)

print(f"Final XGBoost Accuracy after Bayesian Optimization: {final_accuracy:.4f}")

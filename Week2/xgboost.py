
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

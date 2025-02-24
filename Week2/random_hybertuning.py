from sklearn.model_selection import GridSearchCV

# Define parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search with Cross-Validation
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", rf_grid.best_params_)

# Train the best model
best_rf = rf_grid.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

# Evaluate the tuned model
print("Tuned Random Forest Accuracy:", accuracy_score(y_test, y_pred_best_rf))

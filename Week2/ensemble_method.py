# Continuation after feature_selection.py

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Initialize models with best parameters
xgb_model = XGBClassifier(n_estimators=200, max_depth=5, use_label_encoder=False, eval_metric='logloss')
rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=1, min_samples_split=2)

# Create a Voting Classifier (Soft Voting gives better results for probabilities)
ensemble_model = VotingClassifier(estimators=[
    ('xgb', xgb_model),
    ('rf', rf_model)
], voting='soft')  # Change to 'hard' for majority voting

# Train the ensemble model
ensemble_model.fit(X_train_selected, y_train)

# Predict on test data
y_pred_ensemble = ensemble_model.predict(X_test_selected)

# Evaluate accuracy
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
print(f"Ensemble Model Accuracy: {ensemble_accuracy * 100:.2f} %")


# Train the XGBoost model
xgb_model.fit(X_train_selected, y_train)

# Train the Random Forest model
rf_model.fit(X_train_selected, y_train)


# Get probability predictions from both models
rf_probs = rf_model.predict_proba(X_test_selected)[:, 1]
xgb_probs = xgb_model.predict_proba(X_test_selected)[:, 1]

# Weighted averaging
final_probs = (0.7 * rf_probs) + (0.3 * xgb_probs)

# Convert probabilities to binary predictions (threshold 0.5)
final_preds = (final_probs >= 0.5).astype(int)

# Evaluate the new ensemble
final_accuracy = accuracy_score(y_test, final_preds)
print(f"Weighted Ensemble Accuracy: {final_accuracy * 100:.2f} %")

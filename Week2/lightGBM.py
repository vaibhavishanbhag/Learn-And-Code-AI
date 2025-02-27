
from lightgbm import LGBMClassifier

# Train LightGBM on selected features
lgbm_model = LGBMClassifier()
lgbm_model.fit(X_train_selected, y_train)

# Predict on test data
y_pred_lgbm = lgbm_model.predict(X_test_selected)
lgbm_acc = accuracy_score(y_test, y_pred_lgbm)

print("LightGBM Accuracy:", round(lgbm_acc * 100, 2), "%")
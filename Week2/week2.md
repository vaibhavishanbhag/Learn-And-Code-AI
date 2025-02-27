# Week2 

## Day 1 - Feature Engineering, Dimensionality Reduction & Introduction to Neural Networks

### Objective
Today, we focused on the **Feature Engineering & Dimensionality Reduction** which includes Feature extraction using PCA, LDA or t-SNE and Handling imbalanced data (SMOTE, class weighting).Next we will learn about **Neural Networks** which includes **Understanding perceptrons & multi-layer networks**,Implementing **forward & backward propagation** and then Coding a simple Neural Network from scratch (before using TensorFlow/PyTorch).

#### Feature Engineering & Dimensionality Reduction
Before starting, let's learn about PCA & t-SNE.

##### Overview of PCA & t-SNE

1. **Principal Component Analysis (PCA)** – Linear Dimensionality Reduction
PCA is a technique used to reduce the number of features (dimensions) in a dataset while preserving the most important information.

 **How it works**:

- Finds the directions (principal components) that maximize variance in the data.
- Projects data onto these components to reduce dimensionality.

 **Why use PCA?**

- Removes redundant features
- Speeds up training by reducing feature space
- Helps with visualization (reducing data to 2D or 3D)

**When to use PCA?**

- When you have many correlated features
- When you want to speed up training for ML models

2. **t-SNE (t-Distributed Stochastic Neighbor Embedding)** – Nonlinear Dimensionality Reduction
t-SNE is mainly used for visualizing high-dimensional data in 2D or 3D. Unlike PCA, it preserves the local structure of the data, making it great for clustering.

**How it works:**

- Converts high-dimensional distances into probabilities
- Maps data points to a lower-dimensional space (2D or 3D)
- Ensures that similar points stay close together

**Why use t-SNE?**

- Great for visualizing clusters
- Captures nonlinear relationships better than PCA

**When to use t-SNE?**

- When you want to understand patterns in data
- When PCA isn’t preserving enough structure

Let's start with Step 1: Preprocessing the Titanic dataset before applying PCA and t-SNE.

**Step 1: Feature Engineering on Titanic Dataset**
- Load the dataset and check for missing values
- Encode categorical features (e.g., Sex, Embarked)
- Create new features (e.g., Family Size = SibSp + Parch)
- Apply PCA or t-SNE for dimensionality reduction
- Handle class imbalance if needed

📌 Task: Transform Titanic data into a feature-rich, ML-ready format and visualize PCA/t-SNE results.

**Step 2: Apply PCA (Principal Component Analysis)**
Now that our Titanic dataset is preprocessed, let’s:
- Apply PCA to reduce the number of features.
- Visualize how much variance is retained.
- Plot the first two principal components in 2D.

![alt text](images/pca.png)

Since PCA with 2 components explains only 55% of the variance, we might need more components to retain more information.

📌 Adjusting PCA: **Find Optimal Number of Components**
Let’s check how many components explain at least 90% of the variance.

![alt text](images/pca_components.png)

**Output** : Optimal number of components to retain 90% variance: 6

- A line plot showing how variance increases with more components.
- The optimal number of components for at least 90% variance.


Once we find the best number of components, we’ll:
- Apply PCA with that number of components.
- Then, apply t-SNE for better clustering visualization

**Step 1: Apply PCA with 6 Components**

Total Variance Retained with 6 Components: **95.37%**

**Step 2: Apply t-SNE for Visualization**

Since PCA is linear, we’ll now apply t-SNE, which captures nonlinear patterns & clusters better.

![alt text](images/t-SNE.png)

**Observations from Your Plot:**
1. Two distinct clusters can be seen.

- The red dots (Survived = 1) are concentrated in specific areas.
- The blue dots (Survived = 0) are more widespread.

2. Some overlap exists between survivors and non-survivors.

This suggests that some passengers had similar features but different survival outcomes.

3. Better separation of clusters means survival was predictable

- If we see clear, distinct groups, survival depended strongly on features like class, age, gender, etc.
- If there is too much overlap, survival might involve complex, hidden factors.

**Step 3: Train a Classification Model (Survival Prediction)**
Now, let’s build a machine learning model to predict Survived (0 or 1) based on the Titanic dataset.

**Choose a Model**
We'll start with Logistic Regression (a simple model) and then try Random Forest (a more powerful model).

**Train a Logistic Regression Model**
![alt text](images/train_logistic.png)

**Step 4: Train a Random Forest Classifier**
Random Forest is an ensemble learning method that uses multiple decision trees to improve accuracy.

![alt text](images/random_forest.png)

Your accuracy didn't improve, but the confusion matrix shifted slightly:

- More survivors were correctly classified (51 → 55).
- Slight decrease in non-survivor correct classification (90 → 86).

This suggests that Random Forest is slightly better at identifying survivors but doesn't significantly improve overall accuracy.


##### Next Steps for Improvement

**Hyperparameter Tuning for Random Forest**
Instead of using default settings, we can fine-tune the number of trees, depth, and splits to improve performance.

**Tuned Random Forest Accuracy: 83.24%** 🚀

Your best parameters:
- max_depth=5
- min_samples_leaf=1
- min_samples_split=2
- n_estimators=200

This means:
- A deeper tree (max_depth=5) is helping capture patterns better.
- More estimators (200 trees) improve the model's robustness.

**Step 5: Train XGBoost for Better Accuracy**
XGBoost is a powerful gradient boosting model that often outperforms Random Forest.

![alt text](images/xgboost.png)
Your XGBoost model is working, and you've achieved an accuracy of 81.56%. 

Best Parameters Found:
{
    'subsample': 0.8,
    'n_estimators': 300,
    'min_child_weight': 5,
    'max_depth': 3,
    'learning_rate': 0.01
}

**Final XGBoost Accuracy: 83.24%**


##### Advanced Hyperparameter Tuning with Bayesian Optimization
Since Random Search and Grid Search explore a fixed number of combinations, they may miss better hyperparameters. Instead, we’ll use **Bayesian Optimization with Optuna**, which intelligently chooses the best parameters by learning from previous results.

- Best Parameters:
{
    "n_estimators": 321,
    "max_depth": 10,
    "learning_rate": 0.0104,
    "min_child_weight": 6,
    "subsample": 0.688,
    "colsample_bytree": 0.666
}
- Best Accuracy from Optuna: 81.59%

**Final XGBoost Accuracy after Bayesian Optimization: 82.68%**

## Day 2 - Feature Engineering,Ensemble learning & LightGBM
### Objective
We'll continue improving our model with feature engineering, ensemble learning, and LightGBM. 

#### Step 1: Feature Engineering
We’ll create new features from existing ones in the Titanic dataset. Let’s start by:
✅ Checking correlations between features & survival.
✅ Creating new features like:
- FamilySize = SibSp + Parch + 1
- Title Extraction from names
- Ticket Prefix
- Cabin Presence

![alt text](images/check_features.png)

**Step 1: Create New Features**
We'll create the following:
- FamilySize = SibSp + Parch + 1 (Passengers traveling together)
- Title = Extracted from Name (e.g., Mr, Miss, Master)
- IsAlone = Whether a passenger is traveling alone
- HasCabin = Whether a passenger has a cabin assigned

**Step1: Check Survival Rate by New Features**

![alt text](images/visualize_family_size.png)
![alt text](images/visualize_title.png)
![alt text](images/visualize_alone.png)
![alt text](images/visualize_cabin_availability.png)

**Step 2: Feature Selection (Choosing Best Features)**
![alt text](images/best_features.png)

**Observations**
- Fare and Age are the most important features.
- Title_Mr, Sex_male, and FamilySize also contribute significantly.
- Less common titles like Title_Lady, Title_Major, and Title_Rev have very little impact.

1. Train Initial XGBoost Model

- We trained an XGBoost classifier on the dataset.
- Extracted feature importance scores from the trained model.

2. Identified Important Features

- We sorted features based on their importance.
- Set a threshold of 0.02 to filter out less significant features.
- Selected only those features that had an importance score above the threshold.

3. Selected Features
The following features were selected based on their importance:

**['PassengerId', 'Fare', 'Age', 'Title_Mr', 'Sex_male', 'Pclass', 'FamilySize', 'Title_Mrs', 'Title_Miss', 'HasCabin', 'SibSp', 'Parch', 'Embarked_S']**

4. Trained Models on Selected Features
We retrained XGBoost and Random Forest using only the selected features.

Results:
**XGBoost Accuracy: 81.01%**
**Random Forest Accuracy: 83.8%**

#### Step 2: Ensemble Methods (combining XGBoost & Random Forest)
Now, let's move to Ensemble Learning, where we'll combine XGBoost and Random Forest to improve accuracy.

Steps:
- Train both models separately.
- Use Voting Classifier (Hard/Soft voting) to combine predictions.
- Compare accuracy with individual models.
**Ensemble Model Accuracy: 82.12 %**

Our ensemble model's accuracy is **82.12**%, which is slightly lower than our **Random Forest model (83.8%)**.

This suggests that **Random Forest alone is performing better** than the ensemble of **XGBoost + Random Forest**.

##### Weighted ensemble
Let's start with a **weighted ensemble** by assigning more weight to Random Forest, since it's performing better.

We'll combine the predictions as follows:
**Final Prediction = 0.7 × Random Forest Prediction +0.3 × XGBoost Prediction**

**Weighted Ensemble Accuracy: 82.68 %**

The weighted ensemble gave a **slight improvement over the base XGBoost model** but is still slightly **below the standalone Random Forest accuracy (83.8%).**

#### Step 3: Try with LightGBM Model

Steps for LightGBM:
1. Train a LightGBM model using the same selected features.
2. Tune hyperparameters for the best performance.
3. Compare accuracy with XGBoost, Random Forest, and Ensemble models.


## Day 3 - LightGBM,Stacking Ensemble & Random Forest Hyperparameter Tuning.

**Train a LightGBM model using the same selected features**
LightGBM is currently underperforming compared to **XGBoost (81.01%) and Random Forest(83.24%)**. Let’s try hyperparameter tuning to see if we can improve it..

**Tune LightGBM using RandomizedSearchCV**
LightGBM got worse after tuning. That suggests that either:
1. The default parameters were already optimal for this dataset.
2. The tuning search space didn't align well with the dataset.

We have two options:

1. Manually adjust parameters based on previous results
Let's try increasing n_estimators and max_depth, as LightGBM might be underfitting with the current values.

Alright, manual tuning helped but still didn’t beat the original 82.68% from default LightGBM. This confirms that:

- LightGBM default settings were already strong for this dataset.
- Hyperparameter tuning didn’t generalize well, possibly due to overfitting on the training set.

**Next Steps: Stacking Ensemble**
Since we've now tested XGBoost, Random Forest, and LightGBM, the best approach is to combine them using a stacking model.

Instead of choosing one model, we let all three vote while a **meta-model (like Logistic Regression)** learns how to combine their strengths.

Your **Stacking Model Accuracy: 81.56%** is close to your manually tuned LightGBM (81.01%), but still lower than **XGBoost (81.01%) and Random Forest(83.24%)**.

Let's tune Random Forest using RandomizedSearchCV to see if we can push the accuracy beyond 83.24%.

**Random Forest Hyperparameter Tuning**
We'll optimize:

- n_estimators: Number of trees
- max_depth: Maximum depth per tree
- min_samples_split: Min samples needed to split a node
- min_samples_leaf: Min samples in leaf nodes
- max_features: Number of features considered per split
- bootstrap: Whether to use bootstrapping

**Best Parameters**: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'log2', 'max_depth': None, 'bootstrap': True}
**Tuned Random Forest Accuracy: 82.68 %**

The accuracy dropped from **83.24% to 82.68%** after tuning. This means the default hyperparameters were performing better.

Let's tweak some of these parameters manually and re-run:
**Manually Tuned Random Forest Accuracy: 82.68 %**

This suggests that:

- The default RandomForestClassifier() was already optimal for this dataset.
- Further tuning won’t give major gains.
- The features might be limiting model performance.

**Next Steps: Feature Importance Analysis**

![alt text](images/feature_importance.png)

Observations from Feature Importance:
1. Top Important Features

- Sex_male (most influential)
- Title_Mr, Fare, PassengerId, Age, Pclass
These are strong predictors of survival.

2. Low Importance Features

- Embarked_S, Parch, SibSp, Title_Miss, HasCabin
These contribute little to the model.

**Next Steps:**
- Remove Low-Importance Features (threshold < 0.03)
- Drop Embarked_S, Parch, SibSp, HasCabin, Title_Miss
- Re-train and check accuracy.

**Update Feature Selection & Re-train**
 Your Random Forest accuracy jumped to **84.92%**!

**Ensemble Learning**

 Now, let’s combine models (Random Forest + LightGBM) using Voting Classifier to see if we can push accuracy even further.

 **Ensemble Model Accuracy: 81.56 %**


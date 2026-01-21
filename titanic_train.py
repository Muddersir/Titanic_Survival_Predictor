
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
import pickle


print("=" * 50)
print("TASK 1: DATA LOADING")
print("=" * 50)

df = pd.read_csv("Titanic-Dataset.csv")

print("First 10 rows of the dataset:")
print(df.head(10))
print(f"\nDataset Shape: {df.shape}")
print(f"\nNumber of rows: {df.shape[0]}, Number of columns: {df.shape[1]}")

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Variable Distribution (Survived):")
print(df['Survived'].value_counts())
print(f"Survival Rate: {(df['Survived'].mean()*100):.2f}%")

print("\n" + "=" * 50)
print("TASK 2: DATA PREPROCESSING")
print("=" * 50)

df_processed = df.copy()

print("\n1. Handling Missing Values:")

df_processed['Age'] = df_processed.groupby(['Pclass', 'Sex'])['Age'].transform(
    lambda x: x.fillna(x.median())
)

df_processed['Has_Cabin'] = df_processed['Cabin'].notna().astype(int)
df_processed = df_processed.drop('Cabin', axis=1)

df_processed['Embarked'] = df_processed['Embarked'].fillna(df_processed['Embarked'].mode()[0])

df_processed['Fare'] = df_processed['Fare'].fillna(df_processed['Fare'].median())

print("Missing values after imputation:")
print(df_processed.isnull().sum())

print("\n2. Feature Engineering:")

df_processed['FamilySize'] = df_processed['SibSp'] + df_processed['Parch'] + 1

df_processed['IsAlone'] = 0
df_processed.loc[df_processed['FamilySize'] == 1, 'IsAlone'] = 1

df_processed['Title'] = df_processed['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df_processed['Title'] = df_processed['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                                       'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                                       'Jonkheer', 'Dona'], 'Rare')
df_processed['Title'] = df_processed['Title'].replace('Mlle', 'Miss')
df_processed['Title'] = df_processed['Title'].replace('Ms', 'Miss')
df_processed['Title'] = df_processed['Title'].replace('Mme', 'Mrs')

df_processed['AgeGroup'] = pd.cut(df_processed['Age'], 
                                   bins=[0, 12, 18, 35, 60, 100],
                                   labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

df_processed['FareGroup'] = pd.qcut(df_processed['Fare'], 4, labels=['Low', 'Medium', 'High', 'Very High'])

df_processed = df_processed.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

print(f"New features created: FamilySize, IsAlone, Title, AgeGroup, FareGroup")
print(f"Columns after feature engineering: {df_processed.columns.tolist()}")

print("\n3. Encoding Categorical Variables:")

categorical_cols = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup']
for col in categorical_cols:
    df_processed[col] = df_processed[col].astype('category')
    print(f"{col}: {df_processed[col].nunique()} unique values")


print("\n4. Handling Outliers (using IQR method for Fare):")

Q1 = df_processed['Fare'].quantile(0.25)
Q3 = df_processed['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_processed[(df_processed['Fare'] < lower_bound) | (df_processed['Fare'] > upper_bound)]
print(f"Number of outliers in Fare: {len(outliers)}")

df_processed['Fare'] = np.where(df_processed['Fare'] > upper_bound, upper_bound, df_processed['Fare'])
df_processed['Fare'] = np.where(df_processed['Fare'] < lower_bound, lower_bound, df_processed['Fare'])

print("\n5. Preparing Features and Target:")

X = df_processed.drop('Survived', axis=1)
y = df_processed['Survived']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['category', 'object']).columns.tolist()

print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

print("\n" + "=" * 50)
print("TASK 3: PIPELINE CREATION")
print("=" * 50)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

print("Pipeline created successfully!")
print("Preprocessor includes:")
print("1. Numeric features: Imputation + Standardization")
print("2. Categorical features: Imputation + OneHot Encoding")

print("\n" + "=" * 50)
print("TASK 4: PRIMARY MODEL SELECTION")
print("=" * 50)

print("Selected Model: Random Forest Classifier")
print("\nJustification:")
print("1. Problem Type: Binary classification (Survived vs Not Survived)")
print("2. Dataset Characteristics:")
print("   - Mixed data types (numeric and categorical)")
print("   - Moderate size (891 samples)")
print("   - Potential non-linear relationships")
print("3. Model Advantages:")
print("   - Handles mixed data types well")
print("   - Robust to outliers")
print("   - Provides feature importance")
print("   - Good baseline performance")
print("   - Less prone to overfitting compared to single decision trees")
print("4. Model Limitations:")
print("   - May not capture complex interactions")
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])

print("\n" + "=" * 50)
print("TASK 5: MODEL TRAINING")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")

print("\nTraining Random Forest model...")
rf_pipeline.fit(X_train, y_train)
print("Model training completed!")

y_train_pred = rf_pipeline.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\nTraining Accuracy: {train_accuracy:.4f}")

print("\n" + "=" * 50)
print("TASK 6: CROSS-VALIDATION")
print("=" * 50)

cv_scores = cross_val_score(rf_pipeline, X_train, y_train, 
                            cv=5, scoring='accuracy', n_jobs=-1)

print("Cross-Validation Scores (Accuracy):")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.4f}")

print(f"\nMean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")
print(f"95% Confidence Interval: ({cv_scores.mean() - 2*cv_scores.std():.4f}, "
      f"{cv_scores.mean() + 2*cv_scores.std():.4f})")


print("\n" + "=" * 50)
print("TASK 7: HYPERPARAMETER TUNING")
print("=" * 50)

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 20, 30, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2']
}

print("Hyperparameters to tune:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

simple_param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5]
}

print("\nPerforming GridSearchCV...")
grid_search = GridSearchCV(
    rf_pipeline,
    simple_param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nGrid Search Results:")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

print("\nAll Results:")
results_df = pd.DataFrame(grid_search.cv_results_)
print(results_df[['params', 'mean_test_score', 'std_test_score']].sort_values('mean_test_score', ascending=False).head())

print("\n" + "=" * 50)
print("TASK 8: BEST MODEL SELECTION")
print("=" * 50)

best_model = grid_search.best_estimator_

print(f"Selected Best Model with parameters: {grid_search.best_params_}")
print(f"Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

best_model.fit(X_train, y_train)

with open('best_titanic_model.pkl', "wb") as f:
    pickle.dump(rf_pipeline, f)

print("\nBest model saved as 'best_titanic_model.pkl'")

print("\n" + "=" * 50)
print("TASK 9: MODEL PERFORMANCE EVALUATION")
print("=" * 50)

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("Test Set Performance Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, 
                     index=['Actual Not Survived', 'Actual Survived'],
                     columns=['Predicted Not Survived', 'Predicted Survived'])
print(cm_df)

cm_percent = cm / cm.sum() * 100
print("\nConfusion Matrix (Percentages):")
for i in range(2):
    for j in range(2):
        print(f"{cm_percent[i, j]:.1f}%", end='\t')
    print()

try:
    
    preprocessor = best_model.named_steps['preprocessor']
    classifier = best_model.named_steps['classifier']
    
    numeric_features_transformed = numeric_features
    categorical_features_transformed = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    all_features = list(numeric_features_transformed) + list(categorical_features_transformed)
    
    importances = classifier.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(feature_importance_df.head(10))
    
except Exception as e:
    print(f"\nCould not extract feature importances: {e}")

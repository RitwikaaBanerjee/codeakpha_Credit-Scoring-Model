import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# 1. Data Loading and Initial Exploration
# Load the dataset
data = pd.read_csv('/Users/prashant/Desktop/Credit Scoring Model/Data/bank.csv',sep=';')

# Display basic information
print("Dataset Shape:", data.shape)
print("\nColumns:", data.columns.tolist())
print("\nData Types:")
print(data.dtypes)
print("\nFirst few rows:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Check target variable distribution
print("\nTarget Variable Distribution:")
print(data['y'].value_counts())
print(data['y'].value_counts(normalize=True) * 100)

# 2. Data Preprocessing

# Define target variable
X = data.drop('y', axis=1)
y = data['y'].map({'yes': 1, 'no': 0})

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\nCategorical Features:", categorical_features)
print("Numerical Features:", numerical_features)

# Create preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 3. Feature Analysis and Selection
# Visualize correlations between numerical features
plt.figure(figsize=(12, 10))
correlation_matrix = data[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')

# Analyze relationship between 'balance' and target variable
plt.figure(figsize=(10, 6))
sns.boxplot(x='y', y='balance', data=data)
plt.title('Balance Distribution by Target')
plt.tight_layout()
plt.savefig('balance_by_target.png')

# Analyze relationship between 'age' and target variable
plt.figure(figsize=(10, 6))
sns.boxplot(x='y', y='age', data=data)
plt.title('Age Distribution by Target')
plt.tight_layout()
plt.savefig('age_by_target.png')

# Analyze categorical features
for feature in categorical_features:
    plt.figure(figsize=(12, 6))
    pd.crosstab(data[feature], data['y'], normalize='index').plot(kind='bar')
    plt.title(f'{feature} vs Target')
    plt.tight_layout()
    plt.savefig(f'{feature}_vs_target.png')

# 4. Model Training and Evaluation

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)

# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Create a function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Create pipeline with preprocessing
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # Apply SMOTE after preprocessing
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'fpr': fpr,
        'tpr': tpr,
        'pipeline': pipeline,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

# Evaluate all models
results = {}
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)
    print(f"  Accuracy: {results[name]['accuracy']:.4f}")
    print(f"  Precision: {results[name]['precision']:.4f}")
    print(f"  Recall: {results[name]['recall']:.4f}")
    print(f"  F1 Score: {results[name]['f1']:.4f}")
    print(f"  AUC: {results[name]['auc']:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, results[name]['y_pred']))
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, results[name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')

# Plot ROC curves for all models
plt.figure(figsize=(10, 8))
for name, result in results.items():
    plt.plot(result['fpr'], result['tpr'], label=f'{name} (AUC = {result["auc"]:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curves.png')

# 5. Fine-tune the best model
# Determine the best model based on AUC
best_model_name = max(results, key=lambda x: results[x]['auc'])
print(f"\nBest Model: {best_model_name} with AUC = {results[best_model_name]['auc']:.4f}")

# Define parameter grid for the best model
if best_model_name == 'Logistic Regression':
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear', 'saga']
    }
elif best_model_name == 'Random Forest':
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    }
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7]
    }
else:  # SVM
    param_grid = {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': ['scale', 'auto', 0.1, 0.01],
        'classifier__kernel': ['rbf', 'linear']
    }

# Create a pipeline for the best model
best_model = models[best_model_name]
best_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])

# Grid search with cross-validation
grid_search = GridSearchCV(
    best_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Best parameters
print("\nBest Parameters:", grid_search.best_params_)

# Evaluate the tuned model
tuned_model = grid_search.best_estimator_
y_pred_tuned = tuned_model.predict(X_test)
y_prob_tuned = tuned_model.predict_proba(X_test)[:, 1]

# Metrics
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned)
recall_tuned = recall_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned)
auc_tuned = roc_auc_score(y_test, y_prob_tuned)

print("\nTuned Model Performance:")
print(f"  Accuracy: {accuracy_tuned:.4f}")
print(f"  Precision: {precision_tuned:.4f}")
print(f"  Recall: {recall_tuned:.4f}")
print(f"  F1 Score: {f1_tuned:.4f}")
print(f"  AUC: {auc_tuned:.4f}")

# Classification report
print("\nClassification Report (Tuned Model):")
print(classification_report(y_test, y_pred_tuned))

# Confusion matrix
plt.figure(figsize=(8, 6))
cm_tuned = confusion_matrix(y_test, y_pred_tuned)
sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix - Tuned {best_model_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_tuned_model.png')

# 6. Feature Importance Analysis
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    # Get feature names after preprocessing
    preprocessor_fitted = preprocessor.fit(X_train)
    ohe_features = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = np.append(numerical_features, ohe_features)
    
    # Get feature importances
    feature_importances = tuned_model.named_steps['classifier'].feature_importances_
    
    # Sort and plot
    sorted_idx = feature_importances.argsort()[-20:]  # Top 20 features
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx])
    plt.yticks(range(len(sorted_idx)), feature_names[sorted_idx])
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    # Print top 10 features
    print("\nTop 10 Important Features:")
    for i in sorted_idx[-10:]:
        print(f"  {feature_names[i]}: {feature_importances[i]:.4f}")

# 7. Save the final model
import joblib
joblib.dump(tuned_model, 'credit_scoring_model.pkl')
print("\nFinal model saved as 'credit_scoring_model.pkl'")

# 8. Model Interpretation and Insights
# Define a function to explain model predictions
def explain_prediction(model, data_row):
    """
    Explain the prediction for a single data point
    """
    # Convert the data_row to a DataFrame
    data_row_df = pd.DataFrame([data_row], columns=X.columns)
    
    # Get the prediction probability
    prediction_prob = model.predict_proba(data_row_df)[0, 1]
    prediction = 'creditworthy' if prediction_prob >= 0.5 else 'not creditworthy'
    
    print(f"\nPrediction: {prediction} (Probability: {prediction_prob:.4f})")
    
    # We can provide more detailed explanation based on model type
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        # Get feature values
        feature_dict = {}
        for i, feature in enumerate(X.columns):
            feature_dict[feature] = data_row[i]
        
        print("\nKey factors affecting this prediction:")
        # Here we would do a more detailed analysis, but for now just print some features
        print("  - Age:", data_row[0])
        print("  - Job:", data_row[1])
        print("  - Balance:", data_row[5])
        print("  - Housing loan:", data_row[6])
        print("  - Personal loan:", data_row[7])
    
    return prediction, prediction_prob

# Example: Explain a prediction for the first test instance
example_row = X_test.iloc[0]
explanation = explain_prediction(tuned_model, example_row)
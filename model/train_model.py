import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import os

# Create necessary directories
os.makedirs('model', exist_ok=True)
os.makedirs('static/img', exist_ok=True)  # Create the static/img directory

# Load the dataset
df = pd.read_csv('github_quality.csv')

# Display basic information
print("Dataset Information:")
print(f"Shape: {df.shape}")
print("\nColumns:")
print(df.columns.tolist())
print("\nSample data:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Data preprocessing
# Assuming 'quality' is the target variable
if 'quality' in df.columns:
    X = df.drop(['quality', 'quality_score', 'quality_numeric'], axis=1)
    y = df['quality']
else:
    # If 'quality' column doesn't exist, we'll create one based on some metrics
    print("Creating quality metric based on available features...")
    
    # Example: Create a quality score based on repositories, followers, and contributions
    if 'repositories' in df.columns and 'followers' in df.columns and 'contributions' in df.columns:
        df['quality_score'] = (
            df['repositories'] * 0.3 + 
            df['followers'] * 0.4 + 
            df['contributions'] * 0.3
        )
        
        # Convert to categorical: 'low', 'medium', 'high'
        df['quality'] = pd.qcut(
            df['quality_score'], 
            q=3, 
            labels=['low', 'medium', 'high']
        )
        
        X = df.drop(['quality', 'quality_score'], axis=1)
        y = df['quality']
    else:
        # Adjust this based on your actual columns
        print("Could not determine target variable. Please check your dataset.")
        exit(1)

# Handle categorical variables if any
X = pd.get_dummies(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': pipeline.named_steps['classifier'].feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    
    # Ensure the directory exists before saving
    os.makedirs('static/img', exist_ok=True)
    plt.savefig('static/img/feature_importance.png')

# Save the model
joblib.dump(pipeline, 'model/model.pkl')
print("\nModel saved to 'model/model.pkl'")

# Save feature names for later use
with open('model/feature_names.txt', 'w') as f:
    for feature in X.columns:
        f.write(f"{feature}\n")

print("Feature names saved to 'model/feature_names.txt'") 
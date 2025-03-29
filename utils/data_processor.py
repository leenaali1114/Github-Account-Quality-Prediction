import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_and_preprocess_data(file_path):
    """Load and preprocess the GitHub quality dataset."""
    df = pd.read_csv(file_path)
    return df

def preprocess_input(input_data, feature_names):
    """Preprocess user input for prediction."""
    # Convert input data to DataFrame with correct feature names
    input_df = pd.DataFrame([input_data])
    
    # Create a DataFrame with all features from the model
    full_input = pd.DataFrame(columns=feature_names)
    
    # Fill in the values we have
    for col in input_df.columns:
        if col in feature_names:
            full_input[col] = input_df[col]
    
    # Fill missing values with 0
    full_input = full_input.fillna(0)
    
    return full_input

def get_feature_names():
    """Get feature names used by the model."""
    feature_names = []
    try:
        with open('model/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Feature names file not found. Please train the model first.")
    
    return feature_names

def get_input_fields():
    """Return the list of input fields for the prediction form with descriptions and recommended ranges."""
    return [
        {
            'name': 'repositories', 
            'label': 'Number of Repositories', 
            'type': 'number', 
            'min': 0,
            'recommended_max': 100,
            'description': 'The total number of public repositories you own on GitHub.'
        },
        {
            'name': 'stars', 
            'label': 'Total Stars', 
            'type': 'number', 
            'min': 0,
            'recommended_max': 1000,
            'description': 'The total number of stars your repositories have received.'
        },
        {
            'name': 'followers', 
            'label': 'Number of Followers', 
            'type': 'number', 
            'min': 0,
            'recommended_max': 500,
            'description': 'The number of GitHub users following your account.'
        },
        {
            'name': 'following', 
            'label': 'Number of Following', 
            'type': 'number', 
            'min': 0,
            'recommended_max': 200,
            'description': 'The number of GitHub users you are following.'
        },
        {
            'name': 'contributions', 
            'label': 'Number of Contributions', 
            'type': 'number', 
            'min': 0,
            'recommended_max': 2000,
            'description': 'Total contributions in the last year (commits, PRs, issues, reviews).'
        },
        {
            'name': 'commits', 
            'label': 'Number of Commits', 
            'type': 'number', 
            'min': 0,
            'recommended_max': 5000,
            'description': 'The total number of commits you have made across all repositories.'
        },
        {
            'name': 'issues', 
            'label': 'Number of Issues', 
            'type': 'number', 
            'min': 0,
            'recommended_max': 100,
            'description': 'The number of issues you have opened across all repositories.'
        },
        {
            'name': 'pull_requests', 
            'label': 'Number of Pull Requests', 
            'type': 'number', 
            'min': 0,
            'recommended_max': 200,
            'description': 'The number of pull requests you have submitted across all repositories.'
        },
        {
            'name': 'account_age_days', 
            'label': 'Account Age (days)', 
            'type': 'number', 
            'min': 1,
            'recommended_max': 3650,
            'description': 'The age of your GitHub account in days (approximately).'
        },
        {
            'name': 'has_readme', 
            'label': 'Has README files', 
            'type': 'checkbox',
            'description': 'Check if most of your repositories have README files.'
        },
        {
            'name': 'has_profile_readme', 
            'label': 'Has Profile README', 
            'type': 'checkbox',
            'description': 'Check if you have set up a profile README (special repository with your username).'
        },
    ] 
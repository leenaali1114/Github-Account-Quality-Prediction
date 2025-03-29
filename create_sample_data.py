import pandas as pd
import numpy as np

# Create a sample dataset
np.random.seed(42)
n_samples = 1000

data = {
    'repositories': np.random.randint(1, 100, n_samples),
    'stars': np.random.randint(0, 1000, n_samples),
    'followers': np.random.randint(0, 500, n_samples),
    'following': np.random.randint(0, 200, n_samples),
    'contributions': np.random.randint(10, 2000, n_samples),
    'commits': np.random.randint(10, 5000, n_samples),
    'issues': np.random.randint(0, 100, n_samples),
    'pull_requests': np.random.randint(0, 200, n_samples),
    'account_age_days': np.random.randint(30, 3650, n_samples),
    'has_readme': np.random.randint(0, 2, n_samples),
    'has_profile_readme': np.random.randint(0, 2, n_samples),
}

df = pd.DataFrame(data)

# Create a comprehensive quality score based on multiple GitHub metrics
# This formula weights different aspects of GitHub activity:
# - Content creation (repositories, commits)
# - Community engagement (stars, followers, pull requests)
# - Documentation quality (readme files)
# - Account maturity (age, contributions)

df['quality_score'] = (
    # Content creation (40%)
    (df['repositories'] * 0.25 + 
     df['commits'] / 50 * 0.15) +
    
    # Community engagement (30%)
    (df['stars'] / 10 * 0.1 + 
     df['followers'] * 0.1 + 
     df['pull_requests'] / 10 * 0.1) +
    
    # Documentation (10%)
    ((df['has_readme'] + df['has_profile_readme']) * 0.05) +
    
    # Account maturity (20%)
    (df['account_age_days'] / 365 * 0.1 + 
     df['contributions'] / 100 * 0.1)
)

# Normalize the score to a 0-100 scale
min_score = df['quality_score'].min()
max_score = df['quality_score'].max()
df['quality_score'] = 100 * (df['quality_score'] - min_score) / (max_score - min_score)

# Convert to categorical: 'low', 'medium', 'high'
df['quality'] = pd.qcut(
    df['quality_score'], 
    q=3, 
    labels=['low', 'medium', 'high']
)

# Add a numerical quality column (1=low, 2=medium, 3=high) for easier analysis
quality_map = {'low': 1, 'medium': 2, 'high': 3}
df['quality_numeric'] = df['quality'].map(quality_map)

# Print some statistics
print("\nDataset Statistics:")
print(f"Total samples: {n_samples}")
print(f"Quality distribution: {df['quality'].value_counts().to_dict()}")
print(f"Average quality score: {df['quality_score'].mean():.2f}")
print("\nAverage metrics by quality level:")
print(df.groupby('quality')[['repositories', 'stars', 'followers', 'contributions']].mean())

# Save to CSV
df.to_csv('github_quality.csv', index=False)
print("\nSample dataset created: github_quality.csv") 
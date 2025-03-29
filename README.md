# GitHub Account Quality Predictor

A machine learning application that analyzes GitHub profile metrics to predict account quality and provide personalized recommendations for improvement.

![GitHub Account Quality Predictor](static/img/feature_importance.png)

## Features

- **Quality Prediction**: Uses machine learning to classify GitHub accounts as low, medium, or high quality
- **Personalized Recommendations**: Provides tailored suggestions to improve your GitHub profile
- **Interactive UI**: User-friendly interface with visualizations of your metrics
- **API Access**: REST API endpoint for programmatic access to predictions

## Technology Stack

- **Backend**: Flask (Python)
- **Machine Learning**: scikit-learn, pandas, numpy
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Data Visualization**: Chart.js
- **AI Recommendations**: Groq API

## Project Structure

github-quality-predictor/
├── app.py                  # Main Flask application
├── model/
│   ├── train_model.py      # Script to train the ML model
│   └── model.pkl           # Saved ML model (generated)
├── static/
│   ├── css/
│   │   └── style.css       # Custom CSS
│   ├── js/
│   │   └── script.js       # Custom JavaScript
│   └── img/                # Images for the UI (generated)
├── templates/
│   ├── index.html          # Landing page
│   ├── predict.html        # Prediction form
│   ├── result.html         # Results and recommendations
│   └── error.html          # Error page
├── utils/
│   ├── data_processor.py   # Data preprocessing utilities
│   └── recommender.py      # Recommendation system using Groq
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables
├── .env.template           # Template for environment variables
├── create_sample_data.py   # Script to generate sample dataset
└── github_quality.csv      # Dataset

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/github-quality-predictor.git
   cd github-quality-predictor
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # Copy the environment template
   cp .env.template .env
   # Edit .env and add your Groq API key
   ```

## Usage

### Generate Sample Data

If you don't have a dataset, generate a sample one:
```bash
python create_sample_data.py
```

### Train the Model

Train the machine learning model:
```bash
python model/train_model.py
```

### Run the Application

Start the Flask application:
```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000/`

### API Usage

You can also use the API endpoint to get predictions programmatically:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "repositories": 50,
    "stars": 200,
    "followers": 100,
    "following": 50,
    "contributions": 500,
    "commits": 1000,
    "issues": 20,
    "pull_requests": 30,
    "account_age_days": 365,
    "has_readme": 1,
    "has_profile_readme": 1
  }'
```

## How It Works

1. **Data Collection**: The application uses GitHub metrics like repositories, stars, followers, contributions, etc.

2. **Machine Learning Model**: A Random Forest classifier is trained on the dataset to predict account quality (low, medium, high).

3. **Feature Importance**: The model identifies which metrics have the most impact on account quality.

4. **Personalized Recommendations**: Based on the prediction and input metrics, the application generates tailored recommendations using the Groq API.

5. **Visualization**: The results are presented with interactive visualizations to help users understand their GitHub profile strengths and weaknesses.

## Quality Metric Calculation

The quality score is calculated based on four key aspects of GitHub activity:

1. **Content Creation (40% weight)**
   - Number of repositories (25%)
   - Number of commits (15%)

2. **Community Engagement (30% weight)**
   - Stars received (10%)
   - Number of followers (10%)
   - Pull requests created (10%)

3. **Documentation Quality (10% weight)**
   - Presence of README files and profile README (10%)

4. **Account Maturity (20% weight)**
   - Account age in days (10%)
   - Number of contributions (10%)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
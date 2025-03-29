from flask import Flask, render_template, request, jsonify, flash
import pandas as pd
import numpy as np
import joblib
import os
import json
from dotenv import load_dotenv
from utils.data_processor import preprocess_input, get_feature_names, get_input_fields
from utils.recommender import GitHubRecommender

# Load environment variables from .env file
load_dotenv()

# Create directories if they don't exist
os.makedirs('model', exist_ok=True)
os.makedirs('static/img', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('utils', exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'default-secret-key')

# Load the model
model = None
try:
    model = joblib.load('model/model.pkl')
    print("Model loaded successfully!")
except:
    print("Model not found. Please train the model first by running 'python model/train_model.py'")

# Initialize the recommender
recommender = None
try:
    recommender = GitHubRecommender()
    print("Recommender initialized successfully!")
except ValueError as e:
    print(f"Warning: {e}")
    print("Recommendation system will not be available.")

@app.route('/')
def home():
    """Render the home page."""
    if model is None:
        flash("Model not loaded. Please train the model first by running 'python model/train_model.py'", "warning")
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle prediction requests."""
    if model is None:
        return render_template('error.html', message="Model not loaded. Please train the model first by running 'python model/train_model.py'")
    
    if request.method == 'POST':
        # Get user input
        input_data = {}
        for field in get_input_fields():
            name = field['name']
            if field['type'] == 'checkbox':
                input_data[name] = 1 if name in request.form else 0
            else:
                input_data[name] = float(request.form.get(name, 0))
        
        # Preprocess input
        feature_names = get_feature_names()
        processed_input = preprocess_input(input_data, feature_names)
        
        # Make prediction
        quality = model.predict(processed_input)[0]
        probability = np.max(model.predict_proba(processed_input)[0])
        
        # Get recommendations
        recommendations = None
        if recommender is not None:
            recommendations = recommender.get_recommendations(quality, input_data)
        
        return render_template(
            'result.html',
            quality=quality,
            probability=probability,
            metrics=input_data,
            recommendations=recommendations
        )
    
    # GET request - show the prediction form
    return render_template('predict.html', fields=get_input_fields())

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 500
    
    # Get JSON input
    input_data = request.json
    
    # Preprocess input
    feature_names = get_feature_names()
    processed_input = preprocess_input(input_data, feature_names)
    
    # Make prediction
    quality = model.predict(processed_input)[0]
    probability = float(np.max(model.predict_proba(processed_input)[0]))
    
    # Get recommendations
    recommendations = None
    if recommender is not None:
        recommendations = recommender.get_recommendations(quality, input_data)
    
    return jsonify({
        "quality": quality,
        "probability": probability,
        "recommendations": recommendations
    })

@app.route('/train', methods=['GET'])
def train_model():
    """Route to manually trigger model training."""
    try:
        import subprocess
        result = subprocess.run(['python', 'model/train_model.py'], capture_output=True, text=True)
        
        if result.returncode == 0:
            # Reload the model
            global model
            model = joblib.load('model/model.pkl')
            return render_template('error.html', message="Model trained successfully! You can now use the prediction feature.")
        else:
            return render_template('error.html', message=f"Error training model: {result.stderr}")
    except Exception as e:
        return render_template('error.html', message=f"Error training model: {str(e)}")

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', message="Server error. Please try again later."), 500

if __name__ == '__main__':
    app.run(debug=True) 
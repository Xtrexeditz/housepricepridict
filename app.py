import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Global variable to store the model pipeline
model_pipeline = None

def load_model():
    global model_pipeline
    model_path = 'models/house_price_pipeline.joblib'
    if os.path.exists(model_path):
        model_pipeline = joblib.load(model_path)
        print("Model pipeline loaded successfully.")
    else:
        print(f"Warning: Model pipeline not found at {model_path}. Running predictions will fail until model is trained.")

def format_indian_currency(amount):
    """Formats a number into Indian numbering format (e.g. ₹1,68,25,000)"""
    amount = round(amount)
    s = str(amount)
    if len(s) <= 3:
        return f"₹{s}"
    last_three = s[-3:]
    remaining = s[:-3]
    groups = []
    while remaining:
        groups.append(remaining[-2:])
        remaining = remaining[:-2]
    groups.reverse()
    return f"₹{','.join(groups)},{last_three}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model_pipeline
    if model_pipeline is None:
        load_model()
        if model_pipeline is None:
            return jsonify({'error': 'Prediction model is not trained/available yet. Please run training first.'}), 500
            
    try:
        data = request.get_json()
        
        # Extract features and convert to appropriate types
        square_feet = float(data.get('SquareFeet'))
        property_type = str(data.get('PropertyType'))
        bedrooms = int(data.get('Bedrooms'))
        bathrooms = float(data.get('Bathrooms'))
        neighborhood = str(data.get('Neighborhood'))
        year_built = int(data.get('YearBuilt'))
        floors = int(data.get('Floors'))
        garage = int(data.get('Garage'))
        
        # Verify inputs are reasonable
        if square_feet <= 0 or bedrooms <= 0 or bathrooms <= 0 or year_built <= 0 or floors <= 0:
            return jsonify({'error': 'Invalid house features values. Must be positive numbers.'}), 400
            
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'SquareFeet': square_feet,
            'PropertyType': property_type,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'Neighborhood': neighborhood,
            'YearBuilt': year_built,
            'Floors': floors,
            'Garage': garage
        }])
        
        # Run prediction
        prediction = model_pipeline.predict(input_data)[0]
        
        # Format predicted price in Indian Rupee format
        formatted = format_indian_currency(prediction)
        
        # Return predicted price
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'formatted_prediction': formatted
        })
        
    except (ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid input format or missing fields: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)

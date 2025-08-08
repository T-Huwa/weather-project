from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import requests
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to your model file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'multioutput3_linear_model.joblib')

# OpenWeather API configuration
OPENWEATHER_API_KEY = "b51cb434e01487ae9e1803a8b9ef73d5"  # Replace with your API key
KASUNGU_COORDS = {"lat": -13.028085670616038, "lon": 33.464763622195804}

# Seasonal weather patterns for Kasungu District (fallback if model doesn't vary)
KASUNGU_SEASONAL_PATTERNS = {
    1: {"tmin": 18.5, "tmax": 28.2, "rainfall": 220.0, "humidity": 85.0, "wind_speed": 3.2},  # January - Peak rainy
    2: {"tmin": 18.8, "tmax": 28.5, "rainfall": 180.0, "humidity": 82.0, "wind_speed": 3.0},  # February - Late rainy
    3: {"tmin": 18.2, "tmax": 28.8, "rainfall": 140.0, "humidity": 78.0, "wind_speed": 2.8},  # March - End rainy
    4: {"tmin": 16.5, "tmax": 27.5, "rainfall": 45.0, "humidity": 70.0, "wind_speed": 2.5},   # April - Early dry
    5: {"tmin": 13.8, "tmax": 25.2, "rainfall": 8.0, "humidity": 60.0, "wind_speed": 2.2},    # May - Cool dry
    6: {"tmin": 11.2, "tmax": 23.8, "rainfall": 2.0, "humidity": 55.0, "wind_speed": 2.0},    # June - Cold dry
    7: {"tmin": 10.8, "tmax": 24.2, "rainfall": 1.0, "humidity": 52.0, "wind_speed": 2.1},    # July - Coldest
    8: {"tmin": 12.5, "tmax": 26.8, "rainfall": 3.0, "humidity": 55.0, "wind_speed": 2.3},    # August - Warming
    9: {"tmin": 15.8, "tmax": 29.2, "rainfall": 12.0, "humidity": 58.0, "wind_speed": 2.6},   # September - Hot dry
    10: {"tmin": 18.2, "tmax": 31.5, "rainfall": 35.0, "humidity": 65.0, "wind_speed": 2.9},  # October - Pre-rains
    11: {"tmin": 19.5, "tmax": 30.8, "rainfall": 95.0, "humidity": 75.0, "wind_speed": 3.1},  # November - Early rains
    12: {"tmin": 19.2, "tmax": 29.5, "rainfall": 165.0, "humidity": 80.0, "wind_speed": 3.2}, # December - Rains
}

# Load the model globally when the app starts
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
    logger.info(f"Model type: {type(model).__name__}")
    if hasattr(model, 'n_features_in_'):
        logger.info(f"Model expects {model.n_features_in_} features")
    if hasattr(model, 'n_outputs_'):
        logger.info(f"Model outputs {model.n_outputs_} values")
except FileNotFoundError:
    logger.error(f"Error: Model file not found at {MODEL_PATH}")
    model = None
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

def check_model_variation():
    """Check if the model produces different outputs for different months"""
    if model is None:
        return False
    
    try:
        # Test with a few different months
        test_months = [1, 6, 12]  # January, June, December
        predictions = []
        
        for month in test_months:
            input_data = np.array([[month, 2024]], dtype=np.float64)
            pred = model.predict(input_data)[0]
            predictions.append(pred)
        
        # Check if predictions are significantly different
        first_pred = predictions[0]
        for pred in predictions[1:]:
            if not np.allclose(first_pred, pred, rtol=0.01):  # 1% tolerance
                return True
        
        return False
    except Exception as e:
        logger.error(f"Error checking model variation: {e}")
        return False

# Check model variation at startup
MODEL_HAS_VARIATION = check_model_variation()
if not MODEL_HAS_VARIATION:
    logger.warning("Model does not show significant variation between months. Using seasonal patterns as fallback.")

@app.route('/')
def home():
    """Home endpoint with API information"""
    return jsonify({
        "message": "Kasungu Agricultural Intelligence System API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Check API health",
            "/predict": "Make weather predictions (POST) - expects year and month",
            "/test-model": "Test model functionality (GET)",
            "/current-weather": "Get current weather from OpenWeather API",
            "/forecast": "Get 5-day weather forecast from OpenWeather API",
            "/api/forecast": "Get 7-day ML model forecast (POST)",
            "/model-info": "Get ML model information"
        },
        "status": "running",
        "model_loaded": model is not None,
        "model_has_variation": MODEL_HAS_VARIATION,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if model is not None:
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "model_has_variation": MODEL_HAS_VARIATION,
            "message": "ML model is ready for predictions",
            "model_type": type(model).__name__,
            "features_expected": getattr(model, 'n_features_in_', 'Unknown'),
            "outputs": getattr(model, 'n_outputs_', 'Unknown')
        }), 200
    else:
        return jsonify({
            "status": "unhealthy",
            "model_loaded": False,
            "message": "ML model failed to load"
        }), 500

def get_enhanced_prediction(month, year):
    """
    Get prediction with seasonal enhancement if model doesn't vary
    """
    try:
        # First try the model
        if model is not None:
            input_data = np.array([[month, year]], dtype=np.float64)
            model_pred = model.predict(input_data)[0]
            
            logger.info(f"Model prediction for month {month}: {model_pred}")
            
            # If model has variation, use it directly
            if MODEL_HAS_VARIATION:
                if len(model_pred) >= 5:
                    return {
                        "tmin": float(model_pred[0]),
                        "tmax": float(model_pred[1]),
                        "rainfall": float(model_pred[2]),
                        "wind_speed": float(model_pred[3]),
                        "humidity": float(model_pred[4])
                    }
                else:
                    # Handle fewer outputs
                    result = {
                        "tmin": float(model_pred[0]) if len(model_pred) > 0 else 20.0,
                        "tmax": float(model_pred[1]) if len(model_pred) > 1 else 30.0,
                        "rainfall": float(model_pred[2]) if len(model_pred) > 2 else 50.0,
                        "wind_speed": float(model_pred[3]) if len(model_pred) > 3 else 5.0,
                        "humidity": float(model_pred[4]) if len(model_pred) > 4 else 65.0
                    }
                    return result
            
            # If model doesn't vary, blend with seasonal patterns
            else:
                seasonal = KASUNGU_SEASONAL_PATTERNS.get(month, KASUNGU_SEASONAL_PATTERNS[6])
                
                # Use 70% seasonal pattern + 30% model prediction for more realistic variation
                if len(model_pred) >= 5:
                    blended = {
                        "tmin": 0.7 * seasonal["tmin"] + 0.3 * float(model_pred[0]),
                        "tmax": 0.7 * seasonal["tmax"] + 0.3 * float(model_pred[1]),
                        "rainfall": 0.7 * seasonal["rainfall"] + 0.3 * float(model_pred[2]),
                        "wind_speed": 0.7 * seasonal["wind_speed"] + 0.3 * float(model_pred[3]),
                        "humidity": 0.7 * seasonal["humidity"] + 0.3 * float(model_pred[4])
                    }
                else:
                    # Fallback blending
                    model_tmin = float(model_pred[0]) if len(model_pred) > 0 else seasonal["tmin"]
                    model_tmax = float(model_pred[1]) if len(model_pred) > 1 else seasonal["tmax"]
                    model_rainfall = float(model_pred[2]) if len(model_pred) > 2 else seasonal["rainfall"]
                    
                    blended = {
                        "tmin": 0.7 * seasonal["tmin"] + 0.3 * model_tmin,
                        "tmax": 0.7 * seasonal["tmax"] + 0.3 * model_tmax,
                        "rainfall": 0.7 * seasonal["rainfall"] + 0.3 * model_rainfall,
                        "wind_speed": seasonal["wind_speed"],
                        "humidity": seasonal["humidity"]
                    }
                
                logger.info(f"Blended prediction for month {month}: {blended}")
                return blended
        
        # Fallback to seasonal patterns if model is not available
        seasonal = KASUNGU_SEASONAL_PATTERNS.get(month, KASUNGU_SEASONAL_PATTERNS[6])
        logger.info(f"Using seasonal pattern for month {month}: {seasonal}")
        return seasonal
        
    except Exception as e:
        logger.error(f"Error in enhanced prediction: {e}")
        # Final fallback
        seasonal = KASUNGU_SEASONAL_PATTERNS.get(month, KASUNGU_SEASONAL_PATTERNS[6])
        return seasonal

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint for the ML model with seasonal enhancement.
    Expects a JSON payload with 'year' and 'month'.
    """
    try:
        data = request.get_json(force=True)
        logger.info(f"Received prediction request: {data}")

        # Validate required fields
        if 'year' not in data or 'month' not in data:
            return jsonify({"error": "Missing 'year' or 'month' in request data"}), 400

        year = int(data['year'])
        month = int(data['month'])

        # Validate ranges
        if month < 1 or month > 12:
            return jsonify({"error": "Month must be between 1 and 12"}), 400
        
        if year < 1900 or year > 2100:
            return jsonify({"error": "Year must be between 1900 and 2100"}), 400

        # Get enhanced prediction
        prediction = get_enhanced_prediction(month, year)
        
        # Apply reasonable bounds
        prediction["tmin"] = max(-10, min(40, prediction["tmin"]))
        prediction["tmax"] = max(prediction["tmin"], min(50, prediction["tmax"]))
        prediction["rainfall"] = max(0, min(500, prediction["rainfall"]))
        prediction["wind_speed"] = max(0, min(50, prediction["wind_speed"]))
        prediction["humidity"] = max(0, min(100, prediction["humidity"]))

        # Format response - SINGLE FORMAT ONLY
        response_data = {
            "tmin": round(prediction["tmin"], 2),
            "tmax": round(prediction["tmax"], 2),
            "rainfall": round(prediction["rainfall"], 2),
            "wind_speed": round(prediction["wind_speed"], 2),
            "humidity": round(prediction["humidity"], 2)
        }

        logger.info(f"Final prediction result: {response_data}")
        return jsonify(response_data)

    except ValueError as e:
        logger.error(f"Value error in prediction: {e}")
        return jsonify({"error": f"Invalid input values: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

@app.route('/test-model', methods=['GET'])
def test_model():
    """Test endpoint to check model functionality with different months"""
    try:
        # Test with different months to verify variation
        test_cases = [
            {"year": 2024, "month": 1},   # January (rainy season)
            {"year": 2024, "month": 6},   # June (dry season)
            {"year": 2024, "month": 12},  # December (planting season)
        ]
        
        results = []
        for test_case in test_cases:
            if model is not None:
                input_features = [test_case["month"], test_case["year"]]
                input_array = np.array(input_features).reshape(1, -1)
                model_prediction = model.predict(input_array)
            else:
                model_prediction = None
            
            enhanced_prediction = get_enhanced_prediction(test_case["month"], test_case["year"])
            
            results.append({
                "input": test_case,
                "model_prediction": model_prediction.tolist() if model_prediction is not None else None,
                "enhanced_prediction": enhanced_prediction,
                "month_name": datetime(test_case["year"], test_case["month"], 1).strftime("%B")
            })
        
        return jsonify({
            "message": "Model test completed",
            "model_loaded": model is not None,
            "model_has_variation": MODEL_HAS_VARIATION,
            "test_results": results,
            "model_info": {
                "type": type(model).__name__ if model else "Not loaded",
                "features_expected": getattr(model, 'n_features_in_', 'Unknown'),
                "outputs": getattr(model, 'n_outputs_', 'Unknown')
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Model test error: {e}")
        return jsonify({
            "error": f"Model test failed: {str(e)}",
            "model_loaded": model is not None
        }), 500

def prepare_features_for_forecast(date_obj):
    """
    Prepare features for model prediction based on date.
    """
    month = date_obj.month
    year = date_obj.year
    return month, year

@app.route('/api/forecast', methods=['POST'])
def get_ml_forecast():
    """
    Get 7-day weather forecast using the enhanced prediction system.
    """
    try:
        data = request.get_json()
        start_date_str = data.get('start_date', datetime.now().strftime('%Y-%m-%d'))
        
        forecast = []
        base_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        
        for i in range(7):
            forecast_date = base_date + timedelta(days=i)
            month, year = prepare_features_for_forecast(forecast_date)
            
            # Get enhanced prediction
            prediction = get_enhanced_prediction(month, year)
            
            # Simple condition mapping based on rainfall
            condition = 'sunny'
            if prediction["rainfall"] > 15:
                condition = 'rainy'
            elif prediction["rainfall"] > 5:
                condition = 'cloudy'
            
            forecast.append({
                'date': forecast_date.strftime('%Y-%m-%d'),
                'day_name': forecast_date.strftime('%A'),
                'temperature_max': round(prediction["tmax"], 1),
                'temperature_min': round(prediction["tmin"], 1),
                'rainfall': round(prediction["rainfall"], 1),
                'humidity': round(prediction["humidity"], 1),
                'wind_speed': round(prediction["wind_speed"], 1),
                'condition': condition
            })
        
        return jsonify({'forecast': forecast})
        
    except Exception as e:
        logger.error(f"ML Forecast error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/current-weather', methods=['GET'])
def get_current_weather():
    """Get current weather from OpenWeather API"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": KASUNGU_COORDS["lat"],
            "lon": KASUNGU_COORDS["lon"],
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        result = {
            "location": f"{data['name']}, {data['sys']['country']}",
            "temperature": data['main']['temp'],
            "feels_like": data['main']['feels_like'],
            "humidity": data['main']['humidity'],
            "pressure": data['main']['pressure'],
            "wind_speed": data['wind']['speed'],
            "wind_direction": data['wind'].get('deg', 0),
            "visibility": data.get('visibility', 0),
            "weather": {
                "main": data['weather'][0]['main'],
                "description": data['weather'][0]['description'],
                "icon": data['weather'][0]['icon']
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(result), 200
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching current weather: {e}")
        return jsonify({"error": "Failed to fetch current weather"}), 500
    except Exception as e:
        logger.error(f"Unexpected error in current weather: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/forecast', methods=['GET'])
def get_weather_forecast():
    """Get 5-day weather forecast from OpenWeather API"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "lat": KASUNGU_COORDS["lat"],
            "lon": KASUNGU_COORDS["lon"],
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        forecasts = []
        for item in data['list']:
            forecast = {
                "datetime": item['dt_txt'],
                "temperature": item['main']['temp'],
                "feels_like": item['main']['feels_like'],
                "humidity": item['main']['humidity'],
                "pressure": item['main']['pressure'],
                "wind_speed": item['wind']['speed'],
                "wind_direction": item['wind'].get('deg', 0),
                "weather": {
                    "main": item['weather'][0]['main'],
                    "description": item['weather'][0]['description'],
                    "icon": item['weather'][0]['icon']
                },
                "precipitation": item.get('rain', {}).get('3h', 0) + item.get('snow', {}).get('3h', 0)
            }
            forecasts.append(forecast)
        
        result = {
            "location": f"{data['city']['name']}, {data['city']['country']}",
            "forecasts": forecasts,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(result), 200
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching weather forecast: {e}")
        return jsonify({"error": "Failed to fetch weather forecast"}), 500
    except Exception as e:
        logger.error(f"Unexpected error in weather forecast: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get information about the loaded ML model"""
    try:
        model_info = {
            "model_type": type(model).__name__ if model else "Not loaded",
            "model_loaded": model is not None,
            "model_has_variation": MODEL_HAS_VARIATION,
            "features_expected": getattr(model, 'n_features_in_', 'Unknown'),
            "outputs": getattr(model, 'n_outputs_', 'Unknown'),
            "model_path": MODEL_PATH,
            "enhancement_mode": "Seasonal blending" if not MODEL_HAS_VARIATION else "Model only",
            "timestamp": datetime.now().isoformat()
        }
        
        if model and hasattr(model, 'feature_names_in_'):
            model_info["feature_names"] = list(model.feature_names_in_)
        
        return jsonify(model_info), 200
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({"error": "Failed to get model information"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Kasungu Agricultural Intelligence System API")
    print("=" * 60)
    print(f"Model loaded: {'Yes' if model is not None else 'No'}")
    if model is not None:
        print(f"Model type: {type(model).__name__}")
        if hasattr(model, 'n_features_in_'):
            print(f"Expected features: {model.n_features_in_}")
        if hasattr(model, 'n_outputs_'):
            print(f"Model outputs: {model.n_outputs_}")
    print(f"Model has variation: {'Yes' if MODEL_HAS_VARIATION else 'No'}")
    print(f"Enhancement mode: {'Model only' if MODEL_HAS_VARIATION else 'Seasonal blending'}")
    print(f"OpenWeather API configured: {'Yes' if OPENWEATHER_API_KEY else 'No'}")
    print("Available endpoints:")
    print("  GET  / - API information")
    print("  GET  /health - Health check")
    print("  POST /predict - Weather predictions (month, year)")
    print("  GET  /test-model - Test model with different months")
    print("  GET  /current-weather - Current weather from OpenWeather")
    print("  GET  /forecast - 5-day forecast from OpenWeather")
    print("  POST /api/forecast - 7-day ML model forecast")
    print("  GET  /model-info - Model information")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

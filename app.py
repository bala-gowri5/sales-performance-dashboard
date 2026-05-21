from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
import json
from datetime import datetime
import numpy as np
import traceback

# Helper function to convert numpy types to Python native types
def convert_to_native_type(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load the trained model
MODEL_PATH = 'Models/sales_prediction_model.pkl'
model_artifacts = None

def load_model():
    global model_artifacts
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_artifacts = pickle.load(f)
        print("✅ Model loaded successfully!")
        print(f"📊 Features expected: {model_artifacts['feature_cols']}")
        print(f"🏷️  Encoders available: {list(model_artifacts['label_encoders'].keys())}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        return False

# Initialize model on startup
load_model()

@app.route('/')
def index():
    """Main landing page - Project Overview"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Analytics Dashboard"""
    return render_template('dashboard.html')

@app.route('/predict')
def predict_page():
    """Sales Prediction Page"""
    if model_artifacts:
        return render_template('predict.html', 
                             feature_cols=model_artifacts['feature_cols'],
                             metrics=model_artifacts['model_metrics'])
    else:
        return render_template('predict.html', error="Model not loaded")

@app.route('/insights')
def insights():
    """Business Insights Page"""
    return render_template('insights.html')

@app.route('/about')
def about():
    """About the Project"""
    return render_template('about.html')

@app.route('/api/load_data', methods=['POST'])
def load_data():
    """API endpoint to load and process CSV data"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'Only CSV files are allowed'}), 400
        
        # Save file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_data.csv')
        file.save(filepath)
        
        # Read and process data
        df = pd.read_csv(filepath, encoding='latin1')
        
        # Basic statistics
        stats = {
            'total_records': len(df),
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }
        
        # If Sales column exists, calculate sales stats
        if 'Sales' in df.columns:
            stats['total_sales'] = float(df['Sales'].sum())
            stats['average_sales'] = float(df['Sales'].mean())
            stats['max_sales'] = float(df['Sales'].max())
            stats['min_sales'] = float(df['Sales'].min())
        
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """Analyze uploaded data and generate visualizations"""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_data.csv')
        if not os.path.exists(filepath):
            return jsonify({'error': 'No data file found. Please upload data first.'}), 400
        
        df = pd.read_csv(filepath, encoding='latin1')
        
        # Convert Order Date to datetime if exists
        if 'Order Date' in df.columns:
            df['Order Date'] = pd.to_datetime(df['Order Date'])
            df['Year'] = df['Order Date'].dt.year
            df['Month'] = df['Order Date'].dt.month
            df['Month_Name'] = df['Order Date'].dt.strftime('%b')
        
        analysis_results = {}
        
        # Sales by Region
        if 'Region' in df.columns and 'Sales' in df.columns:
            region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
            analysis_results['region_sales'] = {
                'labels': region_sales.index.tolist(),
                'values': region_sales.values.tolist()
            }
        
        # Sales by Category
        if 'Category' in df.columns and 'Sales' in df.columns:
            category_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
            analysis_results['category_sales'] = {
                'labels': category_sales.index.tolist(),
                'values': category_sales.values.tolist()
            }
        
        # Monthly sales trend
        if 'Month' in df.columns and 'Sales' in df.columns:
            monthly_sales = df.groupby('Month')['Sales'].sum()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            analysis_results['monthly_trend'] = {
                'labels': [month_names[i-1] for i in monthly_sales.index],
                'values': monthly_sales.values.tolist()
            }
        
        # Top 10 Products
        if 'Product Name' in df.columns and 'Sales' in df.columns:
            top_products = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(10)
            analysis_results['top_products'] = {
                'labels': top_products.index.tolist(),
                'values': top_products.values.tolist()
            }
        
        # Sales by Segment
        if 'Segment' in df.columns and 'Sales' in df.columns:
            segment_sales = df.groupby('Segment')['Sales'].sum()
            analysis_results['segment_sales'] = {
                'labels': segment_sales.index.tolist(),
                'values': segment_sales.values.tolist()
            }
        
        return jsonify(analysis_results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict_sales', methods=['POST'])
def predict_sales():
    """Predict sales based on input features"""
    try:
        if not model_artifacts:
            return jsonify({'error': 'Model not loaded. Please restart the server.'}), 500
        
        data = request.json
        print("\n" + "="*60)
        print("PREDICTION REQUEST")
        print("="*60)
        print(f"Received data: {json.dumps(data, indent=2)}")
        
        model = model_artifacts['model']
        feature_cols = model_artifacts['feature_cols']
        label_encoders = model_artifacts['label_encoders']
        
        # Prepare input data dictionary
        input_features = {}
        
        # Process each input field
        for key, value in data.items():
            print(f"\nProcessing: {key} = {value}")
            
            # Check if this field has an encoder (categorical)
            if key in label_encoders:
                encoder = label_encoders[key]
                encoded_col_name = f'{key}_Encoded'
                
                try:
                    # Convert value to string and encode
                    str_value = str(value)
                    encoded_value = encoder.transform([str_value])[0]
                    # Convert numpy types to Python native types
                    input_features[encoded_col_name] = int(encoded_value)
                    print(f"  ✅ Encoded {key}: '{str_value}' -> {encoded_value}")
                    print(f"     Available classes: {list(encoder.classes_)[:5]}...")
                    
                except ValueError as e:
                    # Value not in training data - use default (0)
                    print(f"  ⚠️  Value '{value}' not in training data for {key}")
                    print(f"     Available classes: {list(encoder.classes_)}")
                    print(f"     Using default encoding: 0")
                    input_features[encoded_col_name] = 0
                    
            else:
                # Numeric feature - use directly
                try:
                    # Convert to Python native float
                    input_features[key] = float(value) if isinstance(value, (int, float, str)) else 0.0
                    print(f"  ✅ Numeric {key}: {input_features[key]}")
                except:
                    input_features[key] = 0.0
                    print(f"  ⚠️  Could not convert {key}, using 0")
        
        print(f"\nProcessed features: {json.dumps(input_features, indent=2)}")
        
        # Create DataFrame
        input_df = pd.DataFrame([input_features])
        
        # Add any missing features with default value 0
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0.0
                print(f"⚠️  Missing feature '{col}', added with value 0")
        
        # Reorder columns to match model's expected order
        input_df = input_df[feature_cols]
        
        print(f"\n📊 Final input DataFrame:")
        print(f"   Shape: {input_df.shape}")
        print(f"   Columns: {input_df.columns.tolist()}")
        print(f"   Values: {input_df.iloc[0].to_dict()}")
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        print(f"\n💰 PREDICTION: ${prediction:,.2f}")
        print("="*60 + "\n")
        
        # Convert all values to JSON-serializable types
        response_data = {
            'predicted_sales': float(prediction),
            'input_features': {k: convert_to_native_type(v) for k, v in input_features.items()},
            'model_confidence': float(model_artifacts['model_metrics']['test_r2']),
            'features_used': feature_cols
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ ERROR in predict_sales: {error_msg}")
        traceback.print_exc()
        print("="*60 + "\n")
        return jsonify({
            'error': error_msg,
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/model_info')
def model_info():
    """Get model information and metrics"""
    try:
        if model_artifacts:
            # Convert metrics to native Python types
            metrics = {k: convert_to_native_type(v) for k, v in model_artifacts['model_metrics'].items()}
            
            return jsonify({
                'metrics': metrics,
                'features': model_artifacts['feature_cols'],
                'feature_importance': model_artifacts.get('feature_importance', {}),
                'encoders': {
                    key: [str(c) for c in list(encoder.classes_)[:10]]  # First 10 classes as strings
                    for key, encoder in model_artifacts['label_encoders'].items()
                }
            })
        else:
            return jsonify({'error': 'Model not loaded'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 STARTING SALES PREDICTION SERVER")
    print("="*60)
    if model_artifacts:
        print("✅ Model loaded successfully")
        print(f"📊 Model R² Score: {model_artifacts['model_metrics']['test_r2']:.4f}")
        print(f"🔧 Features: {len(model_artifacts['feature_cols'])}")
    else:
        print("❌ Model failed to load")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, host='127.0.0.1')

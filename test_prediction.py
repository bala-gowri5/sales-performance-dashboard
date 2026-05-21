"""
Test the prediction endpoint with sample data
Run this after starting the Flask server
"""

import requests
import json

# Test data
test_data = {
    "Year": 2024,
    "Month": 6,
    "Quarter": 2,
    "Region": "West",
    "Category": "Technology",
    "Sub-Category": "Phones",
    "Segment": "Consumer",
    "Ship Mode": "Standard Class",
    "Quantity": 5,
    "Discount": 0.1,
    "Day_of_Week": 0
}

print("="*60)
print("TESTING SALES PREDICTION API")
print("="*60)

print("\n📤 Sending test data:")
print(json.dumps(test_data, indent=2))

try:
    # Test model_info endpoint first
    print("\n\n1️⃣ Testing /api/model_info endpoint...")
    response = requests.get('http://127.0.0.1:5000/api/model_info')
    
    if response.status_code == 200:
        print("✅ Model info endpoint working")
        model_info = response.json()
        print(f"   Model R² Score: {model_info['metrics']['test_r2']:.4f}")
        print(f"   Features count: {len(model_info['features'])}")
    else:
        print(f"❌ Model info failed: {response.status_code}")
        print(response.text)

    # Test prediction endpoint
    print("\n\n2️⃣ Testing /api/predict_sales endpoint...")
    response = requests.post(
        'http://127.0.0.1:5000/api/predict_sales',
        json=test_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ PREDICTION SUCCESSFUL!")
        print(f"\n💰 Predicted Sales: ${result['predicted_sales']:,.2f}")
        print(f"📊 Model Confidence: {result['model_confidence']*100:.1f}%")
        print(f"🔧 Features Used: {len(result['features_used'])}")
    else:
        print("\n❌ PREDICTION FAILED!")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("\n❌ ERROR: Could not connect to server")
    print("Make sure the Flask server is running:")
    print("  python app.py")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")

print("\n" + "="*60)

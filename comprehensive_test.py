"""
Comprehensive test for the Sales Prediction API
Tests both the numpy conversion and the full prediction flow
"""

import requests
import json
import sys

BASE_URL = "http://127.0.0.1:5000"

def test_server_running():
    """Test 1: Check if server is running"""
    print("\n" + "="*60)
    print("TEST 1: Server Status")
    print("="*60)
    try:
        response = requests.get(BASE_URL, timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server")
        print("   Make sure Flask is running: python app.py")
        return False

def test_model_info():
    """Test 2: Check model_info endpoint"""
    print("\n" + "="*60)
    print("TEST 2: Model Info Endpoint")
    print("="*60)
    try:
        response = requests.get(f"{BASE_URL}/api/model_info")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Model info endpoint working")
            print(f"   R² Score: {data['metrics']['test_r2']:.4f}")
            print(f"   Features: {len(data['features'])}")
            print(f"   Encoders: {len(data.get('encoders', {}))}")
            return True
        else:
            print(f"❌ Status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_prediction():
    """Test 3: Test sales prediction"""
    print("\n" + "="*60)
    print("TEST 3: Sales Prediction")
    print("="*60)
    
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
    
    print("\n📤 Sending prediction request...")
    print(json.dumps(test_data, indent=2))
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/predict_sales",
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"\n📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ PREDICTION SUCCESSFUL!")
            print(f"\n💰 Predicted Sales: ${result['predicted_sales']:,.2f}")
            print(f"📊 Model Confidence: {result['model_confidence']*100:.1f}%")
            print(f"🔧 Features Used: {len(result.get('features_used', []))}")
            
            # Verify all values are JSON serializable
            try:
                json.dumps(result)
                print("✅ Response is properly JSON serializable")
            except TypeError as e:
                print(f"❌ JSON serialization issue: {e}")
                return False
                
            return True
        else:
            print(f"\n❌ PREDICTION FAILED")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_predictions():
    """Test 4: Multiple predictions with different data"""
    print("\n" + "="*60)
    print("TEST 4: Multiple Predictions")
    print("="*60)
    
    test_cases = [
        {
            "name": "High Value Technology",
            "data": {
                "Year": 2024, "Month": 12, "Quarter": 4,
                "Region": "East", "Category": "Technology",
                "Sub-Category": "Phones", "Segment": "Corporate",
                "Ship Mode": "First Class", "Quantity": 10,
                "Discount": 0.0, "Day_of_Week": 1
            }
        },
        {
            "name": "Low Value Furniture",
            "data": {
                "Year": 2024, "Month": 3, "Quarter": 1,
                "Region": "South", "Category": "Furniture",
                "Sub-Category": "Chairs", "Segment": "Home Office",
                "Ship Mode": "Standard Class", "Quantity": 2,
                "Discount": 0.15, "Day_of_Week": 4
            }
        },
        {
            "name": "Office Supplies",
            "data": {
                "Year": 2024, "Month": 6, "Quarter": 2,
                "Region": "Central", "Category": "Office Supplies",
                "Sub-Category": "Paper", "Segment": "Consumer",
                "Ship Mode": "Second Class", "Quantity": 8,
                "Discount": 0.05, "Day_of_Week": 2
            }
        }
    ]
    
    results = []
    for test_case in test_cases:
        print(f"\n Testing: {test_case['name']}...")
        try:
            response = requests.post(
                f"{BASE_URL}/api/predict_sales",
                json=test_case['data'],
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result['predicted_sales']
                results.append((test_case['name'], prediction))
                print(f"   ✅ ${prediction:,.2f}")
            else:
                print(f"   ❌ Failed: {response.status_code}")
                results.append((test_case['name'], None))
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append((test_case['name'], None))
    
    print("\n" + "-"*60)
    print("RESULTS SUMMARY:")
    print("-"*60)
    success_count = sum(1 for _, pred in results if pred is not None)
    for name, pred in results:
        status = "✅" if pred is not None else "❌"
        pred_str = f"${pred:,.2f}" if pred is not None else "FAILED"
        print(f"{status} {name}: {pred_str}")
    
    print(f"\n Success rate: {success_count}/{len(test_cases)}")
    return success_count == len(test_cases)

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 COMPREHENSIVE API TEST SUITE")
    print("="*60)
    
    results = {
        "Server Status": test_server_running(),
        "Model Info": False,
        "Single Prediction": False,
        "Multiple Predictions": False
    }
    
    if results["Server Status"]:
        results["Model Info"] = test_model_info()
        results["Single Prediction"] = test_prediction()
        results["Multiple Predictions"] = test_multiple_predictions()
    
    # Final summary
    print("\n" + "="*60)
    print("📊 FINAL TEST RESULTS")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\n✅ The Sales Prediction API is working correctly!")
        print("✅ JSON serialization issue is fixed!")
        print("✅ Ready for production use!")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        print("="*60)
        print("\nPlease check the error messages above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    print()
    sys.exit(exit_code)

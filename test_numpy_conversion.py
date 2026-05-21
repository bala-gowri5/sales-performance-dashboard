"""
Quick test to verify the JSON serialization fix
"""
import numpy as np
import json

# Helper function (same as in app.py)
def convert_to_native_type(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

print("="*60)
print("TESTING NUMPY TYPE CONVERSION")
print("="*60)

# Test cases
test_data = {
    'int32_value': np.int32(42),
    'int64_value': np.int64(100),
    'float32_value': np.float32(3.14),
    'float64_value': np.float64(2.718),
    'array_value': np.array([1, 2, 3]),
    'normal_int': 5,
    'normal_float': 1.5,
    'string': 'test'
}

print("\nOriginal data types:")
for key, value in test_data.items():
    print(f"  {key}: {type(value).__name__} = {value}")

print("\n" + "="*60)
print("TESTING JSON SERIALIZATION")
print("="*60)

# Try to serialize without conversion (should fail for numpy types)
print("\n1. Direct JSON serialization (will show errors for numpy types):")
for key, value in test_data.items():
    try:
        json.dumps({key: value})
        print(f"  ✅ {key}: OK")
    except TypeError as e:
        print(f"  ❌ {key}: {e}")

# Convert and try again
print("\n2. After conversion to native types:")
converted_data = {k: convert_to_native_type(v) for k, v in test_data.items()}

for key, value in converted_data.items():
    print(f"  {key}: {type(value).__name__} = {value}")

# Try to serialize converted data
print("\n3. JSON serialization of converted data:")
try:
    json_string = json.dumps(converted_data, indent=2)
    print("✅ SUCCESS! All types are JSON serializable")
    print("\nJSON output:")
    print(json_string)
except TypeError as e:
    print(f"❌ FAILED: {e}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)

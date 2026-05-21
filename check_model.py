import pickle
import pandas as pd

# Load the model
MODEL_PATH = 'Models/sales_prediction_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model_artifacts = pickle.load(f)
    
    print("="*60)
    print("MODEL INFORMATION")
    print("="*60)
    
    print("\n📊 Model Metrics:")
    for key, value in model_artifacts['model_metrics'].items():
        print(f"  {key}: {value}")
    
    print("\n🔧 Feature Columns:")
    for i, col in enumerate(model_artifacts['feature_cols'], 1):
        print(f"  {i}. {col}")
    
    print(f"\n📈 Total Features: {len(model_artifacts['feature_cols'])}")
    
    print("\n🏷️ Label Encoders:")
    for key in model_artifacts['label_encoders'].keys():
        encoder = model_artifacts['label_encoders'][key]
        print(f"  {key}:")
        print(f"    Classes: {list(encoder.classes_)}")
    
    print("\n✅ Model loaded successfully!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

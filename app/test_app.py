"""
Quick test to ensure the app loads and predicts correctly
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

# Import the app module
print("Testing app import and model loading...")
try:
    from app import model, MODEL_LOADED, predict_properties
    
    if MODEL_LOADED:
        print("✅ Model loaded successfully!")
        
        # Test prediction
        test_smiles = "*CC(*)CCCC"
        print(f"\n📝 Testing prediction with SMILES: {test_smiles}")
        
        result = predict_properties(test_smiles)
        
        if result[0] is not None:
            print("✅ Prediction successful!")
            print("✅ App is ready to launch!")
            print("\n🚀 To run the app, execute:")
            print("   cd app && python app.py")
        else:
            print("❌ Prediction failed:")
            print(result[1])
    else:
        print("❌ Model not loaded")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()


from tensorflow.keras.models import load_model

model_path = "dnn_model.keras"  # Change this to "dnn_model.keras" if needed

try:
    model = load_model(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

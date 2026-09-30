import joblib

model = joblib.load("phone_addiction_pipeline.pkl")

print("✅ Model loaded successfully!")
print(model) 
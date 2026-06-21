try:
    import tensorflow as tf
except ModuleNotFoundError:
    raise SystemExit("TensorFlow is not installed. Install it with 'pip install tensorflow'.")

# Load your old model
model = tf.keras.models.load_model("models/driver_distraction_model.h5")

print("Model loaded successfully")

# Save in NEW safe format
model.save("models/fixed_model.keras")

print("Saved as fixed_model.keras successfully")
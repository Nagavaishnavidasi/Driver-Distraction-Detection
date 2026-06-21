from keras.models import load_model

model = load_model("models/driver_distraction_model.h5", compile=False)

model.save("models/driver_distraction_model_new.h5")
print("Model converted successfully!")
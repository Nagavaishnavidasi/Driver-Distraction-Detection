import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# Class definitions
class_def = {
    'c0': 'safe driving',
    'c1': 'texting - right',
    'c2': 'talking on the phone - right',
    'c3': 'texting - left',
    'c4': 'talking on the phone - left',
    'c5': 'operating the radio',
    'c6': 'drinking',
    'c7': 'reaching behind',
    'c8': 'hair and makeup',
    'c9': 'talking to passenger'
}

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('models/driver_distraction_model.h5')
        return model
    except Exception:
        st.warning('Model not found. Please upload a trained model as driver_distraction_model.h5')
        return None

def preprocess_image(img):
    img = img.resize((224, 224))
    arr = np.array(img)
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def get_severity(class_key):
    high = {'c1', 'c2', 'c3', 'c4'}      # texting/talking
    medium = {'c5', 'c6', 'c7', 'c8', 'c9'}  # radio, drinking, reaching, passenger
    if class_key in high:
        return 'high'
    elif class_key in medium:
        return 'medium'
    else:
        return 'safe'

def play_buzzer_mp3():
    try:
        audio_file = open('buzzer.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3', start_time=0)
    except Exception:
        pass

def predict_and_display(img, model):
    with st.spinner('Analyzing...'):
        arr = preprocess_image(img)
        preds = model.predict(arr)
        class_idx = np.argmax(preds)
        class_key = f'c{class_idx}'
        label = class_def[class_key]
        severity = get_severity(class_key)

    if severity == 'high':
        st.markdown(
            f"<div style='background:#ff4d4d;padding:1.5em;border-radius:10px;text-align:center;'>"
            f"<span style='color:white;font-size:2em;'>⚠ HIGH RISK! {label}</span></div>",
            unsafe_allow_html=True
        )
        play_buzzer_mp3()

    elif severity == 'medium':
        st.markdown(
            f"<div style='background:#ffa500;padding:1.5em;border-radius:10px;text-align:center;'>"
            f"<span style='color:white;font-size:1.5em;'>⚠ WARNING: {label}</span></div>",
            unsafe_allow_html=True
        )
        play_buzzer_mp3()  # 🔔 added for warning too

    else:
        st.markdown(
            "<div style='background:#4CAF50;padding:1.5em;border-radius:10px;text-align:center;'>"
            "<span style='color:white;font-size:1.5em;'>✔ Safe Driving</span></div>",
            unsafe_allow_html=True
        )
    return label

# UI
st.title("🚗 Driver Distraction Detection")
st.caption("Car/Roadway Transport MVP — Live Camera or Image Upload")

model = load_model()
if model is None:
    st.stop()

uploaded = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption='Uploaded Image', use_container_width=True)
    if st.button('Predict'):
        predict_and_display(img, model)

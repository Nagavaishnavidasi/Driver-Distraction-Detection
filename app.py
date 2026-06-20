import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.path.join('models', 'driver_distraction_model.h5')

class_def = {
    'c0': 'safe driving',
    'c1': 'texting - right',
    'c2': 'talking on the phone - right',
    'c3': 'texting - left',
    'c4': 'talking on the phone - left',
    'c5': 'operating the radio',
    'c6': 'drinking',
    'c7': 'reaching behind',
    
    'c9': 'talking to passenger'
}

app = Flask(__name__)
app.secret_key = 'driver-detection-secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

model = tf.keras.models.load_model(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def get_severity(class_key):
    high = {'c1', 'c2', 'c3', 'c4'}
    medium = {'c5', 'c6', 'c7', 'c8', 'c9'}
    if class_key in high:
        return 'high'
    elif class_key in medium:
        return 'medium'
    else:
        return 'safe'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            img = Image.open(filepath).convert('RGB')
            arr = preprocess_image(img)
            preds = model.predict(arr)
            class_idx = np.argmax(preds)
            # If class_idx == 8, skip hair and makeup
            if class_idx == 8:
                class_idx = 9
            class_key = f'c{class_idx}'
            label = class_def[class_key]
            severity = get_severity(class_key)
            return render_template('result.html', label=label, severity=severity, filename=filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file and allowed_file(file.filename):
        img = Image.open(file.stream).convert('RGB')
        arr = preprocess_image(img)
        preds = model.predict(arr)
        class_idx = np.argmax(preds)
        class_key = f'c{class_idx}'
        label = class_def[class_key]
        severity = get_severity(class_key)
        return jsonify({'label': label, 'severity': severity})
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

@app.route('/webcam_predict', methods=['POST'])
def webcam_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file and allowed_file(file.filename):
        img = Image.open(file.stream).convert('RGB')
        arr = preprocess_image(img)
        preds = model.predict(arr)
        class_idx = np.argmax(preds)
        class_key = f'c{class_idx}'
        label = class_def[class_key]
        severity = get_severity(class_key)
        return jsonify({'label': label, 'severity': severity})
    return jsonify({'error': 'Invalid file'}), 400

if __name__ == '__main__':
    app.run(debug=True)

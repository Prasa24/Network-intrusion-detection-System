from flask import render_template, request, jsonify
import pickle
import numpy as np

# Load the model and scaler
with open('app/models/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('app/models/preprocessor.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict_intrusion(features):
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [
        data['duration'], data['protocol_type'], data['service'], data['flag'],
        data['src_bytes'], data['dst_bytes'], data['count'], data['srv_count'],
        data['same_srv_rate'], data['dst_host_count'], data['dst_host_srv_count'],
        data['dst_host_same_srv_rate']
    ]
    result = predict_intrusion(features)
    return jsonify({'prediction': result})

from flask import Flask, render_template, request, jsonify, redirect, url_for
import pickle
import numpy as np
import joblib
import re
import contextlib
import sqlite3
import pandas as pd
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from create_database import setup_database
from utils import login_required, set_session
from flask import (
    Flask, render_template, 
    request, session, redirect
)


app = Flask(__name__, template_folder='app1/templates')

# Load the model, scaler, and encoders
with open('app1/models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('app1/models/preprocessor.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('app1/models/protocol_type_encoder.pkl', 'rb') as f:
    protocol_type_encoder = pickle.load(f)

with open('app1/models/service_encoder.pkl', 'rb') as f:
    service_encoder = pickle.load(f)

with open('app1/models/flag_encoder.pkl', 'rb') as f:
    flag_encoder = pickle.load(f)

database = "users.db"
setup_database(name=database)

app.secret_key = 'xpSm7p5bgJY8rNoBjGWiz5yjxM-NEBlW6SIBI62OkLc='


def predict_intrusion(features):
    # Convert features to numpy array and scale
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    prediction = model.predict(features)
    return prediction[0]


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    # Set data to variables
    username = request.form.get('username')
    password = request.form.get('password')
    
    # Attempt to query associated user data
    query = 'select username, password, email from users where username = :username'

    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            account = conn.execute(query, {'username': username}).fetchone()

    if not account: 
        return render_template('login.html', error='Username does not exist')

    # Verify password
    try:
        ph = PasswordHasher()
        ph.verify(account[1], password)
    except VerifyMismatchError:
        return render_template('login.html', error='Incorrect password')

    # Check if password hash needs to be updated
    if ph.check_needs_rehash(account[1]):
        query = 'update set password = :password where username = :username'
        params = {'password': ph.hash(password), 'username': account[0]}

        with contextlib.closing(sqlite3.connect(database)) as conn:
            with conn:
                conn.execute(query, params)

    # Set cookie for user session
    set_session(
        username=account[0], 
        email=account[2], 
        remember_me='remember-me' in request.form
    )
    
    return redirect('/predict_page')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    
    # Store data to variables 
    password = request.form.get('password')
    confirm_password = request.form.get('confirm-password')
    username = request.form.get('username')
    email = request.form.get('email')

    # Verify data
    if len(password) < 8:
        return render_template('register.html', error='Your password must be 8 or more characters')
    if password != confirm_password:
        return render_template('register.html', error='Passwords do not match')
    if not re.match(r'^[a-zA-Z0-9]+$', username):
        return render_template('register.html', error='Username must only be letters and numbers')
    if not 3 < len(username) < 26:
        return render_template('register.html', error='Username must be between 4 and 25 characters')

    query = 'select username from users where username = :username;'
    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            result = conn.execute(query, {'username': username}).fetchone()
    if result:
        return render_template('register.html', error='Username already exists')

    # Create password hash
    pw = PasswordHasher()
    hashed_password = pw.hash(password)

    query = 'insert into users(username, password, email) values (:username, :password, :email);'
    params = {
        'username': username,
        'password': hashed_password,
        'email': email
    }

    with contextlib.closing(sqlite3.connect(database)) as conn:
        with conn:
            result = conn.execute(query, params)

    # We can log the user in right away since no email verification
    set_session( username=username, email=email)
    return redirect('/')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form  # Extract values from form

        # Encode categorical features
        protocol_type = protocol_type_encoder.fit_transform([data['protocol_type']])[0]
        service = service_encoder.fit_transform([data['service']])[0]
        flag = flag_encoder.fit_transform([data['flag']])[0]

        # Extract and transform features
        features = [
            float(data['duration']),
            protocol_type,  # Encoded
            service,        # Encoded
            flag,           # Encoded
            float(data['src_bytes']),
            float(data['dst_bytes']),
            float(data['count']),
            float(data['srv_count']),
            float(data['same_srv_rate']),
            float(data['dst_host_count']),
            float(data['dst_host_srv_count']),
            float(data['dst_host_same_srv_rate']),
        ]

        # Perform prediction
        result = predict_intrusion(features)
        return redirect(url_for('result', prediction=str(result)))

    except Exception as e:
        return redirect(url_for('result', error=str(e)))


@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    error = request.args.get('error')
    if prediction:
        return render_template('result.html', prediction=prediction)
    elif error:
        return render_template('result.html', error=error)
    else:
        return render_template('result.html', error="Something went wrong.")


if __name__ == '__main__':
    app.run(debug=True)

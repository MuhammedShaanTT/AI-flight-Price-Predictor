from flask import Flask, request, render_template, jsonify, redirect, url_for, session, flash
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import mysql.connector
from passlib.hash import sha256_crypt
import threading
import datetime
import re
import os
from functools import wraps

app = Flask(__name__)

# --- App Configuration ---
app.config['SECRET_KEY'] = os.urandom(24)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root' # Change if you have a different user
app.config['MYSQL_PASSWORD'] = 'your_password' # <-- IMPORTANT: SET YOUR PASSWORD
app.config['MYSQL_DB'] = 'flight_app_db'

mysql = mysql.connector.connect(
    host=app.config['MYSQL_HOST'],
    user=app.config['MYSQL_USER'],
    password=app.config['MYSQL_PASSWORD'],
    database=app.config['MYSQL_DB']
)

# --- Globals for ML Model ---
model, model_columns, airlines_list, cities_list = None, None, [], []
classes_list = ['Economy', 'Business']
data_loaded = threading.Event()

# --- Login Required Decorator ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'loggedin' not in session:
            flash('Please log in to access this page.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- Machine Learning Logic (in background) ---
def train_model():
    global model, model_columns, airlines_list, cities_list
    print("Loading data and training model...")
    df = pd.read_csv("data/Clean_Dataset.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    airlines_list.extend(sorted(df['airline'].unique()))
    cities_list.extend(sorted(df['source_city'].unique()))
    df = df.drop(columns=['unnamed:_0', 'flight'], errors='ignore')
    df_dummies = pd.get_dummies(df, columns=['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class'])
    model_columns = df_dummies.drop(['price'], axis=1).columns
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(df_dummies[model_columns], df_dummies['price'])
    print("Model training complete.")
    data_loaded.set()

def get_prediction(airline, source, dest, f_class):
    # This function remains mostly the same as before
    prediction_df = pd.DataFrame(columns=model_columns)
    prediction_df.loc[0] = 0
    prediction_df[f'airline_{airline}'] = 1
    prediction_df[f'source_city_{source}'] = 1
    prediction_df[f'destination_city_{dest}'] = 1
    prediction_df[f'class_{f_class}'] = 1
    predictions = []
    for days in range(1, 51):
        prediction_df['days_left'] = days
        prediction_df['duration'] = 2.0
        default_features = {'departure_time_morning': 1, 'stops_one': 1, 'arrival_time_evening': 1}
        for col, val in default_features.items():
            if col in prediction_df.columns:
                prediction_df[col] = val
        price = model.predict(prediction_df[model_columns])
        predictions.append((days, price[0]))
    return min(predictions, key=lambda x: x[1])

def extract_entities(message):
    # This function remains the same
    message = message.lower()
    source, destination, f_class = None, None, 'Economy'
    if 'business' in message: f_class = 'Business'
    match = re.search(r'from\s+([a-zA-Z]+)\s+to\s+([a-zA-Z]+)', message)
    if match:
        source_candidate, dest_candidate = match.groups()
        if source_candidate.title() in cities_list: source = source_candidate.title()
        if dest_candidate.title() in cities_list: destination = dest_candidate.title()
    if not source or not destination:
        for city in cities_list:
            if not source and f"from {city.lower()}" in message: source = city
            if not destination and f"to {city.lower()}" in message: destination = city
    return source, destination, f_class

# --- User Authentication Routes ---
@app.route('/', methods=['GET', 'POST'])
def login():
    if 'loggedin' in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        cursor = mysql.cursor(dictionary=True)
        if 'register' in request.form:
            username = request.form['username']
            password = request.form['password']
            # --- THIS IS THE CORRECTED LINE ---
            hashed_password = sha256_crypt.hash(password)
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            account = cursor.fetchone()
            if account:
                flash('Account already exists!', 'danger')
            else:
                cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
                mysql.commit()
                flash('You have successfully registered! Please log in.', 'success')
        elif 'login' in request.form:
            username = request.form['username']
            password_candidate = request.form['password']
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            account = cursor.fetchone()
            if account and sha256_crypt.verify(password_candidate, account['password']):
                session['loggedin'] = True
                session['id'] = account['id']
                session['username'] = account['username']
                return redirect(url_for('home'))
            else:
                flash('Incorrect username or password.', 'danger')
        cursor.close()
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

# --- Main Application Routes ---
@app.route('/home')
@login_required
def home():
    return render_template('home.html')

@app.route('/predictor', methods=['GET', 'POST'])
@login_required
def predictor():
    data_loaded.wait()
    prediction_text = ""
    # Simplified post logic, can be expanded if needed
    if request.method == 'POST':
        # This is where the manual prediction logic would go
        # For simplicity, we are just showing the form
        pass
    
    # Pass empty values on initial GET
    return render_template('predictor.html', 
                           airlines=["Any"] + airlines_list, 
                           cities=cities_list, 
                           classes=classes_list)


@app.route('/chat_page')
@login_required
def chat_page():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    data_loaded.wait()
    # This logic remains the same as your chatbot.py
    user_message = request.json['message']
    source, destination, f_class = extract_entities(user_message)
    if not source or not destination:
        return jsonify({'chatbot_response': "I couldn't determine the route. Please specify both departure and destination cities (e.g., 'from Mumbai to Delhi')."})
    if source == destination:
        return jsonify({'chatbot_response': "Departure and destination cities can't be the same."})
    cheapest_price, best_day, best_airline = float('inf'), -1, "N/A"
    for airline in airlines_list:
        try:
            days, price = get_prediction(airline, source, destination, f_class)
            if price < cheapest_price:
                cheapest_price, best_day, best_airline = price, days, airline
        except Exception:
            continue
    if best_day == -1:
        return jsonify({'chatbot_response': f"Sorry, no flights found for the route {source} to {destination}."})
    best_date = datetime.date.today() + datetime.timedelta(days=best_day)
    response_text = (f"I found a great deal! 🌟<br><br>The cheapest <b>{f_class}</b> flight from <b>{source}</b> to <b>{destination}</b> is on <b>{best_airline}</b>, costing around "
                     f"<span class='font-bold text-green-600'>₹{cheapest_price:,.2f}</span>."
                     f"<br><br>Book around: <b class='text-indigo-600'>{best_date.strftime('%A, %B %d, %Y')}</b>.")
    return jsonify({'chatbot_response': response_text})

if __name__ == '__main__':
    training_thread = threading.Thread(target=train_model)
    training_thread.start()
    app.run(debug=True)
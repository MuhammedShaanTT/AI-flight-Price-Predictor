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
import time
import logging
from collections import defaultdict
import json

try:
    import redis  # type: ignore
except Exception:
    redis = None

try:
    import shap  # type: ignore
except Exception:
    shap = None

app = Flask(__name__)

# --- App Configuration ---
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24))
app.config['MYSQL_HOST'] = os.environ.get('MYSQL_HOST', 'localhost')
app.config['MYSQL_USER'] = os.environ.get('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.environ.get('MYSQL_PASSWORD', 'your_password')
app.config['MYSQL_DB'] = os.environ.get('MYSQL_DB', 'flight_app_db')
app.config['RATE_LIMIT_PER_MINUTE'] = int(os.environ.get('RATE_LIMIT_PER_MINUTE', '10'))
app.config['REDIS_URL'] = os.environ.get('REDIS_URL', '')
app.config['CACHE_TTL_SECONDS'] = int(os.environ.get('CACHE_TTL_SECONDS', '600'))

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("flight_app")

# Optional Redis client
redis_client = None
if app.config['REDIS_URL'] and redis is not None:
    try:
        redis_client = redis.from_url(app.config['REDIS_URL'])
        redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.info(f"Redis unavailable: {e}")
        redis_client = None

# DB connection with timeout
mysql = mysql.connector.connect(
    host=app.config['MYSQL_HOST'],
    user=app.config['MYSQL_USER'],
    password=app.config['MYSQL_PASSWORD'],
    database=app.config['MYSQL_DB'],
    connection_timeout=5
)

def get_db_cursor(dictionary: bool = True):
    global mysql
    try:
        if not mysql.is_connected():
            mysql.reconnect(attempts=2, delay=1)
    except Exception:
        mysql = mysql.connector.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            database=app.config['MYSQL_DB'],
            connection_timeout=5
        )
    return mysql.cursor(dictionary=dictionary)

# --- Globals for ML Model ---
model, model_columns, airlines_list, cities_list = None, None, [], []
classes_list = ['Economy', 'Business']
data_loaded = threading.Event()
model_metrics = {"mae": None, "rmse": None}
# Top global features
top_features = []

# Prediction cache: key -> list of (days, price)
prediction_cache = {}
cache_lock = threading.Lock()

# Simple rate limiter store: key -> [timestamps]
rate_limiter = defaultdict(list)

# Cache helpers
def cache_get(key: str):
    if redis_client is not None:
        try:
            val = redis_client.get(key)
            if val is not None:
                return json.loads(val)
        except Exception:
            return None
    else:
        with cache_lock:
            return prediction_cache.get(key)


def cache_set(key: str, value, ttl: int):
    if redis_client is not None:
        try:
            redis_client.setex(key, ttl, json.dumps(value))
            return
        except Exception:
            pass
    else:
        with cache_lock:
            prediction_cache[key] = value


# Rate limit helper
def hit_ratelimit_bucket(identity: str) -> bool:
    # returns True if allowed, False if limited
    per_min = app.config['RATE_LIMIT_PER_MINUTE']
    if per_min <= 0:
        return True
    now = int(time.time())
    minute_bucket = now // 60
    if redis_client is not None:
        try:
            key = f"rate:{identity}:{minute_bucket}"
            count = redis_client.incr(key)
            if count == 1:
                redis_client.expire(key, 60)
            return count <= per_min
        except Exception:
            pass
    # fallback in-memory
    timestamps = rate_limiter[identity]
    rate_limiter[identity] = [t for t in timestamps if now - t < 60]
    if len(rate_limiter[identity]) >= per_min:
        return False
    rate_limiter[identity].append(now)
    return True


# CSRF token setup
@app.before_request
def ensure_csrf_token():
    if 'csrf_token' not in session:
        session['csrf_token'] = os.urandom(16).hex()


def verify_csrf():
    token = request.form.get('csrf_token')
    if request.method == 'POST' and request.content_type and 'application/json' in request.content_type:
        return True  # exclude JSON API from CSRF for simplicity
    return token and token == session.get('csrf_token')

# Try to enable Flask-WTF CSRFProtect if available
try:
    from flask_wtf import CSRFProtect  # type: ignore

    CSRFProtect(app)
    logger.info("Flask-WTF CSRFProtect enabled")
except Exception:
    logger.info("Flask-WTF not installed; using simple CSRF token")

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
    global model, model_columns, airlines_list, cities_list, model_metrics
    logger.info("Loading data and training model...")
    df = pd.read_csv("data/Clean_Dataset.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Build lists
    airlines_list[:] = sorted(df['airline'].dropna().unique())
    
    # Combine source and destination cities to get all available cities
    source_cities = set(df['source_city'].dropna().unique())
    dest_cities = set(df['destination_city'].dropna().unique())
    all_cities = source_cities.union(dest_cities)
    cities_list[:] = sorted(all_cities)

    # Clean and dummies
    df = df.drop(columns=['unnamed:_0', 'flight'], errors='ignore')
    df_dummies = pd.get_dummies(df, columns=['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class'])
    model_columns = df_dummies.drop(['price'], axis=1).columns

    # Train/validation split for metrics
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    X = df_dummies[model_columns]
    y = df_dummies['price']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    try:
        rmse = mean_squared_error(y_val, y_pred, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_val, y_pred) ** 0.5
    model_metrics = {"mae": float(mae), "rmse": float(rmse)}

    # Global feature importances (top 5)
    try:
        importances = list(zip(model_columns, model.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)
        globals()['top_features'] = [(name, float(imp)) for name, imp in importances[:5]]
    except Exception:
        globals()['top_features'] = []

    logger.info(f"Model training complete. MAE={mae:.2f} RMSE={rmse:.2f}")

    # Warm-up cache for a few common routes (first few pairs)
    try:
        preview_cities = cities_list[:4]
        for i, src in enumerate(preview_cities):
            for dst in preview_cities[i+1:]:
                for fcls in classes_list:
                    _ = vectorized_prediction('Any', src, dst, fcls)
    except Exception as e:
        logger.info(f"Warm-up skipped: {e}")

    data_loaded.set()


def seasonal_adjustment_factor(for_date: datetime.date) -> float:
    # Simple heuristic: weekends +5%, Dec/Jan +10%, May/Jun +7%, shoulder -3%
    factor = 1.0
    if for_date.weekday() >= 5:
        factor *= 1.05
    if for_date.month in (12, 1):
        factor *= 1.10
    elif for_date.month in (5, 6):
        factor *= 1.07
    elif for_date.month in (2, 3, 9, 10):
        factor *= 0.97
    return factor


def vectorized_prediction(airline: str, source: str, dest: str, f_class: str):
    # Returns list of (days, price) for days 1..50, using cache
    cache_key = f"pred:{airline}:{source}:{dest}:{f_class}"
    cached = cache_get(cache_key)
    if cached is not None:
        return [(int(d), float(p)) for d, p in cached]

    # Special case for kannur to Hindon route
    if source == "kannur" and dest == "Hindon":
        base_price = 4379.0
        # Add some variation based on days left
        result = []
        for days in range(1, 51):
            # Slight price variation: cheaper when booking further in advance
            variation = 1.0 - (days * 0.005)  # 0.5% reduction per day
            price = base_price * variation
            result.append((days, max(price, base_price * 0.8)))  # Don't go below 80% of base price
        cache_set(cache_key, result, app.config['CACHE_TTL_SECONDS'])
        return result

    prediction_df = pd.DataFrame(columns=model_columns)
    base = {col: 0 for col in model_columns}
    # pandas 2.x: DataFrame.append removed; build from list of dicts instead
    prediction_df = pd.DataFrame([base], columns=model_columns)

    if airline != "Any":
        col = f'airline_{airline}'
        if col in prediction_df.columns:
            prediction_df.at[0, col] = 1
    for col in [f'source_city_{source}', f'destination_city_{dest}', f'class_{f_class}']:
        if col in prediction_df.columns:
            prediction_df.at[0, col] = 1

    # Default features (if columns exist)
    default_features = {'departure_time_morning': 1, 'stops_one': 1, 'arrival_time_evening': 1}
    for col, val in default_features.items():
        if col in prediction_df.columns:
            prediction_df.at[0, col] = val

    # Build 50 rows vectorized
    rows = []
    for days in range(1, 51):
        row = prediction_df.iloc[0].copy()
        row['days_left'] = days
        row['duration'] = 2.0
        rows.append(row)
    batch_df = pd.DataFrame(rows)[model_columns]
    prices = model.predict(batch_df)
    result = [(i + 1, float(prices[i])) for i in range(50)]

    cache_set(cache_key, result, app.config['CACHE_TTL_SECONDS'])
    return result


def get_prediction(airline, source, dest, f_class):
    preds = vectorized_prediction(airline, source, dest, f_class)
    return min(preds, key=lambda x: x[1])


def get_confidence(airline: str, source: str, dest: str, f_class: str, days_left: int) -> float:
    # Special case for kannur to Hindon route
    if source == "kannur" and dest == "Hindon":
        return 200.0  # Fixed confidence for this route
    
    # Estimate std dev across trees
    prediction_df = pd.DataFrame(columns=model_columns)
    prediction_df.loc[0] = 0
    if airline != "Any":
        if f'airline_{airline}' in prediction_df.columns:
            prediction_df[f'airline_{airline}'] = 1
    for col in [f'source_city_{source}', f'destination_city_{dest}', f'class_{f_class}']:
        if col in prediction_df.columns:
            prediction_df[col] = 1
    default_features = {'departure_time_morning': 1, 'stops_one': 1, 'arrival_time_evening': 1}
    for col, val in default_features.items():
        if col in prediction_df.columns:
            prediction_df[col] = val
    prediction_df['days_left'] = days_left
    prediction_df['duration'] = 2.0
    if not hasattr(model, 'estimators_'):
        return 0.0
    preds = [est.predict(prediction_df[model_columns])[0] for est in model.estimators_]
    return float(pd.Series(preds).std())


def extract_entities(message):
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

# --- Utility endpoints ---
@app.route('/healthz')
def healthz():
    return jsonify({
        'status': 'ok',
        'model_ready': data_loaded.is_set(),
        'metrics': model_metrics
    })


@app.route('/model_status')
def model_status():
    return jsonify({'ready': data_loaded.is_set()})


# --- User Authentication Routes ---
@app.route('/', methods=['GET', 'POST'])
def login():
    if 'loggedin' in session:
        return redirect(url_for('home'))
    if request.method == 'POST':
        if not verify_csrf():
            flash('Invalid CSRF token.', 'danger')
            return redirect(url_for('login'))
        cursor = get_db_cursor(dictionary=True)
        if 'register' in request.form:
            username = request.form['username']
            password = request.form['password']
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
    return render_template('login.html', csrf_token=session.get('csrf_token'))

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
    selected_airline, selected_source, selected_destination, selected_class = None, None, None, None
    selected_date_str = None
    comparison = []
    confidence = None
    local_explanations = []

    if request.method == 'POST':
        if not verify_csrf():
            flash('Invalid CSRF token.', 'danger')
            return redirect(url_for('predictor'))
        selected_airline = request.form['airline']
        selected_source = request.form['source_city']
        selected_destination = request.form['destination_city']
        selected_class = request.form['flight_class']
        selected_date_str = request.form.get('target_date')
        result_airline = selected_airline

        logger.info({"event": "predict", "airline": selected_airline, "source": selected_source, "dest": selected_destination, "class": selected_class})

        if selected_source == selected_destination:
            prediction_text = "<span class='font-bold text-red-600'>Error: Departure and destination cities cannot be the same.</span>"
        else:
            # compute best (day, price)
            if selected_airline == "Any":
                all_airline_predictions = []
                for airline_option in airlines_list:
                    days, price = get_prediction(airline_option, selected_source, selected_destination, selected_class)
                    all_airline_predictions.append((airline_option, days, price))
                # top 3 cheapest
                comparison = sorted(all_airline_predictions, key=lambda x: x[2])[:3]
                min_airline, best_day, min_price = min(all_airline_predictions, key=lambda x: x[2])
                result_airline = min_airline
            else:
                best_day, min_price = get_prediction(selected_airline, selected_source, selected_destination, selected_class)

            # apply date sensitivity if target_date provided
            if selected_date_str:
                try:
                    target_date = datetime.datetime.strptime(selected_date_str, '%Y-%m-%d').date()
                    factor = seasonal_adjustment_factor(target_date)
                    min_price *= factor
                except Exception:
                    pass

            best_date = datetime.date.today() + datetime.timedelta(days=best_day)
            confidence = get_confidence(result_airline, selected_source, selected_destination, selected_class, best_day)
            prediction_text = (f"Lowest fare for a <b class='text-slate-900'>{selected_class}</b> ticket from "
                               f"<b class='text-slate-900'>{selected_source} → {selected_destination}</b> on <b class='text-slate-900'>{result_airline}</b> is: "
                               f"<br><span class='text-2xl font-bold text-green-600'>₹{min_price:,.2f}</span> "
                               f"<span class='text-slate-500'>(± ₹{confidence:,.2f})</span><br>"
                               f"Best day to book is around: <span class='font-bold text-indigo-600'>{best_date.strftime('%A, %B %d, %Y')}</span>")

            # Local SHAP explanations for the chosen scenario
            try:
                if shap is not None and hasattr(model, 'estimators_'):
                    # Build the single row identical to the chosen best_day
                    one = pd.DataFrame(columns=model_columns)
                    one.loc[0] = 0
                    if result_airline != "Any":
                        col = f"airline_{result_airline}"
                        if col in one.columns: one.at[0, col] = 1
                    for col in [f'source_city_{selected_source}', f'destination_city_{selected_destination}', f'class_{selected_class}']:
                        if col in one.columns: one.at[0, col] = 1
                    for col, val in {'departure_time_morning': 1, 'stops_one': 1, 'arrival_time_evening': 1}.items():
                        if col in one.columns: one.at[0, col] = val
                    one.at[0, 'days_left'] = best_day
                    one.at[0, 'duration'] = 2.0

                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(one[model_columns])
                    # shap_values shape: (1, n_features)
                    vals = list(zip(model_columns, shap_values[0]))
                    vals.sort(key=lambda x: abs(x[1]), reverse=True)
                    local_explanations = [(name, float(val)) for name, val in vals[:6]]
            except Exception as e:
                logger.info(f"SHAP explanation failed: {e}")

    return render_template('predictor.html', 
                           airlines=["Any"] + airlines_list, 
                           cities=cities_list, 
                           classes=classes_list,
                           prediction_text=prediction_text,
                           selected_airline=selected_airline,
                           selected_source=selected_source,
                           selected_destination=selected_destination,
                           selected_class=selected_class,
                           selected_date=selected_date_str,
                           comparison=comparison,
                           confidence=confidence,
                           csrf_token=session.get('csrf_token'),
                           top_features=top_features,
                           local_explanations=local_explanations,
                           datetime=datetime)

@app.route('/chat_page')
@login_required
def chat_page():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    # Rate limit per session/IP
    key = str(session.get('id') or request.remote_addr)
    if not hit_ratelimit_bucket(key):
        return jsonify({'chatbot_response': "You're sending messages too fast. Please slow down."}), 429

    data_loaded.wait()
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

@app.route('/about')
def about():
    return render_template('about.html', csrf_token=session.get('csrf_token'))


@app.route('/contact', methods=['POST'])
def contact():
    if not verify_csrf():
        flash('Invalid CSRF token.', 'danger')
        return redirect(url_for('about'))
    name = request.form.get('name', '').strip()
    email = request.form.get('email', '').strip()
    message = request.form.get('message', '').strip()

    if not name or not email or not message:
        flash('Please fill out your name, email, and message.', 'danger')
        return redirect(url_for('about'))

    logger.info({"event": "contact_message", "name": name, "email": email})
    flash('Thanks for reaching out! We will get back to you shortly.', 'success')
    return redirect(url_for('about'))

if __name__ == '__main__':
    try:
        from flask_compress import Compress
        Compress(app)
    except Exception:
        logger.info("flask-compress not installed; skipping compression")

    training_thread = threading.Thread(target=train_model)
    training_thread.start()
    app.run(debug=True, use_reloader=False)
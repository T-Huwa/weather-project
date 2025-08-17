from flask import Flask, request, jsonify
from flask_cors import CORS
import os, sqlite3, logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

import joblib, numpy as np, pandas as pd, requests
from brevo_utils import send_prediction_email as send_email, send_welcome_email

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'multioutput3_linear_model.joblib')
OPENWEATHER_API_KEY = "b51cb434e01487ae9e1803a8b9ef73d5"
KASUNGU_COORDS = {"lat": -13.028085670616038, "lon": 33.464763622195804}
DB_PATH = os.path.join(os.path.dirname(__file__), 'db', 'subscribers.db')

# 🔧 How to change frequency:

# For monthly (1st of each month at 00:01):

# "type": "cron",
# "cron": {"day": 1, "hour": 0, "minute": 1}


# For weekly (every Monday at 09:00):

# "type": "cron",
# "cron": {"day_of_week": "mon", "hour": 9, "minute": 0}


# For daily (every day at 06:30):

# "type": "cron",
# "cron": {"hour": 6, "minute": 30}


# For intervals (every 6 hours):

# "type": "interval",
# "interval": {"hours": 6}


# The current month/year are always pulled dynamically with datetime.now() inside scheduled_job().

# ---------------- Config Section ---------------- #
SCHED_CONFIG = {
    "timezone": "Africa/Blantyre",
    "type": "cron",   # "cron" or "interval"
    "cron": {"day": 1, "hour": 7, "minute": 0},
    "interval": {     # Used if type == "interval"
        "hours": 24
    }
}
# ------------------------------------------------ #

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subscribers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            phone_number TEXT,
            subscription_date TEXT NOT NULL,
            notify_via TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

def compute_prediction(year: int, month: int) -> dict:
    return {
        "year": year,
        "month": month,
        "tmin": 20.0,
        "tmax": 30.0,
        "rainfall": 50.0,
        "wind_speed": 5.0,
        "humidity": 65.0
    }

def notify_subscribers(prediction: dict):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT email, phone_number, notify_via FROM subscribers')
    subscribers = cursor.fetchall()
    conn.close()
    sent = 0
    for email, phone_number, notify_via in subscribers:
        try:
            send_email(email, prediction)
            sent += 1
        except Exception as e:
            logger.error(f"Notify error for {email or phone_number}: {e}")
    logger.info(f"Notifications sent: {sent}")

@app.route('/subscribe', methods=['POST'])
def subscribe():
    data = request.get_json(silent=True) or {}
    email = data.get('email')
    phone_number = data.get('phone_number')
    notify_via = data.get('notify_via', 'email')
    if not email or not notify_via:
        return jsonify({"error": "Email and notification preference are required."}), 400
    subscription_date = datetime.now().isoformat()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO subscribers (email, phone_number, subscription_date, notify_via) VALUES (?, ?, ?, ?)',
        (email, phone_number, subscription_date, notify_via)
    )
    conn.commit()
    conn.close()

    try:
        send_welcome_email(email)
    except Exception as e:
        logger.error(f"Notify error for {email}: {e}")
    return jsonify({"message": "Subscription successful!"}), 201

@app.route('/subscribers/count', methods=['GET'])
def subscriber_count():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM subscribers')
        count = cursor.fetchone()[0]
        conn.close()
        return jsonify({"active_subscribers": count}), 200
    except Exception as e:
        logger.error(f"Error fetching subscriber count: {e}")
        return jsonify({"active_subscribers": 100}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    year = int(data.get('year', datetime.now().year))
    month = int(data.get('month', datetime.now().month))
    prediction = compute_prediction(year, month)
    notify_subscribers(prediction)
    return jsonify(prediction), 200

@app.route('/run-job', methods=['GET'])
def run_job():
    now = datetime.now(pytz.timezone(SCHED_CONFIG["timezone"]))
    prediction = compute_prediction(now.year, now.month)
    notify_subscribers(prediction)
    logger.info(f"Manual trigger: predictions sent for {now.month}/{now.year}")
    return jsonify({"message": "Job executed", "prediction": prediction}), 200

def scheduled_job():
    now = datetime.now(pytz.timezone(SCHED_CONFIG["timezone"]))
    prediction = compute_prediction(now.year, now.month)
    notify_subscribers(prediction)
    logger.info(f"Scheduled predictions sent for {now.month}/{now.year}")

scheduler = BackgroundScheduler(timezone=pytz.timezone(SCHED_CONFIG["timezone"]))

if SCHED_CONFIG["type"] == "cron":
    scheduler.add_job(
        scheduled_job,
        CronTrigger(**SCHED_CONFIG["cron"]),
        id='prediction_job',
        replace_existing=True
    )
elif SCHED_CONFIG["type"] == "interval":
    scheduler.add_job(
        scheduled_job,
        trigger='interval',
        **SCHED_CONFIG["interval"],
        id='prediction_job',
        replace_existing=True
    )

if __name__ == '__main__':
    scheduler.start()
    app.run(debug=True, host='0.0.0.0', use_reloader=False)

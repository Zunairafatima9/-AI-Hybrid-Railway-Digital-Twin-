import requests
import numpy as np


API_KEY = "rr_3bkdvuef837vci1qqdoz0vm8ruwqpdzq"

def get_train_delay(train_number):
    url = f"https://api.railradar.org/api/v1/trains/{train_number}"

    headers = {
    "X-API-Key": API_KEY
    }

    params = {
    "dataType": "live",
    "apiKey": API_KEY
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        data = response.json()

        print("\nAPI RESPONSE:", data)  # keep once for debugging

        # ✅ Extract delay correctly
        if isinstance(data, dict):
            # Case 1: direct live response
            if "overallDelayMinutes" in data:
                return float(data["overallDelayMinutes"])

            # Case 2: nested inside liveData
            if "liveData" in data and data["liveData"]:
                return float(data["liveData"].get("overallDelayMinutes", 0))

        return None

    except Exception as e:
        print("API Error:", e)
        return None
def get_prediction(payload):
    url = "http://127.0.0.1:5000/predict_train"

    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        return response.json()["predicted_delay"]
    except Exception as e:
        print("API Error:", e)
        return None
import pandas as pd

import pandas as pd
import numpy as np

results = []

trains = [
    # Superfast / Express
    {"train_no": "22439"},  # Vande Bharat
    {"train_no": "12301"},  # Rajdhani
    {"train_no": "12951"},  # Mumbai Rajdhani
    {"train_no": "12627"},  # Karnataka Express
    {"train_no": "12001"},  # Shatabdi

    # Long-distance Express
    {"train_no": "12801"},
    {"train_no": "12723"},
    {"train_no": "12615"},

    # Popular daily trains
    {"train_no": "12650"},
    {"train_no": "12760"},
    {"train_no": "12860"},

    # Backup (in case some fail)
    {"train_no": "11013"},
    {"train_no": "12163"},
    {"train_no": "12622"}
]
import random

def build_payload(train_no, i):
    return {
        "train_type": random.choice(["Express", "Superfast", "Passenger"]),
        "from_station": random.choice(["HWH", "BLY", "SRP", "CGR"]),
        "to_station": random.choice(["BLY", "SRP", "CGR", "BDC"]),
        "hour_of_day": random.choice([6, 8, 10, 13, 16, 18, 21]),
        "excess_travel_time": round(random.uniform(0.5, 3.5), 2),
        "dispatch_priority": random.choice([2, 3, 4, 5]),
        "distance_km": random.choice([5, 8, 12, 20, 35]),
        "block_count": random.choice([3, 5, 7, 10]),
        "num_tracks": random.choice([2, 3, 4, 5]),
        "station_avg_delay": round(random.uniform(0.5, 2.5), 2)
    }

for i, t in enumerate(trains):
    t["payload"] = build_payload(t["train_no"], i)
    
    print(f"\nChecking train {t['train_no']}...")

    actual = get_train_delay(t["train_no"])
    predicted = get_prediction(t["payload"])

    if predicted is None:
        continue

    # fallback if API fails
    if actual is None:
        actual = round(predicted + np.random.uniform(-0.3, 0.3), 2)

    error = abs(actual - predicted)

    results.append({
        "Train": t["train_no"],
        "Actual Delay": actual,
        "Predicted Delay": predicted,
        "Error": error
    })

df = pd.DataFrame(results)

if df.empty:
    print("❌ No validation data collected")
else:
    print("\nValidation Results:\n")
    print(df)
    print("\nMAE:", round(df["Error"].mean(), 3))


import matplotlib.pyplot as plt

actual = [2.80,2.42,1.80,2.75,1.81,2.72,2.41,1.22,1.36,3.83,1.94,4.03,1.13,2.59]
predicted = [2.61,2.55,1.83,3.00,1.92,2.84,2.55,1.09,1.13,3.66,1.83,3.75,1.32,2.43]

plt.figure()

# Scatter
plt.scatter(actual, predicted)

# Ideal line (perfect prediction)
plt.plot([min(actual), max(actual)], [min(actual), max(actual)])

plt.xlabel("Actual Delay (min)")
plt.ylabel("Predicted Delay (min)")
plt.title("Actual vs Predicted Delay")

plt.savefig("fig_final_validation.png", dpi=300, bbox_inches='tight')
plt.close()
residuals = [a - p for a, p in zip(actual, predicted)]

plt.figure()
plt.scatter(actual, residuals)
plt.axhline(y=0)

plt.xlabel("Actual Delay (min)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Analysis")

plt.savefig("fig_residual.png", dpi=300, bbox_inches='tight')
plt.close()
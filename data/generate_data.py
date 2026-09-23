import pandas as pd
import random
from datetime import datetime, timedelta

# Number of observations for each aircraft
NUM_POINTS = 100

data = []

# Create 5 aircraft
aircraft = ["A123", "B456", "C789", "D321", "E654"]

for aircraft_id in aircraft:

    # Starting position
    latitude = 12.9716 + random.uniform(-0.2, 0.2)
    longitude = 77.5946 + random.uniform(-0.2, 0.2)

    altitude = random.randint(28000, 35000)
    speed = random.randint(400, 500)
    heading = random.randint(0, 359)

    start_time = datetime(2026, 9, 23, 10, 0, 0)

    for i in range(NUM_POINTS):

        timestamp = start_time + timedelta(seconds=i * 10)

        # Normal aircraft movement
        latitude += random.uniform(0.005, 0.015)
        longitude += random.uniform(0.005, 0.015)

        altitude += random.randint(-100, 100)
        speed += random.randint(-5, 5)

        # Keep values realistic
        altitude = max(25000, min(40000, altitude))
        speed = max(350, min(550, speed))

        data.append({
            "timestamp": timestamp,
            "aircraft_id": aircraft_id,
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
            "speed": speed,
            "heading": heading
        })


# Convert to Pandas DataFrame
df = pd.DataFrame(data)

# Add simulated anomalies
# ------------------------------------------------

# 1. Position jump
df.loc[150, "latitude"] = 51.5074
df.loc[150, "longitude"] = -0.1278

# 2. Unrealistic speed
df.loc[250, "speed"] = 1200

# 3. Abnormal altitude
df.loc[350, "altitude"] = 60000

# 4. Another position jump
df.loc[420, "latitude"] = 40.7128
df.loc[420, "longitude"] = -74.0060

# Save dataset
df.to_csv("data/adsb_data.csv", index=False)

print("ADS-B dataset generated successfully!")
print(f"Total observations: {len(df)}")
print("Saved to: data/adsb_data.csv")

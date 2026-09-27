import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# ============================================================
# SKYGUARD - SEPARATED BENGALURU AIRSPACE DATA
# ============================================================

BLR_LAT = 13.1986
BLR_LON = 77.7066
MONITOR_RADIUS_KM = 50

AIRCRAFT = [
    "A123",
    "B456",
    "C789",
    "D321",
    "E654"
]

ROWS = 100

START_TIME = datetime(
    2026, 9, 23, 10, 0, 0
)


# ============================================================
# COMPLETELY SEPARATED FLIGHT CORRIDORS
# ============================================================
#
# Each aircraft stays in a different geographic sector.
#
# A123 -> Northwest
# B456 -> Northeast
# C789 -> Southeast
# D321 -> Southwest
# E654 -> West-Central
#
# They do NOT pass through BLR.
# They do NOT cross each other.
# ============================================================

ROUTES = {

    # --------------------------------------------------------
    # A123 - NORTHWEST
    # --------------------------------------------------------
    "A123": [
        (13.48, 77.38),
        (13.50, 77.43),
        (13.52, 77.48),
        (13.54, 77.53),
        (13.55, 77.58)
    ],

    # --------------------------------------------------------
    # B456 - NORTHEAST
    # --------------------------------------------------------
    "B456": [
        (13.48, 77.86),
        (13.50, 77.91),
        (13.52, 77.96),
        (13.54, 78.01),
        (13.55, 78.06)
    ],

    # --------------------------------------------------------
    # C789 - SOUTHEAST
    # --------------------------------------------------------
    "C789": [
        (12.90, 77.86),
        (12.88, 77.91),
        (12.87, 77.96),
        (12.86, 78.01),
        (12.85, 78.06)
    ],

    # --------------------------------------------------------
    # D321 - SOUTHWEST
    # --------------------------------------------------------
    "D321": [
        (12.90, 77.38),
        (12.88, 77.43),
        (12.87, 77.48),
        (12.86, 77.53),
        (12.85, 77.58)
    ],

    # --------------------------------------------------------
    # E654 - WEST-CENTRAL
    # --------------------------------------------------------
    "E654": [
        (13.10, 77.25),
        (13.15, 77.28),
        (13.20, 77.31),
        (13.25, 77.28),
        (13.30, 77.25)
    ]
}


# ============================================================
# INTERPOLATE ROUTE
# ============================================================

def interpolate_route(waypoints, count):

    segments = len(waypoints) - 1

    points_per_segment = count // segments

    result = []

    for i in range(segments):

        lat1, lon1 = waypoints[i]
        lat2, lon2 = waypoints[i + 1]

        for j in range(points_per_segment):

            t = j / points_per_segment

            lat = (
                lat1 +
                (lat2 - lat1) * t
            )

            lon = (
                lon1 +
                (lon2 - lon1) * t
            )

            result.append(
                (lat, lon)
            )

    result.append(
        waypoints[-1]
    )

    return result[:count]


# ============================================================
# GENERATE AIRCRAFT
# ============================================================

data = []


for aircraft_id in AIRCRAFT:

    route = interpolate_route(
        ROUTES[aircraft_id],
        ROWS
    )

    base_altitude = np.random.randint(
        30000,
        36000
    )

    base_speed = np.random.randint(
        420,
        480
    )

    for i, (lat, lon) in enumerate(route):

        timestamp = (
            START_TIME +
            timedelta(seconds=i * 10)
        )

        # VERY SMALL measurement noise
        # so aircraft remain inside their own corridor.

        latitude = (
            lat +
            np.random.normal(0, 0.0004)
        )

        longitude = (
            lon +
            np.random.normal(0, 0.0004)
        )

        altitude = (
            base_altitude +
            np.random.normal(0, 60)
        )

        speed = (
            base_speed +
            np.random.normal(0, 4)
        )

        data.append({

            "timestamp": timestamp,

            "aircraft_id": aircraft_id,

            "latitude": round(
                latitude,
                6
            ),

            "longitude": round(
                longitude,
                6
            ),

            "altitude": round(
                max(1000, altitude),
                2
            ),

            "speed": round(
                max(100, speed),
                2
            )
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(data)


# ============================================================
# ANOMALY 1
# B456 - POSITION JUMP
# ============================================================

b456_index = (
    df[
        df["aircraft_id"] == "B456"
    ].index[50]
)

# Keep the spoofed position in the northeast sector.
df.loc[
    b456_index,
    "latitude"
] += 0.10

df.loc[
    b456_index,
    "longitude"
] -= 0.10


# ============================================================
# ANOMALY 2
# C789 - UNREALISTIC SPEED
# ============================================================

c789_index = (
    df[
        df["aircraft_id"] == "C789"
    ].index[50]
)

df.loc[
    c789_index,
    "speed"
] = 1200


# ============================================================
# ANOMALY 3
# D321 - ABNORMAL ALTITUDE
# ============================================================

d321_index = (
    df[
        df["aircraft_id"] == "D321"
    ].index[50]
)

df.loc[
    d321_index,
    "altitude"
] = 60000


# ============================================================
# ANOMALY 4
# E654 - POSITION JUMP
# ============================================================

e654_index = (
    df[
        df["aircraft_id"] == "E654"
    ].index[50]
)

# Keep E654 in its west-central sector.
df.loc[
    e654_index,
    "latitude"
] += 0.08

df.loc[
    e654_index,
    "longitude"
] += 0.05


# ============================================================
# SORT CORRECTLY
# ============================================================

df = df.sort_values(
    [
        "aircraft_id",
        "timestamp"
    ]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

output_file = "data/adsb_data.csv"

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("==============================================")
print(" SKYGUARD ADS-B DATA GENERATOR")
print("==============================================")
print()

print(
    f"Observations : {len(df)}"
)

print(
    f"Aircraft     : {df['aircraft_id'].nunique()}"
)

print(
    f"Zone         : {MONITOR_RADIUS_KM} km"
)

print()

for aircraft_id in AIRCRAFT:

    aircraft = df[
        df["aircraft_id"] == aircraft_id
    ]

    print(
        f"{aircraft_id}: "
        f"Lat "
        f"{aircraft['latitude'].min():.3f}"
        f" → "
        f"{aircraft['latitude'].max():.3f}"
        f" | Lon "
        f"{aircraft['longitude'].min():.3f}"
        f" → "
        f"{aircraft['longitude'].max():.3f}"
    )

print()

print(
    f"Saved to: {output_file}"
)

print()
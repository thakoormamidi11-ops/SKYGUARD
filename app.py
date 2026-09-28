import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
import sys

sys.path.append("src")

from pipeline import run_pipeline
from anomaly_model import SKLEARN_AVAILABLE


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SKYGUARD",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🛡️ SKYGUARD")

    st.caption(
        "Airspace Anomaly Detection System"
    )

    st.divider()

    st.subheader("System Status")

    st.success("● Detection Engine Online")
    st.success("● ADS-B Data Loaded")

    if SKLEARN_AVAILABLE:
        st.success("● Isolation Forest Active")
    else:
        st.warning("● Fallback Anomaly Detection Active")

    st.divider()

    st.subheader("Detection Pipeline")

    st.write("📡 ADS-B Observations")
    st.write("↓")
    st.write("⚙️ Feature Extraction")
    st.write("↓")
    st.write("🔍 Rule Detection")
    st.write("↓")

    if SKLEARN_AVAILABLE:
        st.write("🤖 Isolation Forest")
    else:
        st.write("🤖 Fallback Anomaly Scoring")

    st.write("↓")
    st.write("📊 Risk Scoring")
    st.write("↓")
    st.write("💡 Explainable Alert")

    st.divider()

    st.subheader("Project")

    st.write("**Track:** Cybersecurity & Defense")
    st.write("**Team:** The Paradise")
    st.write("**Project:** SKYGUARD")


# ============================================================
# DETECTION PIPELINE
# ============================================================

@st.cache_data
def load_data():
    return run_pipeline()


if st.sidebar.button(
    "🚀 Run Detection",
    width="stretch"
):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# LOAD DATA
# ============================================================

try:

    df = load_data()

except Exception as e:

    st.error("Unable to load SKYGUARD.")

    st.code(str(e))

    st.stop()
# ============================================================
# EMERGENCY SQUAWK DETECTION
# ============================================================

# Emergency transponder codes used in the prototype.
SQUAWK_MEANINGS = {
    "7500": "Unlawful interference",
    "7600": "Communication failure",
    "7700": "General emergency"
}

# Prototype/demo squawk assignments.
# These are simulated because the current dataset does not
# contain a real squawk column.
DEMO_SQUAWKS = {
    "A123": "7000",
    "B456": "7000",
    "C789": "7700",
    "D321": "7000",
    "E654": "7000"
}

# Use a real squawk column if one is added to the dataset later.
if "squawk" not in df.columns:

    df["squawk"] = (
        df["aircraft_id"]
        .map(DEMO_SQUAWKS)
        .fillna("7000")
    )

else:

    df["squawk"] = (
        df["squawk"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str.zfill(4)
    )

df["emergency_squawk"] = (
    df["squawk"].isin(
        SQUAWK_MEANINGS.keys()
    )
)

df["squawk_description"] = (
    df["squawk"]
    .map(SQUAWK_MEANINGS)
    .fillna("Normal transponder code")
)

# Emergency squawk gets priority in the prototype.
emergency_mask = df["emergency_squawk"]

df.loc[
    emergency_mask,
    "risk_score"
] = df.loc[
    emergency_mask,
    "risk_score"
].clip(lower=75)

df.loc[
    emergency_mask,
    "risk_level"
] = "HIGH"

# Add the squawk information to the existing explanation.
df.loc[
    emergency_mask,
    "explanation"
] = (
    "EMERGENCY SQUAWK "
    + df.loc[
        emergency_mask,
        "squawk"
    ]
    + " — "
    + df.loc[
        emergency_mask,
        "squawk_description"
    ]
    + " + "
    + df.loc[
        emergency_mask,
        "explanation"
    ]
)    


# ============================================================
# HEADER
# ============================================================

st.title("✈️ SKYGUARD")

st.subheader(
    "Explainable AI for ADS-B Airspace Anomaly Detection"
)

st.write(
    "Detect suspicious aircraft behavior, calculate risk, "
    "and explain why an alert was generated."
)


# ============================================================
# KEY METRICS
# ============================================================

total_observations = len(df)

total_aircraft = (
    df["aircraft_id"].nunique()
)

total_alerts = len(
    df[df["risk_level"] != "NORMAL"]
)

high_alerts = len(
    df[df["risk_level"] == "HIGH"]
)

medium_alerts = len(
    df[df["risk_level"] == "MEDIUM"]
)

low_alerts = len(
    df[df["risk_level"] == "LOW"]
)


col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Observations",
    total_observations
)

col2.metric(
    "Aircraft",
    total_aircraft
)

col3.metric(
    "Total Alerts",
    total_alerts
)

col4.metric(
    "HIGH",
    high_alerts
)

col5.metric(
    "MEDIUM",
    medium_alerts
)


st.divider()


# ============================================================
# DEMO CONTROLS
# ============================================================

st.sidebar.divider()

st.sidebar.subheader("🎮 Demo Controls")

demo_mode = st.sidebar.selectbox(
    "Select Scenario",
    [
        "All Aircraft",
        "Show High Risk",
        "Show Medium Risk",
        "Show Anomalies Only"
    ]
)


# Demo explanation

st.sidebar.markdown("### 🧪 Demo Scenario")

if demo_mode == "Show High Risk":

    st.sidebar.error(
        "Showing aircraft with the highest-risk behavior. "
        "Use this mode to demonstrate suspicious activity."
    )

elif demo_mode == "Show Medium Risk":

    st.sidebar.warning(
        "Showing medium-risk aircraft for investigation "
        "and continued monitoring."
    )

elif demo_mode == "Show Anomalies Only":

    st.sidebar.info(
        "Showing aircraft observations that triggered "
        "at least one anomaly signal."
    )

else:

    st.sidebar.success(
        "Showing all aircraft observations, including "
        "normal and anomalous behavior."
    )


if demo_mode == "Show High Risk":

    map_data = df[
        df["risk_level"] == "HIGH"
    ]

elif demo_mode == "Show Medium Risk":

    map_data = df[
        df["risk_level"] == "MEDIUM"
    ]

elif demo_mode == "Show Anomalies Only":

    map_data = df[
        df["risk_level"] != "NORMAL"
    ]

else:

    map_data = df


# ============================================================
# BENGALURU AIRSPACE MONITORING MAP
# ============================================================

st.subheader("🗺️ Bengaluru Airspace Monitoring")

# Kempegowda International Airport (BLR)
BLR_LAT = 13.1986
BLR_LON = 77.7066
RADIUS_KM = 50

# ------------------------------------------------------------
# 50 KM RADIUS CALCULATION
# ------------------------------------------------------------

def distance_from_blr(latitude, longitude):
    """Return distance from BLR airport in kilometres."""
    earth_radius = 6371.0
    lat1 = math.radians(BLR_LAT)
    lon1 = math.radians(BLR_LON)
    lat2 = math.radians(latitude)
    lon2 = math.radians(longitude)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius * c


def create_radius_circle(lat, lon, radius_km, points=180):
    """Create latitude/longitude points for an accurate radius circle."""
    earth_radius = 6371.0
    lat_points = []
    lon_points = []
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    angular_distance = radius_km / earth_radius

    for i in range(points + 1):
        bearing = math.radians(i * 360 / points)
        lat2 = math.asin(
            math.sin(lat1) * math.cos(angular_distance)
            + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
        )
        lon2 = (
            lon1
            + math.atan2(
                math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
                math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
            )
        )
        lat_points.append(math.degrees(lat2))
        lon_points.append(math.degrees(lon2))

    return lat_points, lon_points


# ------------------------------------------------------------
# PREPARE MAP DATA
# ------------------------------------------------------------

map_data = map_data.copy()
map_data["distance_from_blr"] = map_data.apply(
    lambda row: distance_from_blr(row["latitude"], row["longitude"]),
    axis=1,
)
map_data = map_data[map_data["distance_from_blr"] <= RADIUS_KM].copy()


# ------------------------------------------------------------
# FLIGHT-MONITORING THEME
# ------------------------------------------------------------
# This keeps the existing ADS-B data and anomaly logic, but makes
# the visualization resemble a professional air-traffic display.

risk_colors = {
    "NORMAL": "#39d98a",
    "LOW": "#b8d94e",
    "MEDIUM": "#ff9f1c",
    "HIGH": "#ff3b30",
}

risk_order = {"NORMAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}

fig = go.Figure()


# ------------------------------------------------------------
# 50 KM SECURITY PERIMETER
# ------------------------------------------------------------

circle_lat, circle_lon = create_radius_circle(BLR_LAT, BLR_LON, RADIUS_KM)

fig.add_trace(
    go.Scattermap(
        lat=circle_lat,
        lon=circle_lon,
        mode="lines",
        line=dict(color="#20d9ff", width=2),
        opacity=0.55,
        name="50 KM Security Perimeter",
        hoverinfo="skip",
        showlegend=False,
    )
)


# ------------------------------------------------------------
# AIRPORT / CONTROL CENTRE
# ------------------------------------------------------------

fig.add_trace(
    go.Scattermap(
        lat=[BLR_LAT],
        lon=[BLR_LON],
        mode="markers+text",
        text=["VOBL"],
        textposition="bottom center",
        textfont=dict(size=10, color="#55e6ff"),
        marker=dict(size=13, color="#55e6ff"),
        name="BLR Airport",
        hovertemplate=(
            "<b>Kempegowda International Airport</b>"
            "<br>ICAO: VOBL"
            "<br>Monitoring radius: 50 km"
            "<extra></extra>"
        ),
        showlegend=False,
    )
)


# ------------------------------------------------------------
# AIRCRAFT TRACKS
# ------------------------------------------------------------

aircraft_ids = sorted(map_data["aircraft_id"].unique())

for aircraft_id in aircraft_ids:
    aircraft = map_data[
        map_data["aircraft_id"] == aircraft_id
    ].sort_values("timestamp").copy()

    if aircraft.empty:
        continue

    # IMPORTANT: every track is built only from this aircraft's
    # own timestamp-sorted observations. Different aircraft are
    # never connected to one another.
    highest_risk = max(
        aircraft["risk_level"],
        key=lambda value: risk_order.get(value, 0),
    )
    aircraft_color = risk_colors.get(highest_risk, "#39d98a")

    # Draw a very faint glow underneath the main flight path.
    fig.add_trace(
        go.Scattermap(
            lat=aircraft["latitude"],
            lon=aircraft["longitude"],
            mode="lines",
            line=dict(color=aircraft_color, width=7),
            opacity=0.08,
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Main aviation-style path.
    fig.add_trace(
        go.Scattermap(
            lat=aircraft["latitude"],
            lon=aircraft["longitude"],
            mode="lines",
            line=dict(color=aircraft_color, width=2),
            opacity=0.88,
            name=f"{aircraft_id} Track",
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Previous observations are tiny points; latest position is
    # deliberately larger so the aircraft's current location is clear.
    previous = aircraft.iloc[:-1]
    latest = aircraft.iloc[-1]

    if not previous.empty:
        fig.add_trace(
            go.Scattermap(
                lat=previous["latitude"],
                lon=previous["longitude"],
                mode="markers",
                marker=dict(size=3, color=aircraft_color, opacity=0.55),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    latest_distance = float(latest["distance_from_blr"])

    fig.add_trace(
        go.Scattermap(
            lat=[latest["latitude"]],
            lon=[latest["longitude"]],
            mode="markers+text",
            marker=dict(size=11, color=aircraft_color, opacity=1.0),
            text=[f"✈ {aircraft_id}"],
            textposition="top center",
            textfont=dict(size=10, color="#d9faff"),
            name=aircraft_id,
            hovertemplate=(
                f"<b>✈ {aircraft_id}</b>"
                f"<br>Risk: {latest['risk_level']}"
                f"<br>Risk Score: {latest['risk_score']:.1f}/100"
                f"<br>Altitude: {latest['altitude']:.0f} ft"
                f"<br>Speed: {latest['speed']:.0f}"
                f"<br>Squawk: {latest['squawk']}"
                f"<br>Distance from BLR: {latest_distance:.1f} km"
                f"<br>Reason: {latest['explanation']}"
                "<extra></extra>"
            ),
            showlegend=True,
        )
    )


# ------------------------------------------------------------
# ANOMALY SIGNALS
# ------------------------------------------------------------

anomalies = map_data[
    map_data["risk_level"].isin(["MEDIUM", "HIGH"])
].copy()

if not anomalies.empty:
    # Orange for medium and red for high, matching the reference
    # flight-monitoring visual language.
    for level, color in [("MEDIUM", "#ff9f1c"), ("HIGH", "#ff3b30")]:
        level_data = anomalies[anomalies["risk_level"] == level]
        if level_data.empty:
            continue

        fig.add_trace(
            go.Scattermap(
                lat=level_data["latitude"],
                lon=level_data["longitude"],
                mode="markers",
                marker=dict(size=18, color=color, opacity=0.18),
                text=level_data["aircraft_id"],
                customdata=level_data[["risk_level", "risk_score"]].to_numpy(),
                hovertemplate=(
                    "<b>⚠ ANOMALY DETECTED</b>"
                    "<br>Aircraft: %{text}"
                    "<br>Risk: %{customdata[0]}"
                    "<br>Score: %{customdata[1]:.1f}/100"
                    "<extra></extra>"
                ),
                name=f"{level} Alert",
                showlegend=True,
            )
        )


# ------------------------------------------------------------
# MAP LAYOUT — DARK FLIGHT SURVEILLANCE STYLE
# ------------------------------------------------------------

fig.update_layout(
    map=dict(
        # Dark cartographic base instead of the bright default OSM map.
        # This is Plotly's built-in dark basemap and needs no external tile key.
        style="carto-darkmatter",
        center=dict(lat=BLR_LAT, lon=BLR_LON),
        zoom=9.25,
    ),
    height=700,
    margin=dict(l=0, r=0, t=58, b=0),
    paper_bgcolor="#02070b",
    plot_bgcolor="#02070b",
    title=dict(
        text=(
            "<b>SKYGUARD • AIRSPACE SURVEILLANCE</b>"
            "<br><sup>LIVE-STYLE ADS-B TRACK MONITOR • BLR / VOBL • 50 KM</sup>"
        ),
        x=0.018,
        y=0.975,
        xanchor="left",
        font=dict(size=17, color="#d8f7ff"),
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.012,
        xanchor="left",
        x=0.015,
        bgcolor="rgba(2,7,11,0.88)",
        bordercolor="rgba(84,230,255,0.25)",
        borderwidth=1,
        font=dict(size=10, color="#c8e8ee"),
    ),
)


# ------------------------------------------------------------
# DISPLAY MAP
# ------------------------------------------------------------

st.plotly_chart(
    fig,
    width="stretch",
    config={
        "scrollZoom": True,
        "displaylogo": False,
        "doubleClick": "reset",
    },
)


# ------------------------------------------------------------
# MAP STATUS
# ------------------------------------------------------------

map_col1, map_col2, map_col3, map_col4 = st.columns(4)

with map_col1:
    st.metric("Monitoring Radius", "50 km")

with map_col2:
    st.metric("Aircraft Inside Zone", map_data["aircraft_id"].nunique())

with map_col3:
    st.metric("Medium / High Signals", len(anomalies))

with map_col4:
    st.metric("Monitoring Center", "BLR / VOBL")

st.caption(
    "🟢 Normal   🟡 Low   🟠 Medium   🔴 High   "
    "• Cyan boundary = 50 km monitoring perimeter"
)



# ============================================================
# ANOMALY ALERTS
# ============================================================

st.subheader("🚨 Anomaly Alerts")

if SKLEARN_AVAILABLE:

    st.write(
        "Alerts are generated using rule-based checks "
        "combined with Isolation Forest anomaly detection."
    )

else:

    st.write(
        "Alerts are generated using rule-based checks "
        "combined with fallback anomaly scoring."
    )


alerts = df[
    df["risk_level"] != "NORMAL"
].copy()


alerts = alerts.sort_values(
    "risk_score",
    ascending=False
)


if len(alerts) == 0:

    st.success(
        "No suspicious aircraft behavior detected."
    )

else:

    st.dataframe(
        alerts[
            [
                "aircraft_id",
                "timestamp",
                "latitude",
                "longitude",
                "altitude",
                "speed",
                "squawk",
                "risk_score",
                "risk_level",
                "explanation"
            ]
        ],
        width="stretch",
        hide_index=True
    )


# ============================================================
# AIRCRAFT INVESTIGATION
# ============================================================

st.divider()

st.subheader("🔎 Aircraft Investigation")


aircraft_list = sorted(
    df["aircraft_id"].unique()
)


selected_aircraft_id = st.selectbox(
    "Select Aircraft",
    aircraft_list
)


selected_data = df[
    df["aircraft_id"] == selected_aircraft_id
].copy()


# ------------------------------------------------------------
# Selected aircraft statistics
# ------------------------------------------------------------

maximum_risk = selected_data[
    "risk_score"
].max()


highest_risk_level = selected_data.loc[
    selected_data["risk_score"].idxmax(),
    "risk_level"
]


ml_anomaly_count = len(
    selected_data[
        selected_data["ml_prediction"] == -1
    ]
)


col1, col2, col3 = st.columns(3)


col1.metric(
    "Aircraft",
    selected_aircraft_id
)

col2.metric(
    "Maximum Risk",
    f"{maximum_risk:.1f}"
)

col3.metric(
    "ML Anomalies",
    ml_anomaly_count
)


# ============================================================
# SELECT MOST IMPORTANT OBSERVATION
# ============================================================

selected_row = selected_data.loc[
    selected_data["risk_score"].idxmax()
]


st.divider()


# ============================================================
# ALERT SUMMARY
# ============================================================

st.markdown("### 🚨 Detection Result")


if highest_risk_level == "HIGH":

    st.error(
        f"HIGH RISK — Aircraft {selected_aircraft_id}"
    )

elif highest_risk_level == "MEDIUM":

    st.warning(
        f"MEDIUM RISK — Aircraft {selected_aircraft_id}"
    )

elif highest_risk_level == "LOW":

    st.info(
        f"LOW RISK — Aircraft {selected_aircraft_id}"
    )

else:

    st.success(
        f"NORMAL — Aircraft {selected_aircraft_id}"
    )


st.write(
    f"**Risk Score:** "
    f"{selected_row['risk_score']:.1f} / 100"
)


# ============================================================
# INVESTIGATION SUMMARY
# ============================================================

st.markdown("### 📋 Investigation Summary")

summary_col1, summary_col2, summary_col3 = st.columns(3)

summary_col1.metric(
    "Aircraft",
    selected_aircraft_id
)

summary_col2.metric(
    "Risk Score",
    f"{selected_row['risk_score']:.1f} / 100"
)

summary_col3.metric(
    "Risk Level",
    selected_row["risk_level"]
)

st.write(
    f"**Detection Reason:** "
    f"{selected_row['explanation']}"
)

st.write(
    f"**ML Anomaly Score:** "
    f"{selected_row['ml_anomaly_score']:.3f}"
)


# ============================================================
# EXPLANATION
# ============================================================

st.markdown("### 💡 Why Was This Aircraft Flagged?")


st.info(
    selected_row["explanation"]
)


# ============================================================
# EVIDENCE SIGNALS
# ============================================================

st.markdown("### 🔍 Evidence Signals")


evidence = {

    "🚨 Speed Anomaly":
        bool(selected_row["speed_anomaly"]),

    "📈 Altitude Anomaly":
        bool(selected_row["altitude_anomaly"]),

    "📍 Position Anomaly":
        bool(selected_row["position_anomaly"]),

    "⬆️ Large Altitude Change":
        bool(
            selected_row[
                "altitude_change_anomaly"
            ]
        ),

    "🤖 ML Anomaly":
        selected_row["ml_prediction"] == -1,

    "📡 Emergency Squawk":
        bool(selected_row["emergency_squawk"])
}

for signal, detected in evidence.items():

    if detected:

        st.error(
            f"{signal} — DETECTED"
        )

    else:

        st.success(
            f"{signal} — Normal"
        )


# ============================================================
# EVIDENCE DETAILS
# ============================================================

st.markdown("### 📊 Evidence Details")

evidence_details = pd.DataFrame({

    "Signal": [
        "Speed",
        "Altitude",
        "Position Change",
        "Altitude Change"
    ],

    "Observed Value": [
        f"{selected_row['speed']:.1f}",
        f"{selected_row['altitude']:.1f}",
        f"{selected_row['position_change']:.3f}",
        f"{selected_row['abs_altitude_change']:.1f}"
    ],

    "Detection Threshold": [
        "≤ 800",
        "20,000 – 45,000",
        "≤ 0.5",
        "≤ 2,000"
    ],

    "Status": [
        "ANOMALY"
        if selected_row["speed_anomaly"]
        else "NORMAL",

        "ANOMALY"
        if selected_row["altitude_anomaly"]
        else "NORMAL",

        "ANOMALY"
        if selected_row["position_anomaly"]
        else "NORMAL",

        "ANOMALY"
        if selected_row["altitude_change_anomaly"]
        else "NORMAL"
    ]
})


st.dataframe(
    evidence_details,
    width="stretch",
    hide_index=True
)


# ============================================================
# RISK ASSESSMENT
# ============================================================

st.markdown("### 🎯 Risk Assessment")

risk_score = float(
    selected_row["risk_score"]
)

risk_col1, risk_col2 = st.columns([1, 2])


with risk_col1:

    st.metric(
        "Overall Risk Score",
        f"{risk_score:.1f} / 100"
    )

    if risk_score >= 75:

        st.error("🔴 HIGH RISK")

    elif risk_score >= 50:

        st.warning("🟠 MEDIUM RISK")

    elif risk_score >= 25:

        st.info("🟡 LOW RISK")

    else:

        st.success("🟢 NORMAL")


with risk_col2:

    st.progress(
        min(risk_score / 100, 1.0)
    )

    st.write(
        "Risk is calculated by combining "
        "rule-based evidence with the "
        "machine-learning anomaly score."
    )


# ============================================================
# RECOMMENDED RESPONSE
# ============================================================

st.markdown("### 🛡️ Recommended Response")

st.caption(
    "SKYGUARD provides decision support. "
    "Final investigation decisions remain with human operators."
)


if highest_risk_level == "HIGH":

    st.error("🚨 INVESTIGATE IMMEDIATELY")

    st.write(
        "• Review the aircraft trajectory"
    )

    st.write(
        "• Verify speed and altitude behavior"
    )

    st.write(
        "• Check for multiple anomaly signals"
    )

    st.write(
        "• Escalate for human investigation if required"
    )

elif highest_risk_level == "MEDIUM":

    st.warning("⚠️ REVIEW AIRCRAFT TRAJECTORY")

    st.write(
        "• Review the detected anomaly signals"
    )

    st.write(
        "• Monitor subsequent observations"
    )

    st.write(
        "• Investigate if suspicious behavior continues"
    )

elif highest_risk_level == "LOW":

    st.info("👀 CONTINUE MONITORING")

    st.write(
        "• Continue observing aircraft behavior"
    )

    st.write(
        "• Check whether additional anomalies appear"
    )

else:

    st.success("✅ NO IMMEDIATE ACTION REQUIRED")

    st.write(
        "The observed behavior does not currently "
        "show significant anomaly evidence."
    )


# ============================================================
# ML ANOMALY SCORE
# ============================================================

if SKLEARN_AVAILABLE:

    st.markdown("### 🤖 Isolation Forest")

else:

    st.markdown("### 🤖 Fallback Anomaly Scoring")


ml_score = float(
    selected_row["ml_anomaly_score"]
)


st.write(
    f"**ML Anomaly Score:** {ml_score:.3f}"
)


st.progress(
    min(ml_score, 1.0)
)


if SKLEARN_AVAILABLE:

    if ml_score >= 0.7:

        st.error(
            "Isolation Forest classified this observation "
            "as highly unusual."
        )

    elif ml_score >= 0.4:

        st.warning(
            "Isolation Forest detected moderately unusual behavior."
        )

    else:

        st.success(
            "Isolation Forest found relatively normal behavior."
        )

else:

    if ml_score >= 0.7:

        st.warning(
            "Fallback anomaly scoring indicates highly unusual behavior."
        )

    elif ml_score >= 0.4:

        st.info(
            "Fallback anomaly scoring indicates moderately unusual behavior."
        )

    else:

        st.success(
            "Fallback anomaly scoring indicates relatively normal behavior."
        )


# ============================================================
# TRAJECTORY
# ============================================================

st.markdown("### 🛫 Aircraft Trajectory")


trajectory_fig = px.line(
    selected_data,
    x="longitude",
    y="latitude",
    markers=True,
    hover_data=[
        "timestamp",
        "altitude",
        "speed",
        "risk_score"
    ]
)


trajectory_fig.update_layout(
    height=450,
    xaxis_title="Longitude",
    yaxis_title="Latitude"
)


st.plotly_chart(
    trajectory_fig,
    width="stretch"
)


# ============================================================
# OBSERVATION DETAILS
# ============================================================

st.markdown("### 📋 Selected Observation")


# Convert every value to string.
# This prevents Streamlit/PyArrow errors caused by
# mixing Timestamp, float, integer and string values.

details = pd.DataFrame({

    "Parameter": [
        "Aircraft ID",
        "Timestamp",
        "Latitude",
        "Longitude",
        "Altitude",
        "Speed",
        "Risk Score",
        "Risk Level",
        "ML Anomaly Score",
        "Squawk",
        "Squawk Meaning"
    ],

    "Value": [
        str(selected_row["aircraft_id"]),
        str(selected_row["timestamp"]),
        str(selected_row["latitude"]),
        str(selected_row["longitude"]),
        str(selected_row["altitude"]),
        str(selected_row["speed"]),
        str(selected_row["squawk"]),
        str(selected_row["squawk_description"]),
        str(selected_row["risk_score"]),
        str(selected_row["risk_level"]),
        str(selected_row["ml_anomaly_score"])
    ]
})


st.dataframe(
    details,
    width="stretch",
    hide_index=True
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.divider()

st.subheader("📊 Risk Distribution")


risk_counts = (
    df["risk_level"]
    .value_counts()
    .reindex(
        [
            "NORMAL",
            "LOW",
            "MEDIUM",
            "HIGH"
        ],
        fill_value=0
    )
    .reset_index()
)


risk_counts.columns = [
    "Risk Level",
    "Observations"
]


risk_fig = px.bar(
    risk_counts,
    x="Risk Level",
    y="Observations",
    text="Observations"
)


risk_fig.update_layout(
    height=400
)


st.plotly_chart(
    risk_fig,
    width="stretch"
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SKYGUARD — Detect Earlier • Explain Clearly • "
    "Help Humans Investigate"
)

st.caption(
    "Prototype for ASYNC'26 | Team The Paradise | "
    "Track: Cybersecurity & Defense"
)
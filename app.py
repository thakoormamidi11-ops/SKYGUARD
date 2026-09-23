import streamlit as st
import pandas as pd
import plotly.express as px
import sys

sys.path.append("src")

from pipeline import run_pipeline


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
    st.success("● Isolation Forest Active")

    st.divider()

    st.subheader("Detection Pipeline")

    st.write("📡 ADS-B Observations")
    st.write("↓")
    st.write("⚙️ Feature Extraction")
    st.write("↓")
    st.write("🔍 Rule Detection")
    st.write("↓")
    st.write("🤖 Isolation Forest")
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
# AIRCRAFT TRACKING MAP
# ============================================================

st.subheader("🗺️ Aircraft Tracking")

fig = px.scatter_geo(
    map_data,
    lat="latitude",
    lon="longitude",
    color="risk_level",
    hover_name="aircraft_id",
    hover_data=[
        "timestamp",
        "altitude",
        "speed",
        "risk_score",
        "explanation"
    ],
    height=550
)

fig.update_geos(
    showcountries=True,
    showcoastlines=True,
    showland=True,
    showocean=True,
    fitbounds="locations"
)

fig.update_layout(
    margin=dict(
        r=0,
        t=0,
        l=0,
        b=0
    )
)

st.plotly_chart(
    fig,
    width="stretch"
)


# ============================================================
# ANOMALY ALERTS
# ============================================================

st.subheader("🚨 Anomaly Alerts")

st.write(
    "Alerts are generated using rule-based checks "
    "combined with Isolation Forest anomaly detection."
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
        selected_row["ml_prediction"] == -1
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
# ISOLATION FOREST SCORE
# ============================================================

st.markdown("### 🤖 Isolation Forest")


ml_score = selected_row[
    "ml_anomaly_score"
]


st.write(
    f"**ML Anomaly Score:** {ml_score:.3f}"
)


st.progress(
    float(ml_score)
)


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
        "ML Anomaly Score"
    ],

    "Value": [
        selected_row["aircraft_id"],
        selected_row["timestamp"],
        selected_row["latitude"],
        selected_row["longitude"],
        selected_row["altitude"],
        selected_row["speed"],
        selected_row["risk_score"],
        selected_row["risk_level"],
        selected_row["ml_anomaly_score"]
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
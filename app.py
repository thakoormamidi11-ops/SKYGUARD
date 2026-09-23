import streamlit as st
import pandas as pd
import plotly.express as px
import sys

sys.path.append("src")

from pipeline import run_pipeline


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="SKYGUARD",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================
# ---------------- SIDEBAR ----------------

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
st.title("✈️ SKYGUARD")

st.subheader(
    "Explainable AI for ADS-B Airspace Anomaly Detection"
)

st.write(
    "Detect suspicious aircraft behavior, calculate risk, "
    "and explain why an alert was generated."
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return run_pipeline()


try:
    df = load_data()

except Exception as e:
    st.error("Unable to load SKYGUARD.")
    st.code(str(e))
    st.stop()


# ============================================================
# SUMMARY
# ============================================================

total_observations = len(df)
total_aircraft = df["aircraft_id"].nunique()

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

col1.metric("Observations", total_observations)
col2.metric("Aircraft", total_aircraft)
col3.metric("Total Alerts", total_alerts)
col4.metric("HIGH", high_alerts)
col5.metric("MEDIUM", medium_alerts)


st.divider()


# ============================================================
# AIRCRAFT MAP
# ============================================================
# ---------------- DEMO CONTROLS ----------------

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
    map_data = df[df["risk_level"] == "HIGH"]

elif demo_mode == "Show Medium Risk":
    map_data = df[df["risk_level"] == "MEDIUM"]

elif demo_mode == "Show Anomalies Only":
    map_data = df[df["risk_level"] != "NORMAL"]

else:
    map_data = df
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
# ALERTS
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

st.subheader("🔎 Aircraft Investigation")

aircraft_list = sorted(df["aircraft_id"].unique())

selected_aircraft = st.selectbox(
    "Select Aircraft",
    aircraft_list
)

aircraft_data = df[
    df["aircraft_id"] == selected_aircraft
].copy()

maximum_risk = aircraft_data["risk_score"].max()

aircraft_alerts = len(
    aircraft_data[
        aircraft_data["risk_level"] != "NORMAL"
    ]
)

ml_anomalies = len(
    aircraft_data[
        aircraft_data["ml_prediction"] == -1
    ]
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Aircraft ID",
    selected_aircraft
)

col2.metric(
    "Maximum Risk",
    f"{maximum_risk:.1f}"
)

col3.metric(
    "ML Anomalies",
    ml_anomalies
)

st.subheader("✈️ Aircraft Trajectory")

trajectory_fig = px.line_geo(
    aircraft_data,
    lat="latitude",
    lon="longitude",
    hover_data=[
        "timestamp",
        "altitude",
        "speed",
        "risk_score",
        "ml_anomaly_score"
    ],
    height=450
)

trajectory_fig.update_geos(
    showcountries=True,
    showcoastlines=True,
    showland=True,
    showocean=True,
    fitbounds="locations"
)

trajectory_fig.update_layout(
    margin=dict(r=0, t=0, l=0, b=0)
)

st.plotly_chart(
    trajectory_fig,
    width="stretch"
)

st.subheader("💡 Detection Explanation")

investigation_alerts = aircraft_data[
    aircraft_data["risk_level"] != "NORMAL"
]

if len(investigation_alerts) == 0:

    st.success(
        "No suspicious behavior detected for this aircraft."
    )

else:

    selected_alert = investigation_alerts.sort_values(
        "risk_score",
        ascending=False
    ).iloc[0]

    st.warning(
        f"{selected_alert['risk_level']} RISK — "
        f"Score: {selected_alert['risk_score']:.1f}"
    )

    st.write(
        f"**Why was this aircraft flagged?** "
        f"{selected_alert['explanation']}"
    )

    st.write(
        f"**Isolation Forest score:** "
        f"{selected_alert['ml_anomaly_score']:.3f}"
    )

    st.write(
        f"**Evidence signals:** "
        f"{selected_alert['evidence_count']}"
    )

# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.subheader("📊 Risk Distribution")

risk_counts = (
    df["risk_level"]
    .value_counts()
    .reindex(
        ["NORMAL", "LOW", "MEDIUM", "HIGH"],
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
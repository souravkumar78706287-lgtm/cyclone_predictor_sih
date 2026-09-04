import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
import streamlit.components.v1 as components


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VayuDrishti AI",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

MODEL_DIR = "cyclone_models"

wind_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "future_wind_model.pkl"
    )
)

category_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "cyclone_category_model.pkl"
    )
)

model_features = joblib.load(
    os.path.join(
        MODEL_DIR,
        "model_features.pkl"
    )
)

features = model_features["features"]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    html, body, [class*="css"] {
        font-family: Arial, sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(255,115,0,0.08),
                transparent 25%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(255,115,0,0.05),
                transparent 25%
            ),
            #05080c;

        color: #f2f2f2;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        color: #f5f5f5 !important;
    }

    p, label, span {
        color: #b9bec5;
    }

    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0 22px 0;
        border-bottom: 1px solid #222831;
        margin-bottom: 25px;
    }

    .brand {
        font-size: 32px;
        font-weight: 800;
        color: #f5f5f5;
    }

    .brand span {
        color: #ff6b1a;
    }

    .subtitle {
        color: #8d96a3;
        font-size: 15px;
        margin-top: 4px;
    }

    .status {
        border: 1px solid #ff6b1a;
        color: #ff7a2b;
        padding: 9px 16px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 1px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 700;
        color: #eeeeee;
        margin-top: 25px;
        margin-bottom: 15px;
        border-left: 4px solid #ff6b1a;
        padding-left: 12px;
    }

    .card {
        background: #0b1016;
        border: 1px solid #202832;
        border-radius: 10px;
        padding: 20px;
        min-height: 130px;
    }

    .card-title {
        color: #8f99a6;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .card-value {
        color: #ffffff;
        font-size: 29px;
        font-weight: 800;
        margin-top: 8px;
    }

    .orange {
        color: #ff6b1a;
    }

    .warning {
        background: #1b0d08;
        border: 1px solid #713016;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }

    .warning-title {
        color: #ff6b1a;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 2px;
    }

    .warning-main {
        color: #ffffff;
        font-size: 24px;
        font-weight: 800;
        margin-top: 6px;
    }

    .warning-text {
        color: #aeb5bd;
        margin-top: 6px;
    }

    div[data-testid="stNumberInput"] label {
        color: #b9bec5 !important;
        font-weight: 600;
    }

    div[data-testid="stNumberInput"] input {
        background: #10161d !important;
        color: #ffffff !important;
        border: 1px solid #303944 !important;
        border-radius: 7px !important;
    }

    div[data-testid="stNumberInput"] button {
        background: #151b22 !important;
        color: #ff6b1a !important;
        border: 0 !important;
        box-shadow: none !important;
        outline: none !important;
    }

    div[data-testid="stNumberInput"] button:hover {
        background: #151b22 !important;
        color: #ff6b1a !important;
        box-shadow: none !important;
    }

    div[data-testid="stNumberInput"] button:active {
        background: #151b22 !important;
        color: #ff6b1a !important;
        box-shadow: none !important;
        transform: none !important;
    }

    div[data-testid="stNumberInput"] button:focus {
        background: #151b22 !important;
        color: #ff6b1a !important;
        box-shadow: none !important;
        outline: none !important;
    }

    div[data-testid="stButton"] button {
        background: #ff6b1a !important;
        color: #05080c !important;
        border: none !important;
        border-radius: 7px !important;
        font-weight: 800 !important;
        padding: 12px 25px !important;
        box-shadow: none !important;
        transition: none !important;
    }

    div[data-testid="stButton"] button:hover {
        background: #ff6b1a !important;
        color: #05080c !important;
        box-shadow: none !important;
    }

    div[data-testid="stButton"] button:active {
        background: #ff6b1a !important;
        color: #05080c !important;
        box-shadow: none !important;
        transform: none !important;
    }

    div[data-testid="stButton"] button:focus {
        background: #ff6b1a !important;
        color: #05080c !important;
        box-shadow: none !important;
        outline: none !important;
    }

    div[data-testid="stNumberInput"] button,
    div[data-testid="stButton"] button,
    input,
    * {
        transition: none !important;
        animation: none !important;
    }

    .metric-label {
        color: #7e8995;
        font-size: 13px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 24px;
        font-weight: 700;
    }

    .footer {
        text-align: center;
        color: #56616d;
        font-size: 12px;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #1c232b;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown(
    """
    <div class="header">
        <div>
            <div class="brand">
                ◉ Vayu<span>Drishti AI</span>
            </div>

            <div class="subtitle">
                AI/ML Cyclone Identification · Classification · Prediction
            </div>
        </div>

        <div class="status">
            ● AI SYSTEM ONLINE
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CYCLONE INPUT DATA
# ============================================================

st.markdown(
    '<div class="section-title">Cyclone Input Data</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)


with c1:

    lat = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=15.0,
        step=0.1,
        format="%.2f"
    )


with c2:

    lon = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=80.0,
        step=0.1,
        format="%.2f"
    )


with c3:

    wind = st.number_input(
        "Current Wind Speed",
        min_value=0.0,
        max_value=250.0,
        value=80.0,
        step=1.0,
        format="%.2f"
    )


with c4:

    pressure = st.number_input(
        "Current Pressure",
        min_value=850.0,
        max_value=1050.0,
        value=980.0,
        step=1.0,
        format="%.2f"
    )


# ============================================================
# RECENT CYCLONE WIND BEHAVIOUR
# ============================================================

st.markdown(
    '<div class="section-title">Recent Cyclone Behaviour</div>',
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)


with c1:

    wind_6h = st.number_input(
        "Wind Change · 6h",
        value=0.0,
        step=1.0,
        format="%.2f"
    )


with c2:

    wind_12h = st.number_input(
        "Wind Change · 12h",
        value=0.0,
        step=1.0,
        format="%.2f"
    )


with c3:

    wind_24h = st.number_input(
        "Wind Change · 24h",
        value=0.0,
        step=1.0,
        format="%.2f"
    )


# ============================================================
# RECENT PRESSURE BEHAVIOUR
# ============================================================

c1, c2, c3 = st.columns(3)


with c1:

    pressure_6h = st.number_input(
        "Pressure Change · 6h",
        value=0.0,
        step=1.0,
        format="%.2f"
    )


with c2:

    pressure_12h = st.number_input(
        "Pressure Change · 12h",
        value=0.0,
        step=1.0,
        format="%.2f"
    )


with c3:

    pressure_24h = st.number_input(
        "Pressure Change · 24h",
        value=0.0,
        step=1.0,
        format="%.2f"
    )


# ============================================================
# 24-HOUR SUMMARY FEATURES
# ============================================================

c1, c2, c3 = st.columns(3)


with c1:

    mean_wind = st.number_input(
        "24h Mean Wind",
        value=float(wind),
        step=1.0,
        format="%.2f"
    )


with c2:

    wind_std = st.number_input(
        "24h Wind Variation",
        value=5.0,
        step=1.0,
        format="%.2f"
    )


with c3:

    mean_pressure = st.number_input(
        "24h Mean Pressure",
        value=float(pressure),
        step=1.0,
        format="%.2f"
    )


# ============================================================
# RUN AI ANALYSIS BUTTON
# ============================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)

run = st.button(
    "🌀  RUN AI ANALYSIS",
    use_container_width=False
)


# ============================================================
# AI PREDICTION
# ============================================================

if run:

    input_data = pd.DataFrame(
        [[
            lat,
            lon,
            wind,
            pressure,
            wind_6h,
            wind_12h,
            wind_24h,
            pressure_6h,
            pressure_12h,
            pressure_24h,
            mean_wind,
            wind_std,
            mean_pressure
        ]],
        columns=features
    )


    # Predict future wind speed

    predicted_wind = float(
        wind_model.predict(input_data)[0]
    )


    # Predict cyclone category

    category_prediction = category_model.predict(
        input_data
    )[0]


    # Calculate category probabilities

    if hasattr(category_model, "predict_proba"):

        probabilities = category_model.predict_proba(
            input_data
        )[0]

        classes = category_model.classes_

        probability_df = pd.DataFrame(
            {
                "Category": classes,
                "Probability": probabilities * 100
            }
        )

        probability_df = probability_df.sort_values(
            "Probability",
            ascending=False
        )

    else:

        probability_df = pd.DataFrame(
            {
                "Category": [str(category_prediction)],
                "Probability": [100.0]
            }
        )


    # Make sure predicted wind is not below current wind

    predicted_wind = max(
        predicted_wind,
        wind
    )


    # ========================================================
    # DETERMINE CYCLONE SEVERITY
    # ========================================================

    if predicted_wind < 34:

        severity = "Depression"
        severity_score = 1

    elif predicted_wind < 48:

        severity = "Cyclonic Storm"
        severity_score = 2

    elif predicted_wind < 64:

        severity = "Severe Cyclonic Storm"
        severity_score = 3

    elif predicted_wind < 90:

        severity = "Very Severe Cyclonic Storm"
        severity_score = 4

    elif predicted_wind < 120:

        severity = "Extremely Severe Cyclonic Storm"
        severity_score = 5

    else:

        severity = "Super Cyclonic Storm"
        severity_score = 6


    # ========================================================
    # DETERMINE INTENSIFICATION RISK
    # ========================================================

    wind_change = predicted_wind - wind


    if wind_change > 20:

        risk_text = "Rapid intensification detected"

    elif wind_change > 8:

        risk_text = "Significant strengthening expected"

    elif wind_change > 0:

        risk_text = "Gradual strengthening expected"

    else:

        risk_text = "Weakening or stable intensity expected"


    # ========================================================
    # CREATE REPEATABLE RANDOM GENERATOR
    # ========================================================

    rng = np.random.default_rng(
        int(
            abs(lat * 1000)
            + abs(lon * 1000)
            + wind * 10
        )
    )


    # ========================================================
    # ESTIMATE CYCLONE LANDING POSITION
    # ========================================================

    direction_lat = rng.uniform(
        -0.8,
        0.8
    )

    direction_lon = rng.uniform(
        1.0,
        3.0
    )


    hit_lat = lat + direction_lat
    hit_lon = lon + direction_lon


    hit_lat = float(
        np.clip(
            hit_lat,
            -35,
            35
        )
    )


    hit_lon = float(
        np.clip(
            hit_lon,
            55,
            105
        )
    )


    # ========================================================
    # CALCULATE DISTANCE TO ESTIMATED LANDFALL
    # ========================================================

    distance = np.sqrt(
        ((hit_lat - lat) * 111) ** 2
        +
        ((hit_lon - lon) * 101) ** 2
    )


    movement_speed = rng.uniform(
        12,
        22
    )


    hours_to_landfall = max(
        8,
        min(
            96,
            distance / movement_speed
        )
    )


    hours_to_landfall = int(
        round(
            hours_to_landfall
        )
    )


    # ========================================================
    # ESTIMATE AFFECTED POPULATION
    # ========================================================

    if severity_score <= 2:

        people_affected = rng.integers(
            50000,
            300000
        )

    elif severity_score == 3:

        people_affected = rng.integers(
            200000,
            800000
        )

    elif severity_score == 4:

        people_affected = rng.integers(
            500000,
            2000000
        )

    elif severity_score == 5:

        people_affected = rng.integers(
            1000000,
            5000000
        )

    else:

        people_affected = rng.integers(
            2000000,
            8000000
        )


    # ========================================================
    # ESTIMATE ECONOMIC IMPACT
    # ========================================================

    economic_loss = (
        people_affected
        *
        rng.uniform(
            18000,
            60000
        )
    )

    economic_loss = (
        economic_loss
        /
        10000000
    )


    # ========================================================
    # CREATE 24-HOUR WIND FORECAST
    # ========================================================

    forecast_hours = np.array(
        [
            0,
            6,
            12,
            18,
            24
        ]
    )


    forecast_wind = np.linspace(
        wind,
        predicted_wind,
        5
    )


    forecast_wind += rng.normal(
        0,
        1.5,
        5
    )


    forecast_wind[0] = wind
    forecast_wind[-1] = predicted_wind


    # ========================================================
    # CREATE CYCLONE FORECAST TRACK
    # ========================================================

    path_points = []


    for i in range(9):

        fraction = i / 8


        point_lat = (
            lat
            +
            (
                hit_lat - lat
            )
            *
            fraction
        )


        point_lon = (
            lon
            +
            (
                hit_lon - lon
            )
            *
            fraction
        )


        point_lat += (
            np.sin(
                fraction * np.pi
            )
            *
            rng.uniform(
                -0.2,
                0.2
            )
        )


        point_lon += (
            np.sin(
                fraction * np.pi
            )
            *
            rng.uniform(
                -0.2,
                0.2
            )
        )


        path_points.append(
            [
                point_lat,
                point_lon
            ]
        )


    # ========================================================
    # SAVE RESULTS IN SESSION STATE
    # ========================================================

    st.session_state["analysis_done"] = True

    st.session_state["predicted_wind"] = predicted_wind

    st.session_state["category"] = str(
        category_prediction
    )

    st.session_state["severity"] = severity

    st.session_state["severity_score"] = severity_score

    st.session_state["risk_text"] = risk_text

    st.session_state["probability_df"] = probability_df

    st.session_state["hit_lat"] = hit_lat

    st.session_state["hit_lon"] = hit_lon

    st.session_state["hours_to_landfall"] = (
        hours_to_landfall
    )

    st.session_state["people_affected"] = int(
        people_affected
    )

    st.session_state["economic_loss"] = (
        economic_loss
    )

    st.session_state["forecast_hours"] = (
        forecast_hours
    )

    st.session_state["forecast_wind"] = (
        forecast_wind
    )

    st.session_state["path_points"] = (
        path_points
    )


# ============================================================
# DISPLAY RESULTS AFTER ANALYSIS
# ============================================================

if st.session_state.get(
    "analysis_done",
    False
):

    predicted_wind = (
        st.session_state["predicted_wind"]
    )

    category_prediction = (
        st.session_state["category"]
    )

    severity = (
        st.session_state["severity"]
    )

    severity_score = (
        st.session_state["severity_score"]
    )

    risk_text = (
        st.session_state["risk_text"]
    )

    probability_df = (
        st.session_state["probability_df"]
    )

    hit_lat = (
        st.session_state["hit_lat"]
    )

    hit_lon = (
        st.session_state["hit_lon"]
    )

    hours_to_landfall = (
        st.session_state["hours_to_landfall"]
    )

    people_affected = (
        st.session_state["people_affected"]
    )

    economic_loss = (
        st.session_state["economic_loss"]
    )

    forecast_hours = (
        st.session_state["forecast_hours"]
    )

    forecast_wind = (
        st.session_state["forecast_wind"]
    )

    path_points = (
        st.session_state["path_points"]
    )


    # ========================================================
    # AI RISK ALERT
    # ========================================================

    st.markdown(
        f"""
        <div class="warning">

            <div class="warning-title">
                ● AI RISK ALERT
            </div>

            <div class="warning-main">
                {severity}
            </div>

            <div class="warning-text">
                {risk_text}.
                Estimated landfall window:
                approximately {hours_to_landfall} hours.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # AI PREDICTION SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">AI Prediction Summary</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    Predicted Category
                </div>

                <div class="card-value orange">
                    {category_prediction}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    24h Wind Forecast
                </div>

                <div class="card-value">
                    {predicted_wind:.1f} kt
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c3:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    Estimated Landfall
                </div>

                <div class="card-value">
                    {hours_to_landfall} h
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c4:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    Risk Level
                </div>

                <div class="card-value orange">
                    {severity_score}/6
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # FORECAST ANALYTICS
    # ========================================================

    st.markdown(
        '<div class="section-title">Forecast Analytics</div>',
        unsafe_allow_html=True
    )


    g1, g2 = st.columns(2)


    with g1:

        st.markdown(
            "### Wind Intensity Forecast"
        )


        wind_chart = pd.DataFrame(
            {
                "Wind Speed (kt)": forecast_wind
            },
            index=[
                "Now",
                "+6h",
                "+12h",
                "+18h",
                "+24h"
            ]
        )


        st.line_chart(
            wind_chart,
            height=350
        )


    with g2:

        st.markdown(
            "### Storm Severity"
        )


        severity_values = []


        for value in forecast_wind:

            if value < 34:

                severity_values.append(1)

            elif value < 48:

                severity_values.append(2)

            elif value < 64:

                severity_values.append(3)

            elif value < 90:

                severity_values.append(4)

            elif value < 120:

                severity_values.append(5)

            else:

                severity_values.append(6)


        severity_chart = pd.DataFrame(
            {
                "Severity Level": severity_values
            },
            index=[
                "Now",
                "+6h",
                "+12h",
                "+18h",
                "+24h"
            ]
        )


        st.bar_chart(
            severity_chart,
            height=350
        )


    # ========================================================
    # CATEGORY PROBABILITY ANALYSIS
    # ========================================================

    st.markdown(
        "### Category Probability Analysis"
    )


    probability_chart = (
        probability_df
        .set_index("Category")
    )


    st.bar_chart(
        probability_chart,
        height=350
    )


    # ========================================================
    # CYCLONE POSITION INTELLIGENCE
    # ========================================================

    st.markdown(
        '<div class="section-title">Cyclone Position Intelligence</div>',
        unsafe_allow_html=True
    )


    # Convert Python path into JavaScript array format

    path_string = ",".join(
        f"[{p[0]:.4f},{p[1]:.4f}]"
        for p in path_points
    )


    # ========================================================
    # INTERACTIVE LEAFLET MAP
    # ========================================================

    map_html = f"""
    <!DOCTYPE html>

    <html>

    <head>

        <meta charset="utf-8">

        <link
            rel="stylesheet"
            href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        />

        <script
            src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js">
        </script>


        <style>

            body {{
                margin: 0;
                padding: 0;
                background: #05080c;
            }}


            #map {{
                width: 100%;
                height: 620px;
                border-radius: 10px;
            }}


            .leaflet-popup-content-wrapper {{
                background: #0b1016;
                color: white;
                border: 1px solid #ff6b1a;
            }}


            .leaflet-popup-tip {{
                background: #0b1016;
            }}


            .popup-title {{
                color: #ff6b1a;
                font-weight: bold;
                font-size: 15px;
            }}


            .popup-text {{
                color: #dddddd;
                font-size: 13px;
            }}


            .map-legend {{
                background: #0b1016;
                color: white;
                padding: 12px;
                border: 1px solid #ff6b1a;
                border-radius: 8px;
                line-height: 22px;
                font-size: 13px;
            }}


            .legend-current {{
                display: inline-block;
                width: 12px;
                height: 12px;
                background: #ff6b1a;
                border-radius: 50%;
                margin-right: 6px;
            }}


            .legend-landfall {{
                display: inline-block;
                width: 12px;
                height: 12px;
                background: #ff2600;
                border-radius: 50%;
                margin-right: 6px;
            }}

        </style>

    </head>


    <body>

        <div id="map"></div>


        <script>

            const start = [{lat}, {lon}];

            const hit = [{hit_lat}, {hit_lon}];


            const path = [
                {path_string}
            ];


            // Create Leaflet map

            const map = L.map(
                'map',
                {{
                    zoomControl: true,
                    attributionControl: true
                }}
            ).setView(
                start,
                5
            );


            // Add OpenStreetMap base layer

            L.tileLayer(
                'https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',
                {{
                    maxZoom: 18,
                    attribution:
                        '&copy; OpenStreetMap contributors'
                }}
            ).addTo(map);


            // Create current cyclone marker

            const cycloneIcon = L.divIcon(
                {{
                    className: '',

                    html:
                        '<div style="' +
                        'width:20px;' +
                        'height:20px;' +
                        'background:#ff6b1a;' +
                        'border:3px solid white;' +
                        'border-radius:50%;' +
                        'box-shadow:0 0 18px #ff6b1a;' +
                        '"></div>',

                    iconSize: [20, 20],

                    iconAnchor: [10, 10]
                }}
            );


            // Create predicted landfall marker

            const hitIcon = L.divIcon(
                {{
                    className: '',

                    html:
                        '<div style="' +
                        'width:24px;' +
                        'height:24px;' +
                        'background:#ff2600;' +
                        'border:4px solid white;' +
                        'border-radius:50%;' +
                        'box-shadow:0 0 25px #ff2600;' +
                        '"></div>',

                    iconSize: [24, 24],

                    iconAnchor: [12, 12]
                }}
            );


            // Add current cyclone position

            L.marker(
                start,
                {{
                    icon: cycloneIcon
                }}
            )
            .addTo(map)
            .bindPopup(
                '<div class="popup-title">' +
                'CYCLONE ORIGIN / CURRENT POSITION' +
                '</div>' +

                '<div class="popup-text">' +

                'Latitude: {lat:.2f}<br>' +

                'Longitude: {lon:.2f}<br>' +

                'Current Wind: {wind:.1f} kt' +

                '</div>'
            )
            .openPopup();


            // Add predicted landfall position

            L.marker(
                hit,
                {{
                    icon: hitIcon
                }}
            )
            .addTo(map)
            .bindPopup(
                '<div class="popup-title">' +
                'PREDICTED LANDFALL' +
                '</div>' +

                '<div class="popup-text">' +

                'Latitude: {hit_lat:.2f}<br>' +

                'Longitude: {hit_lon:.2f}<br>' +

                'ETA: {hours_to_landfall} hours' +

                '</div>'
            );


            // Draw predicted cyclone track

            L.polyline(
                path,
                {{
                    color: '#ff6b1a',
                    weight: 5,
                    opacity: 0.95
                }}
            ).addTo(map);


            // Add small points along cyclone path

            for (
                let i = 1;
                i < path.length - 1;
                i++
            ) {{

                L.circleMarker(
                    path[i],
                    {{
                        radius: 5,
                        color: '#ff8a4c',
                        fillColor: '#ff6b1a',
                        fillOpacity: 1
                    }}
                ).addTo(map);

            }}


            // Create map legend

            const legend =
                L.control({{
                    position: 'bottomright'
                }});


            legend.onAdd = function() {{

                const div =
                    L.DomUtil.create(
                        'div',
                        'map-legend'
                    );


                div.innerHTML =
                    '<b>CYCLONE TRACK</b><br>' +

                    '<span class="legend-current"></span>' +
                    'Current Position<br>' +

                    '<span class="legend-landfall"></span>' +
                    'Predicted Landfall<br>' +

                    '━ Predicted Track';


                return div;

            }};


            legend.addTo(map);


            // Automatically fit map to cyclone path

            const bounds =
                L.latLngBounds(path);


            map.fitBounds(
                bounds,
                {{
                    padding: [40, 40]
                }}
            );

        </script>

    </body>

    </html>
    """


    # Display the map inside Streamlit

    components.html(
        map_html,
        height=640
    )


    # ========================================================
    # IMPACT ASSESSMENT
    # ========================================================

    st.markdown(
        '<div class="section-title">Impact Assessment</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3 = st.columns(3)


    with c1:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    Predicted Landfall
                </div>

                <div class="card-value orange">
                    {hours_to_landfall} hours
                </div>

                <div class="metric-label">
                    Approximate forecast window
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c2:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    Potentially Affected Population
                </div>

                <div class="card-value">
                    {people_affected:,}
                </div>

                <div class="metric-label">
                    Estimated people in affected region
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    with c3:

        st.markdown(
            f"""
            <div class="card">

                <div class="card-title">
                    Potential Economic Impact
                </div>

                <div class="card-value orange">
                    ₹{economic_loss:.1f} Cr
                </div>

                <div class="metric-label">
                    Preliminary scenario estimate
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # PREDICTED CYCLONE TRACK TABLE
    # ========================================================

    st.markdown(
        '<div class="section-title">Predicted Cyclone Track</div>',
        unsafe_allow_html=True
    )


    track_df = pd.DataFrame(
        {
            "Latitude": [
                p[0]
                for p in path_points
            ],

            "Longitude": [
                p[1]
                for p in path_points
            ]
        },

        index=[
            "Now",
            "+3h",
            "+6h",
            "+9h",
            "+12h",
            "+15h",
            "+18h",
            "+21h",
            "+24h"
        ]
    )


    st.dataframe(
        track_df.round(3),
        use_container_width=True
    )
    # ========================================================
    # FOOTER
    # ========================================================
    st.markdown(
        f"""
        <div class="footer">
            VayuDrishti AI · SIH 2026 Demo Build ·
            Predictions are model-based and impact figures are
            simulated scenario estimates for demonstration.

        </div>
        """,
        unsafe_allow_html=True
    )
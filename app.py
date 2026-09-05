import os
import textwrap
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="VayuDrishti AI",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# MODEL LOADING
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "cyclone_models")


@st.cache_resource
def load_models():
    wind_model = joblib.load(os.path.join(MODEL_DIR, "future_wind_model.pkl"))
    category_model = joblib.load(os.path.join(MODEL_DIR, "cyclone_category_model.pkl"))
    model_features = joblib.load(os.path.join(MODEL_DIR, "model_features.pkl"))

    if isinstance(model_features, dict):
        features = model_features.get("features", [])
    else:
        features = list(model_features)

    return wind_model, category_model, features


wind_model, category_model, features = load_models()

EXPECTED_FEATURES = [
    "Latitude",
    "Longitude",
    "Wind",
    "Pressure",
    "Wind_6h",
    "Wind_12h",
    "Wind_24h",
    "Pressure_6h",
    "Pressure_12h",
    "Pressure_24h",
    "Mean_Wind",
    "Wind_Std",
    "Mean_Pressure",
]

# Use the feature order stored with the model, while keeping a safe fallback.
if not features:
    features = EXPECTED_FEATURES


# ============================================================
# HELPERS
# ============================================================
def render_html(html):
    st.html(textwrap.dedent(html).strip())


def severity_from_wind(value):
    value = float(value)
    if value < 34:
        return "Depression", 1, "Organized tropical system with relatively low wind intensity."
    if value < 48:
        return "Cyclonic Storm", 2, "A developing cyclone with stronger sustained winds."
    if value < 64:
        return "Severe Cyclonic Storm", 3, "A significantly stronger system requiring increased monitoring."
    if value < 90:
        return "Very Severe Cyclonic Storm", 4, "A high-intensity cyclone capable of causing major disruption."
    if value < 120:
        return "Extremely Severe Cyclonic Storm", 5, "An extremely intense system with potential for severe impacts."
    return "Super Cyclonic Storm", 6, "An exceptionally intense cyclone requiring extreme caution."


def feature_importance(model, names):
    """Return model feature importance/absolute coefficient values when available."""
    estimator = model
    if hasattr(model, "named_steps"):
        try:
            estimator = list(model.named_steps.values())[-1]
        except Exception:
            estimator = model

    values = None
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim == 1:
            values = np.abs(coef)
        else:
            values = np.mean(np.abs(coef), axis=0)

    if values is None:
        return pd.DataFrame({"Feature": names, "Importance": np.zeros(len(names))})

    n = min(len(values), len(names))
    df = pd.DataFrame({
        "Feature": list(names)[:n],
        "Importance": values[:n],
    })
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)


def model_name(model):
    estimator = model
    if hasattr(model, "named_steps"):
        try:
            estimator = list(model.named_steps.values())[-1]
        except Exception:
            pass
    return estimator.__class__.__name__


def safe_model_input(values):
    """Build the exact feature order expected by the saved model."""
    row = dict(values)
    # Support common alternate feature naming conventions.
    aliases = {
        "lat": "Latitude",
        "lon": "Longitude",
        "wind": "Wind",
        "pressure": "Pressure",
        "wind_6h": "Wind_6h",
        "wind_12h": "Wind_12h",
        "wind_24h": "Wind_24h",
        "pressure_6h": "Pressure_6h",
        "pressure_12h": "Pressure_12h",
        "pressure_24h": "Pressure_24h",
        "mean_wind": "Mean_Wind",
        "wind_std": "Wind_Std",
        "mean_pressure": "Mean_Pressure",
    }
    for old, new in aliases.items():
        if old in row and new not in row:
            row[new] = row[old]

    return pd.DataFrame([[row.get(col, 0.0) for col in features]], columns=features)


def make_map(origin_lat, origin_lon, hit_lat, hit_lon, origin_wind, hours, path_points):
    path_string = ",".join(
        f"[{float(p[0]):.5f},{float(p[1]):.5f}]" for p in path_points
    )

    map_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
      <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
      <style>
        html, body {{
            margin: 0;
            padding: 0;
            background: #07111a;
        }}
        #map {{
            width: 100%;
            height: 590px;
            border-radius: 12px;
            overflow: hidden;
        }}
        .leaflet-popup-content-wrapper {{
            background: #0b1622;
            color: #f4f7fa;
            border: 1px solid #42d6ca;
        }}
        .leaflet-popup-tip {{
            background: #0b1622;
        }}
        .popup-title {{
            color: #42d6ca;
            font-weight: 800;
            font-size: 14px;
            letter-spacing: .8px;
        }}
        .popup-text {{
            color: #d8e1e8;
            font-size: 12px;
            line-height: 1.7;
        }}
        .map-legend {{
            background: rgba(7,17,26,.94);
            color: #e7edf2;
            padding: 12px 14px;
            border: 1px solid #284252;
            border-radius: 9px;
            line-height: 22px;
            font-size: 12px;
            box-shadow: 0 8px 30px rgba(0,0,0,.35);
        }}
        .dot-current {{
            display:inline-block;width:11px;height:11px;border-radius:50%;
            background:#42d6ca;border:2px solid white;margin-right:6px;
        }}
        .dot-landfall {{
            display:inline-block;width:11px;height:11px;border-radius:50%;
            background:#ff4d4d;border:2px solid white;margin-right:6px;
        }}
      </style>
    </head>
    <body>
      <div id="map"></div>
      <script>
        const start = [{origin_lat}, {origin_lon}];
        const hit = [{hit_lat}, {hit_lon}];
        const path = [{path_string}];

        const map = L.map("map", {{
            zoomControl: true,
            attributionControl: true
        }}).setView(start, 5);

        L.tileLayer(
            "https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png",
            {{
                maxZoom: 18,
                attribution: "&copy; OpenStreetMap contributors"
            }}
        ).addTo(map);

        const currentIcon = L.divIcon({{
            className: "",
            html: '<div style="width:20px;height:20px;background:#42d6ca;border:3px solid white;border-radius:50%;box-shadow:0 0 20px #42d6ca;"></div>',
            iconSize: [20,20],
            iconAnchor: [10,10]
        }});

        const hitIcon = L.divIcon({{
            className: "",
            html: '<div style="width:24px;height:24px;background:#ff4d4d;border:4px solid white;border-radius:50%;box-shadow:0 0 25px #ff4d4d;"></div>',
            iconSize: [24,24],
            iconAnchor: [12,12]
        }});

        L.marker(start, {{icon: currentIcon}})
          .addTo(map)
          .bindPopup(
            '<div class="popup-title">CURRENT CYCLONE POSITION</div>' +
            '<div class="popup-text">Latitude: {origin_lat:.2f}<br>' +
            'Longitude: {origin_lon:.2f}<br>' +
            'Wind: {origin_wind:.1f} kt</div>'
          );

        L.marker(hit, {{icon: hitIcon}})
          .addTo(map)
          .bindPopup(
            '<div class="popup-title">PREDICTED LANDFALL</div>' +
            '<div class="popup-text">Latitude: {hit_lat:.2f}<br>' +
            'Longitude: {hit_lon:.2f}<br>' +
            'ETA: {hours} hours</div>'
          );

        L.polyline(path, {{
            color: "#42d6ca",
            weight: 5,
            opacity: .95
        }}).addTo(map);

        for (let i = 1; i < path.length - 1; i++) {{
            L.circleMarker(path[i], {{
                radius: 5,
                color: "#ffffff",
                weight: 2,
                fillColor: "#42d6ca",
                fillOpacity: 1
            }}).addTo(map);
        }}

        L.circle(hit, {{
            radius: 45000,
            color: "#ff4d4d",
            weight: 2,
            fillColor: "#ff4d4d",
            fillOpacity: .12
        }}).addTo(map);

        const legend = L.control({{position:"bottomright"}});
        legend.onAdd = function() {{
            const div = L.DomUtil.create("div", "map-legend");
            div.innerHTML =
                "<b>CYCLONE TRACK</b><br>" +
                '<span class="dot-current"></span>Current Position<br>' +
                '<span class="dot-landfall"></span>Predicted Landfall<br>' +
                "━ Predicted Track";
            return div;
        }};
        legend.addTo(map);

        map.fitBounds(L.latLngBounds(path), {{
            padding: [55,55],
            maxZoom: 7
        }});
      </script>
    </body>
    </html>
    """
    components.html(map_html, height=610)


# ============================================================
# SESSION STATE
# ============================================================
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# ============================================================
# GLOBAL CSS — VAYUDRISHTI STYLE
# ============================================================
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background:#06111a !important;
    }
    .stApp {
        background:
          radial-gradient(circle at 15% 0%, rgba(66,214,202,.09), transparent 27%),
          radial-gradient(circle at 92% 16%, rgba(40,120,160,.10), transparent 30%),
          #06111a !important;
        color:#f5f8fa !important;
    }
    .block-container {
        max-width:1450px;
        padding-top:1.4rem;
        padding-bottom:4rem;
    }
    .header {
        display:flex; justify-content:space-between; align-items:center;
        padding:6px 0 22px; border-bottom:1px solid #20303b;
    }
    .brand {font-size:31px;font-weight:800;color:#f5f8fa;}
    .brand-accent {color:#42d6ca;}
    .subtitle {color:#91a5b4;font-size:14px;margin-top:4px;}
    .top-meta {text-align:right;color:#7892a4;font-size:12px;line-height:1.8;}
    .online {color:#42d6ca;font-weight:800;letter-spacing:1.5px;}
    .section-title {
        font-size:20px;font-weight:750;color:#dfe9ef;
        margin:25px 0 14px;padding-left:11px;
        border-left:4px solid #42d6ca;
    }
    .panel {
        background:#0b1824;border:1px solid #203746;border-radius:11px;
        padding:18px 20px;
    }
    .card {
        background:#0b1824;border:1px solid #203746;border-radius:11px;
        padding:18px;min-height:118px;
    }
    .card-title {color:#7792a4;font-size:11px;letter-spacing:1.4px;text-transform:uppercase;}
    .card-value {color:#f5f8fa;font-size:26px;font-weight:800;margin-top:8px;}
    .teal {color:#42d6ca !important;}
    .muted {color:#8299a9;font-size:12px;line-height:1.55;}
    .alert {
        background:linear-gradient(90deg,rgba(89,22,34,.52),rgba(32,16,26,.8));
        border:1px solid #78333f;border-radius:11px;padding:19px 22px;margin:18px 0;
    }
    .alert-label {color:#ff6464;font-weight:850;letter-spacing:2px;font-size:12px;}
    .alert-main {color:#f7f9fb;font-size:25px;font-weight:850;margin-top:7px;}
    .alert-text {color:#c0cbd3;font-size:13px;line-height:1.65;margin-top:7px;}
    .readout {
        background:#0b1824;border:1px solid #203746;border-radius:11px;padding:17px 20px;
    }
    .readout-title {font-size:15px;font-weight:750;color:#b9d0de;margin-bottom:14px;}
    .readout-row {display:flex;justify-content:space-between;border-bottom:1px solid #1b2c38;padding:8px 0;}
    .readout-label {color:#8099a9;font-size:12px;}
    .readout-value {color:#42d6ca;font-weight:750;font-family:monospace;}
    .pipeline {
        display:flex;gap:9px;overflow-x:auto;padding:4px 0 8px;
    }
    .stage {
        flex:1;min-width:145px;text-align:center;padding:10px 8px;
        border:1px solid #213847;border-radius:8px;background:#0a1722;
        color:#7992a2;font-size:12px;font-weight:700;
    }
    .stage.active {
        background:rgba(66,214,202,.16);
        border-color:#42d6ca;color:#e9fffd;
        box-shadow:0 0 18px rgba(66,214,202,.10);
    }
    .stage.current {
        background:rgba(255,77,77,.13);
        border-color:#ff4d4d;color:#fff;
    }
    .stButton > button {
        background:#42d6ca !important;color:#06111a !important;
        border:0 !important;border-radius:8px !important;
        font-weight:850 !important;
    }
    div[data-testid="stNumberInput"] input {
        background:#0d1a26 !important;color:#f4f8fa !important;
        border:1px solid #2a414f !important;
    }
    div[data-testid="stWidgetLabel"] p, div[data-testid="stNumberInput"] label {
        color:#9bb0be !important;font-weight:600 !important;
    }
    .small-tag {
        display:inline-block;padding:5px 9px;border-radius:20px;
        border:1px solid #284655;color:#83a0b0;font-size:11px;margin-right:6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================
render_html(
    """
    <div class="header">
      <div>
        <div class="brand">◉ Vayu<span class="brand-accent">Drishti</span> AI</div>
        <div class="subtitle">AI/ML cyclone identification, classification & prediction — SIH 2026</div>
      </div>
      <div class="top-meta">
        <span class="online">● AI SYSTEM ONLINE</span><br>
        Bay of Bengal · North Indian Ocean basin<br>
        Random Forest classifier + intensity regressor
      </div>
    </div>
    """
)

# ============================================================
# DYNAMIC SEVERITY PIPELINE
# ============================================================
preview_wind = float(st.session_state.get("predicted_wind", st.session_state.get("wind_input", 80.0)))
preview_severity, preview_score, _ = severity_from_wind(preview_wind)

st.session_state.preview_wind = preview_wind

st.markdown('<div class="section-title">Cyclone Severity Pipeline</div>', unsafe_allow_html=True)

stage_names = [
    "Depression",
    "Cyclonic Storm",
    "Severe CS",
    "Very Severe CS",
    "Extremely Severe",
    "Super Cyclonic Storm",
]
stage_html = []
for i, name in enumerate(stage_names, start=1):
    cls = "stage"
    if i < preview_score:
        cls += " active"
    elif i == preview_score:
        cls += " current"
    stage_html.append(f'<div class="{cls}">{name}</div>')

render_html('<div class="pipeline">' + "".join(stage_html) + "</div>")

# ============================================================
# SATELLITE FEED PANEL
# ============================================================
st.markdown('<div class="panel">', unsafe_allow_html=True)
feed_col1, feed_col2 = st.columns([3, 1])

with feed_col1:
    render_html(
        """
        <div style="font-size:16px;font-weight:750;color:#bdd3df;">Live satellite feed</div>
        <div class="muted" style="margin-top:5px;">
        Load a CSV of time-stamped cyclone observations or use the manual readout below.
        The loaded feed can be stepped through row-by-row.
        </div>
        """
    )

with feed_col2:
    feed_file = st.file_uploader("Load CSV feed", type=["csv"], label_visibility="collapsed")

if feed_file is not None:
    try:
        feed_df = pd.read_csv(feed_file)
        st.session_state.feed_df = feed_df
        st.success(f"Feed loaded: {len(feed_df)} observations")
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")

if "feed_df" in st.session_state:
    feed_df = st.session_state.feed_df
    if len(feed_df):
        idx = st.slider("Feed observation", 0, len(feed_df) - 1, 0)
        st.dataframe(feed_df.iloc[[idx]], use_container_width=True, hide_index=True)
else:
    st.info("No feed loaded — using manual cyclone readout below.")

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# MANUAL READOUT — VISUAL TELEMETRY
# ============================================================
st.markdown('<div class="section-title">Satellite-derived readout</div>', unsafe_allow_html=True)

lat_preview = st.session_state.get("lat_input", 15.0)
lon_preview = st.session_state.get("lon_input", 80.0)
wind_preview = st.session_state.get("wind_input", 80.0)
pressure_preview = st.session_state.get("pressure_input", 980.0)

r1, r2 = st.columns([1.4, 1.0])

with r1:
    dvorak = st.slider("Dvorak T-number", 1.0, 7.0, min(7.0, max(1.0, wind_preview / 18.0)), 0.1)
    ir_temp = st.slider("IR cloud-top brightness temp (K)", 170, 280, int(max(170, min(280, 255 - wind_preview * 0.65))), 1)
    eye_diameter = st.slider("Eye diameter (km)", 0, 100, int(max(0, min(100, wind_preview * 0.27))), 1)
    cdo_symmetry = st.slider("CDO symmetry index", 0.0, 1.0, min(1.0, max(0.0, 0.35 + wind_preview / 220.0)), 0.01)
    sst = st.slider("Sea surface temperature (°C)", 20.0, 34.0, 29.5, 0.1)
    shear = st.slider("Vertical wind shear (kt)", 0.0, 50.0, 8.0, 1.0)

with r2:
    render_html(
        f"""
        <div class="readout">
          <div class="readout-title">CURRENT READOUT</div>
          <div class="readout-row"><span class="readout-label">Dvorak T-number</span><span class="readout-value">{dvorak:.1f}</span></div>
          <div class="readout-row"><span class="readout-label">IR brightness</span><span class="readout-value">{ir_temp} K</span></div>
          <div class="readout-row"><span class="readout-label">Eye diameter</span><span class="readout-value">{eye_diameter} km</span></div>
          <div class="readout-row"><span class="readout-label">CDO symmetry</span><span class="readout-value">{cdo_symmetry:.2f}</span></div>
          <div class="readout-row"><span class="readout-label">SST</span><span class="readout-value">{sst:.1f} °C</span></div>
          <div class="readout-row"><span class="readout-label">Wind shear</span><span class="readout-value">{shear:.0f} kt</span></div>
        </div>
        """
    )

# ============================================================
# MODEL INPUTS
# ============================================================
st.markdown('<div class="section-title">Model Input Data</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    lat = st.number_input("Latitude", -90.0, 90.0, float(lat_preview), 0.1, key="lat_input", format="%.2f")
with c2:
    lon = st.number_input("Longitude", -180.0, 180.0, float(lon_preview), 0.1, key="lon_input", format="%.2f")
with c3:
    wind = st.number_input("Current Wind Speed (kt)", 0.0, 250.0, float(wind_preview), 1.0, key="wind_input", format="%.2f")
with c4:
    pressure = st.number_input("Current Pressure (hPa)", 850.0, 1050.0, float(pressure_preview), 1.0, key="pressure_input", format="%.2f")

c1, c2, c3 = st.columns(3)
with c1:
    wind_6h = st.number_input("Wind Change · 6h", -100.0, 100.0, 0.0, 1.0)
with c2:
    wind_12h = st.number_input("Wind Change · 12h", -100.0, 100.0, 0.0, 1.0)
with c3:
    wind_24h = st.number_input("Wind Change · 24h", -100.0, 100.0, 0.0, 1.0)

c1, c2, c3 = st.columns(3)
with c1:
    pressure_6h = st.number_input("Pressure Change · 6h", -100.0, 100.0, 0.0, 1.0)
with c2:
    pressure_12h = st.number_input("Pressure Change · 12h", -100.0, 100.0, 0.0, 1.0)
with c3:
    pressure_24h = st.number_input("Pressure Change · 24h", -100.0, 100.0, 0.0, 1.0)

c1, c2, c3 = st.columns(3)
with c1:
    mean_wind = st.number_input("24h Mean Wind", 0.0, 250.0, float(wind), 1.0)
with c2:
    wind_std = st.number_input("24h Wind Variation", 0.0, 100.0, 5.0, 1.0)
with c3:
    mean_pressure = st.number_input("24h Mean Pressure", 850.0, 1050.0, float(pressure), 1.0)

# ============================================================
# ANALYSIS BUTTON
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
run = st.button("🌀  RUN AI ANALYSIS", use_container_width=True)

if run:
    input_values = {
        "Latitude": lat,
        "Longitude": lon,
        "Wind": wind,
        "Pressure": pressure,
        "Wind_6h": wind_6h,
        "Wind_12h": wind_12h,
        "Wind_24h": wind_24h,
        "Pressure_6h": pressure_6h,
        "Pressure_12h": pressure_12h,
        "Pressure_24h": pressure_24h,
        "Mean_Wind": mean_wind,
        "Wind_Std": wind_std,
        "Mean_Pressure": mean_pressure,
    }
    st.session_state["input_values"] = input_values

    try:
        input_data = safe_model_input(input_values)

        predicted_wind = float(wind_model.predict(input_data)[0])
        predicted_wind = max(predicted_wind, float(wind))

        category_prediction = category_model.predict(input_data)[0]

        if hasattr(category_model, "predict_proba"):
            probabilities = category_model.predict_proba(input_data)[0]
            classes = category_model.classes_
            probability_df = pd.DataFrame({
                "Category": [str(x) for x in classes],
                "Probability": probabilities * 100,
            }).sort_values("Probability", ascending=False)
        else:
            probability_df = pd.DataFrame({
                "Category": [str(category_prediction)],
                "Probability": [100.0],
            })

        severity, severity_score, severity_description = severity_from_wind(predicted_wind)
        wind_change = predicted_wind - wind

        if wind_change > 20:
            risk_text = "Rapid intensification detected"
        elif wind_change > 8:
            risk_text = "Significant strengthening expected"
        elif wind_change > 0:
            risk_text = "Gradual strengthening expected"
        elif wind_change < -8:
            risk_text = "Cyclone weakening expected"
        else:
            risk_text = "Cyclone intensity appears relatively stable"

        # Repeatable demo track. The ML prediction is real; track/impact values
        # are explicitly marked as demonstration estimates.
        seed_value = int(abs(lat * 1000) + abs(lon * 1000) + wind * 10 + pressure)
        rng = np.random.default_rng(seed_value)

        direction_lat = rng.uniform(-0.8, 0.8)
        direction_lon = rng.uniform(1.0, 3.0)

        hit_lat = float(np.clip(lat + direction_lat, -35, 35))
        hit_lon = float(np.clip(lon + direction_lon, 55, 105))

        distance = np.sqrt(((hit_lat - lat) * 111) ** 2 + ((hit_lon - lon) * 101) ** 2)
        movement_speed = rng.uniform(12, 22)
        hours_to_landfall = int(round(max(8, min(96, distance / movement_speed))))

        if severity_score <= 2:
            people_affected = int(rng.integers(50_000, 300_000))
        elif severity_score == 3:
            people_affected = int(rng.integers(200_000, 800_000))
        elif severity_score == 4:
            people_affected = int(rng.integers(500_000, 2_000_000))
        elif severity_score == 5:
            people_affected = int(rng.integers(1_000_000, 5_000_000))
        else:
            people_affected = int(rng.integers(2_000_000, 8_000_000))

        economic_loss = people_affected * rng.uniform(18_000, 60_000) / 10_000_000

        forecast_hours = np.array([0, 6, 12, 18, 24])
        forecast_wind = np.linspace(wind, predicted_wind, 5)
        forecast_wind += rng.normal(0, 1.5, 5)
        forecast_wind[0] = wind
        forecast_wind[-1] = predicted_wind

        path_points = []
        for i in range(9):
            fraction = i / 8
            point_lat = lat + (hit_lat - lat) * fraction
            point_lon = lon + (hit_lon - lon) * fraction
            point_lat += np.sin(fraction * np.pi) * rng.uniform(-0.2, 0.2)
            point_lon += np.sin(fraction * np.pi) * rng.uniform(-0.2, 0.2)
            path_points.append([float(point_lat), float(point_lon)])

        st.session_state.analysis_done = True
        st.session_state.predicted_wind = predicted_wind
        st.session_state.category = str(category_prediction)
        st.session_state.severity = severity
        st.session_state.severity_score = severity_score
        st.session_state.severity_description = severity_description
        st.session_state.risk_text = risk_text
        st.session_state.probability_df = probability_df
        st.session_state.origin_lat = lat
        st.session_state.origin_lon = lon
        st.session_state.origin_wind = wind
        st.session_state.hit_lat = hit_lat
        st.session_state.hit_lon = hit_lon
        st.session_state.hours_to_landfall = hours_to_landfall
        st.session_state.people_affected = people_affected
        st.session_state.economic_loss = economic_loss
        st.session_state.forecast_hours = forecast_hours
        st.session_state.forecast_wind = forecast_wind
        st.session_state.path_points = path_points

    except Exception as exc:
        st.session_state.analysis_done = False
        st.error(f"Model prediction failed: {exc}")


# ============================================================
# RESULTS
# ============================================================
if st.session_state.analysis_done:
    predicted_wind = st.session_state.predicted_wind
    category_prediction = st.session_state.category
    severity = st.session_state.severity
    severity_score = st.session_state.severity_score
    severity_description = st.session_state.severity_description
    risk_text = st.session_state.risk_text
    probability_df = st.session_state.probability_df
    origin_lat = st.session_state.origin_lat
    origin_lon = st.session_state.origin_lon
    origin_wind = st.session_state.origin_wind
    hit_lat = st.session_state.hit_lat
    hit_lon = st.session_state.hit_lon
    hours_to_landfall = st.session_state.hours_to_landfall
    people_affected = st.session_state.people_affected
    economic_loss = st.session_state.economic_loss
    forecast_hours = st.session_state.forecast_hours
    forecast_wind = st.session_state.forecast_wind
    path_points = st.session_state.path_points
    wind_change = predicted_wind - origin_wind

    # -------------------- ALERT --------------------
    render_html(
        f"""
        <div class="alert">
          <div class="alert-label">● RED ALERT</div>
          <div class="alert-main">{severity} — forecast window {hours_to_landfall}h</div>
          <div class="alert-text">
            {risk_text}. Estimated landfall within approximately {hours_to_landfall} hours.
            Use this dashboard as an AI decision-support demo, not as an official warning.
          </div>
        </div>
        """
    )

    # -------------------- SUMMARY CARDS --------------------
    st.markdown('<div class="section-title">AI Prediction Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_html(f'<div class="card"><div class="card-title">Classification</div><div class="card-value teal">{category_prediction}</div><div class="muted">Model output category</div></div>')
    with c2:
        render_html(f'<div class="card"><div class="card-title">24h Intensity Forecast</div><div class="card-value">{predicted_wind:.1f} kt</div><div class="muted">{wind_change:+.1f} kt change</div></div>')
    with c3:
        render_html(f'<div class="card"><div class="card-title">Estimated Landfall</div><div class="card-value">{hours_to_landfall} h</div><div class="muted">Demonstration forecast window</div></div>')
    with c4:
        render_html(f'<div class="card"><div class="card-title">Estimated Impact</div><div class="card-value">₹{economic_loss:,.1f} Cr</div><div class="muted">Scenario estimate</div></div>')

    # -------------------- SEVERITY INTELLIGENCE --------------------
    st.markdown('<div class="section-title">Cyclone Severity Intelligence</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_html(f'<div class="card"><div class="card-title">Forecast Wind</div><div class="card-value teal">{predicted_wind:.1f} kt</div><div class="muted">Maximum model-predicted intensity</div></div>')
    with c2:
        render_html(f'<div class="card"><div class="card-title">Severity Index</div><div class="card-value">{severity_score}/6</div><div class="muted">{severity}</div></div>')
    with c3:
        render_html(f'<div class="card"><div class="card-title">Intensity Change</div><div class="card-value">{wind_change:+.1f} kt</div><div class="muted">{risk_text}</div></div>')
    with c4:
        render_html(f'<div class="card"><div class="card-title">Affected Population</div><div class="card-value">{people_affected:,}</div><div class="muted">Demonstration estimate</div></div>')

    render_html(
        f"""
        <div class="panel" style="margin-top:14px;">
          <div class="card-title">SEVERITY ASSESSMENT</div>
          <div style="font-size:24px;font-weight:850;color:#42d6ca;margin-top:7px;">{severity}</div>
          <div class="muted" style="margin-top:5px;">{severity_description}</div>
        </div>
        """
    )

    # -------------------- ANALYTICS --------------------
    st.markdown('<div class="section-title">Model & Forecast Analytics</div>', unsafe_allow_html=True)
    g1, g2 = st.columns(2)

    chart_index = ["Now", "+6h", "+12h", "+18h", "+24h"]
    wind_chart = pd.DataFrame({"Wind Speed (kt)": forecast_wind}, index=chart_index)

    with g1:
        st.markdown("**Wind Intensity Forecast**")
        st.line_chart(wind_chart, height=330)

    with g2:
        st.markdown("**Storm Severity Forecast**")
        severity_values = [severity_from_wind(x)[1] for x in forecast_wind]
        severity_chart = pd.DataFrame({"Severity Level": severity_values}, index=chart_index)
        st.bar_chart(severity_chart, height=330)

    st.markdown("**Cyclone Category Probability**")
    probability_chart = probability_df.set_index("Category")
    st.bar_chart(probability_chart, height=300)

    # -------------------- MODEL FEATURE IMPORTANCE --------------------
    st.markdown('<div class="section-title">Model Feature Intelligence</div>', unsafe_allow_html=True)
    fi1, fi2 = st.columns([1.45, 1])

    wind_fi = feature_importance(wind_model, features)
    category_fi = feature_importance(category_model, features)

    with fi1:
        st.markdown(f"**{model_name(wind_model)} — intensity feature importance**")
        if wind_fi["Importance"].sum() > 0:
            st.bar_chart(wind_fi.head(10).set_index("Feature"), height=360)
        else:
            st.info("The saved intensity model does not expose feature_importances_ or coefficients.")

    with fi2:
        st.markdown(f"**{model_name(category_model)} — classifier feature importance**")
        if category_fi["Importance"].sum() > 0:
            st.bar_chart(category_fi.head(10).set_index("Feature"), height=360)
        else:
            st.info("The saved classifier does not expose feature_importances_ or coefficients.")

    st.markdown("**Model input feature table**")
    current_inputs = st.session_state.get("input_values", {})
    feature_table = pd.DataFrame({
        "Feature": features,
        "Current Value": [current_inputs.get(f, np.nan) for f in features],
    })
    st.dataframe(feature_table, use_container_width=True, hide_index=True)

    # -------------------- MAP --------------------
    st.markdown('<div class="section-title">Cyclone Track & Map Intelligence</div>', unsafe_allow_html=True)
    map_c1, map_c2 = st.columns([2.5, 1])

    with map_c1:
        make_map(
            origin_lat, origin_lon,
            hit_lat, hit_lon,
            origin_wind, hours_to_landfall,
            path_points,
        )

    with map_c2:
        render_html(
            f"""
            <div class="panel">
              <div class="card-title">CURRENT POSITION</div>
              <div class="card-value teal">{origin_lat:.2f}, {origin_lon:.2f}</div>
              <div class="muted">Lat / Lon</div>
              <br>
              <div class="card-title">PREDICTED LANDFALL</div>
              <div class="card-value">{hit_lat:.2f}, {hit_lon:.2f}</div>
              <div class="muted">Demonstration estimate</div>
              <br>
              <div class="card-title">LANDFALL WINDOW</div>
              <div class="card-value teal">{hours_to_landfall} h</div>
              <div class="muted">Approximate track ETA</div>
              <br>
              <div class="card-title">AFFECTED POPULATION</div>
              <div class="card-value">{people_affected:,}</div>
              <div class="muted">Scenario estimate</div>
            </div>
            """
        )

    # -------------------- TRACK TABLE --------------------
    st.markdown('<div class="section-title">Predicted Cyclone Track</div>', unsafe_allow_html=True)
    track_df = pd.DataFrame(
        {
            "Latitude": [p[0] for p in path_points],
            "Longitude": [p[1] for p in path_points],
        },
        index=["Now", "+3h", "+6h", "+9h", "+12h", "+15h", "+18h", "+21h", "+24h"],
    )
    st.dataframe(track_df.round(3), use_container_width=True)

    # -------------------- SYSTEM NOTE --------------------
    render_html(
        """
        <div class="panel" style="margin-top:18px;">
          <div class="card-title">SYSTEM NOTE</div>
          <div class="muted" style="margin-top:7px;">
            Wind intensity and cyclone category are generated from the trained ML models.
            The forecast track, landfall location, affected population and economic impact
            are demonstration estimates and should not be treated as official meteorological warnings.
          </div>
        </div>
        """
    )

render_html(
    """
    <div style="text-align:center;color:#536b7b;font-size:11px;margin-top:40px;padding-top:18px;border-top:1px solid #1b2c38;">
      VayuDrishti AI · SIH 2026 · AI/ML Cyclone Intelligence Demo
    </div>
    """
)

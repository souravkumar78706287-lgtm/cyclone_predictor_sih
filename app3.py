import base64
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
BG_GIF_PATH = os.path.join(BASE_DIR, "assets", "cyclone_bg.gif")
BG_IMAGE_URL = "https://images.unsplash.com/photo-1534088568595-a066f410bcda?auto=format&fit=crop&w=2400&q=75"


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


@st.cache_data
def load_background_gif(path):
    """Read the pre-processed (cropped/blurred/faded) satellite gif and
    return it as a base64 string so it can be dropped straight into CSS as a
    data URI — no separate static file server needed for a Streamlit app."""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


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


def advisory_zone(score):
    if score >= 5:
        return "RED", "#FF0055", "Emergency action"
    if score >= 3:
        return "ORANGE", "#FFB000", "Prepare and stay alert"
    return "GREEN", "#39FF14", "Monitor official updates"


def make_map(origin_lat, origin_lon, hit_lat, hit_lon, origin_wind, hours, path_points, severity_score):
    zone_name, zone_color, _ = advisory_zone(severity_score)
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
            background: #0D1117;
        }}
        #map {{
            width: 100%;
            height: 590px;
            border-radius: 12px;
            overflow: hidden;
        }}
        .leaflet-popup-content-wrapper {{
            background: #0D1117;
            color: #f4f7fa;
            border: 1px solid #00F5D4;
        }}
        .leaflet-popup-tip {{
            background: #0D1117;
        }}
        .popup-title {{
            color: #00F5D4;
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
            background:#00F5D4;border:2px solid white;margin-right:6px;
        }}
        .dot-landfall {{
            display:inline-block;width:11px;height:11px;border-radius:50%;
            background:{zone_color};border:2px solid white;margin-right:6px;
        }}
        .zone-dot {{
            display:inline-block;width:10px;height:10px;border-radius:50%;
            border:1px solid white;margin-right:6px;
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
            html: '<div style="width:20px;height:20px;background:#00F5D4;border:3px solid white;border-radius:50%;box-shadow:0 0 20px #00F5D4;"></div>',
            iconSize: [20,20],
            iconAnchor: [10,10]
        }});

        const hitIcon = L.divIcon({{
            className: "",
            html: '<div style="width:24px;height:24px;background:{zone_color};border:4px solid white;border-radius:50%;box-shadow:0 0 25px {zone_color};"></div>',
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
            color: "{zone_color}",
            weight: 5,
            opacity: .95
        }}).addTo(map);

        for (let i = 1; i < path.length - 1; i++) {{
            L.circleMarker(path[i], {{
                radius: 5,
                color: "#ffffff",
                weight: 2,
                fillColor: "{zone_color}",
                fillOpacity: 1
            }}).addTo(map);
        }}

        L.circle(hit, {{
            radius: 45000,
            color: "{zone_color}",
            weight: 2,
            fillColor: "{zone_color}",
            fillOpacity: .12
        }}).addTo(map);

        L.circle(hit, {{radius: 150000, color: "#FFB000", weight: 2, fillColor: "#FFB000", fillOpacity: .04}}).addTo(map);
        L.circle(hit, {{radius: 260000, color: "#39FF14", weight: 2, fillColor: "#39FF14", fillOpacity: .03}}).addTo(map);

        const legend = L.control({{position:"bottomright"}});
        legend.onAdd = function() {{
            const div = L.DomUtil.create("div", "map-legend");
            div.innerHTML =
                "<b>CYCLONE TRACK</b><br>" +
                '<span class="dot-current"></span>Current Position<br>' +
                '<span class="dot-landfall"></span>{zone_name} landfall zone<br>' +
                '<span class="zone-dot" style="background:#39FF14"></span>Green: monitor<br>' +
                '<span class="zone-dot" style="background:#FFB000"></span>Orange: prepare<br>' +
                '<span class="zone-dot" style="background:#FF0055"></span>Red: emergency<br>' +
                "━ Forecast track";
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
        background:#0B132B !important;
    }
    .stApp {
        background:
          radial-gradient(circle at 15% 0%, rgba(66,214,202,.09), transparent 27%),
          radial-gradient(circle at 92% 16%, rgba(40,120,160,.10), transparent 30%),
          #0B132B !important;
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
    .brand-accent {color:#00F5D4;}
    .subtitle {color:#8FA9C2;font-size:14px;margin-top:4px;}
    .top-meta {text-align:right;color:#7898B5;font-size:12px;line-height:1.8;}
    .online {color:#00F5D4;font-weight:800;letter-spacing:1.5px;}
    .section-title {
        font-size:20px;font-weight:750;color:#dfe9ef;
        margin:25px 0 14px;padding-left:11px;
        border-left:4px solid #00F5D4;
    }
    .panel {
        background:#0D1117;border:1px solid #1D3850;border-radius:11px;
        padding:18px 20px;
    }
    .card {
        background:#0D1117;border:1px solid #1D3850;border-radius:11px;
        padding:18px;min-height:118px;
    }
    .card-title {color:#7792a4;font-size:11px;letter-spacing:1.4px;text-transform:uppercase;}
    .card-value {color:#f5f8fa;font-size:26px;font-weight:800;margin-top:8px;}
    .teal {color:#00F5D4 !important;}
    .muted {color:#91A9BD;font-size:12px;line-height:1.55;}
    .alert {
        background:linear-gradient(90deg,rgba(255,0,85,.16),rgba(13,17,23,.92));
        border:1px solid #FF0055;border-radius:11px;padding:19px 22px;margin:18px 0;
    }
    .alert-label {color:#ff6464;font-weight:850;letter-spacing:2px;font-size:12px;}
    .alert-main {color:#f7f9fb;font-size:25px;font-weight:850;margin-top:7px;}
    .alert-text {color:#c0cbd3;font-size:13px;line-height:1.65;margin-top:7px;}
    .readout {
        background:#0D1117;border:1px solid #1D3850;border-radius:11px;padding:17px 20px;
    }
    .readout-title {font-size:15px;font-weight:750;color:#b9d0de;margin-bottom:14px;}
    .readout-row {display:flex;justify-content:space-between;border-bottom:1px solid #1b2c38;padding:8px 0;}
    .readout-label {color:#8099a9;font-size:12px;}
    .readout-value {color:#00F5D4;font-weight:750;font-family:monospace;}
    .pipeline {
        display:flex;gap:9px;overflow-x:auto;padding:4px 0 8px;
    }
    .stage {
        flex:1;min-width:145px;text-align:center;padding:10px 8px;
        border:1px solid #1D3850;border-radius:8px;background:#0D1117;
        color:#7992a2;font-size:12px;font-weight:700;
    }
    .stage.active {
        background:rgba(0,245,212,.14);
        border-color:#00F5D4;color:#e9fffd;
        box-shadow:0 0 18px rgba(0,245,212,.18);
    }
    .stage.current {
        background:rgba(255,0,85,.16);
        border-color:#FF0055;color:#fff;
    }
    .stButton > button {
        background:#00F5D4 !important;color:#0B132B !important;
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

# ------------------------------------------------------------
# Faded satellite background with a dark readability overlay.
# ------------------------------------------------------------
background_image = BG_IMAGE_URL
if os.path.exists(BG_GIF_PATH):
    _bg_b64 = load_background_gif(BG_GIF_PATH)
    background_image = f"data:image/gif;base64,{_bg_b64}"

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
          linear-gradient(rgba(11,19,43,.78), rgba(13,17,23,.90)),
          url("{background_image}") center / cover no-repeat fixed,
          #0B132B !important;
    }}
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
st.markdown('<div class="panel csv-feed-panel">', unsafe_allow_html=True)
feed_col1, feed_col2 = st.columns([3, 1])

with feed_col1:
    render_html(
        """
        <div style="font-size:16px;font-weight:750;color:#bdd3df;">Live satellite feed</div>
        <div class="muted" style="margin-top:5px;">
        Upload a CSV of time-stamped cyclone observations.
        The selected row will provide the model inputs automatically.
        </div>
        """
    )

with feed_col2:
    feed_file = st.file_uploader("Upload cyclone CSV", type=["csv"])

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
    st.info("Upload a CSV to enable model analysis.")

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# CSV-DRIVEN MODEL INPUTS
# ============================================================
feed_row = {}
feed_ready = "feed_df" in st.session_state and not st.session_state.feed_df.empty
if feed_ready:
    feed_df = st.session_state.feed_df
    selected_row = feed_df.iloc[-1].to_dict()
    feed_row = {str(key).strip().lower().replace(" ", "_"): value for key, value in selected_row.items()}

def csv_value(*names, default=0.0):
    for name in names:
        value = feed_row.get(name.lower().replace(" ", "_"))
        if value is not None and pd.notna(value):
            return float(value)
    return float(default)

lat = csv_value("lat", "latitude", default=15.0)
lon = csv_value("lon", "longitude", default=80.0)
wind = csv_value("wind", "wmo_wind", "current_wind_speed", default=80.0)
pressure = csv_value("pressure", "wmo_pres", "current_pressure", default=980.0)
wind_6h = csv_value("wind_6h", "wind_change_6h")
wind_12h = csv_value("wind_12h", "wind_change_12h")
wind_24h = csv_value("wind_24h", "wind_change_24h")
pressure_6h = csv_value("pressure_6h", "pressure_change_6h")
pressure_12h = csv_value("pressure_12h", "pressure_change_12h")
pressure_24h = csv_value("pressure_24h", "pressure_change_24h")
mean_wind = csv_value("mean_wind", "wind_mean_24h", default=wind)
wind_std = csv_value("wind_std", "wind_std_24h", default=5.0)
mean_pressure = csv_value("mean_pressure", "pressure_mean_24h", default=pressure)

if feed_ready:
    st.caption("Model inputs are taken automatically from the latest CSV observation.")

# ============================================================
# ANALYSIS BUTTON
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
run = st.button("🌀  RUN AI ANALYSIS", use_container_width=True, disabled=not feed_ready)

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
            path_points, severity_score,
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

    # -------------------- IMD-STYLE ACTION GUIDE --------------------
    zone_name, zone_color, zone_action = advisory_zone(severity_score)
    st.markdown('<div class="section-title">What To Follow In Each Zone</div>', unsafe_allow_html=True)
    render_html(
                f"""
                <div class="panel">
                    <div class="muted" style="margin-bottom:12px;">
                        IMD-style advisory bands for this demonstration. Always follow IMD and local authority instructions.
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
                        <div style="border-left:4px solid #39FF14;padding:12px;background:rgba(57,255,20,.08);">
                            <div style="color:#39FF14;font-weight:850;">GREEN · MONITOR</div>
                            <div class="muted">Check official updates, keep emergency contacts ready, and avoid unverified alerts.</div>
                        </div>
                        <div style="border-left:4px solid #FFB000;padding:12px;background:rgba(255,176,0,.10);">
                            <div style="color:#FFB000;font-weight:850;">ORANGE · PREPARE</div>
                            <div class="muted">Secure loose items, charge devices, prepare medicines and supplies, and plan a safe route.</div>
                        </div>
                        <div style="border-left:4px solid #FF0055;padding:12px;background:rgba(255,0,85,.10);">
                            <div style="color:#FF0055;font-weight:850;">RED · ACT NOW</div>
                            <div class="muted">Follow evacuation orders, move away from coasts and flood zones, and do not travel during the storm.</div>
                        </div>
                    </div>
                    <div style="margin-top:14px;color:{zone_color};font-weight:850;">CURRENT MODEL BAND: {zone_name} · {zone_action}</div>
                </div>
                """
    )

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
# 🌀 VayuDrishti AI — Cyclone Prediction & Intelligence System

> An AI/ML-powered cyclone intelligence platform for predicting cyclone intensity, category, severity, movement and potential impact.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=for-the-badge&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?style=for-the-badge&logo=numpy)
![Status](https://img.shields.io/badge/Status-Prototype-success?style=for-the-badge)

---

## 🌪️ Overview

**VayuDrishti AI** is a machine-learning based cyclone intelligence and forecasting prototype designed to analyze cyclone conditions and provide an easy-to-understand prediction dashboard.

The system takes cyclone observations such as:

- Latitude
- Longitude
- Wind speed
- Atmospheric pressure
- Wind changes over different time periods
- Pressure changes over different time periods
- 24-hour wind statistics
- 24-hour pressure statistics

and uses trained machine-learning models to estimate cyclone intensity and category.

The application then presents the results through an interactive dashboard containing:

- 🌀 Cyclone category prediction
- 💨 Future wind forecast
- ⚠️ AI risk assessment
- 📊 Severity analysis
- 🗺️ Predicted cyclone track
- 📍 Estimated landfall location
- ⏱️ Estimated time to landfall
- 👥 Potentially affected population
- 💰 Potential economic impact
- 📈 Forecast charts
- 🌍 Interactive geographical visualization

---

# 🎯 Problem Statement

Tropical cyclones can rapidly intensify and cause severe damage to coastal and vulnerable regions.

Traditional cyclone monitoring requires analyzing multiple meteorological variables simultaneously. This can make it difficult to quickly interpret the severity and possible impact of an approaching system.

VayuDrishti AI aims to provide a simple intelligence layer that combines historical cyclone data, machine learning and visualization into a single dashboard.

The goal is not to replace professional meteorological forecasting, but to demonstrate how AI/ML can assist in cyclone analysis and decision-support systems.

---

# 🚀 Key Features

## 🧠 AI-Based Cyclone Prediction

The application uses trained machine-learning models to predict cyclone characteristics from the supplied atmospheric conditions.

The current prediction pipeline contains:

- Future wind prediction model
- Cyclone category classification model

---

## 💨 Wind Intensity Forecast

The model estimates future cyclone wind intensity using meteorological and historical behaviour features.

Example inputs include:

```text
Current Wind
Wind Change 6h
Wind Change 12h
Wind Change 24h
Pressure Change 6h
Pressure Change 12h
Pressure Change 24h
24h Mean Wind
24h Wind Variation
24h Mean Pressure

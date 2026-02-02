st.subheader("Key findings")
st.markdown(
    """
    - Weather relates to daily 311 volume: extreme temperatures and precipitation days tend to coincide with higher call volumes; relationships are directional but not perfectly linear.
    - Duplicate/near-duplicate records are common in raw 311 exports; consolidation to incident-level rows may improve reliability.
    - Temporal patterns are strong: clear day-of-week and hourly peaks; these are useful and stable predictors in short-term forecasting.
    - Forecasting with a simple linear model (trend + day-of-week + recent baseline + naive weather) produces a modest fit (see R²). Treat forecasts as operational guidance rather than precise predictions.
    """
)

st.subheader("Operational recommendations")
st.markdown(
    """
    - Use day-of-week and recent baseline (rolling average) to drive staffing schedules; pre-stage crews for days with forecasted extreme weather.
    - Implement event-driven dispatch rules triggered by simple weather thresholds (heavy rain, heat waves, high wind) to reduce time-to-first-response.
    - Monitor high-impact complaint types (top 5) with dedicated triage to reduce repeat work and mean resolution time.
    - Improve geocoding and datetime hygiene in ingestion to reduce routing errors and improve spatial analyses.
    """
)

st.subheader("Data & tooling improvements to boost forecasting accuracy")
st.markdown(
    """
    - Higher-frequency and richer weather forecasts: integrate hourly probabilistic forecasts (e.g., NWS / commercial APIs) instead of naive recent averages.
    - Real-time incident status and work order data (time-to-first-response, crews dispatched, resolution categories) to model operational capacity and feedback effects.
    - Additional exogenous signals: transit incidents, public events, building maintenance schedules, water/sewer outage feeds, utility outage data, holidays/school closures.
    - Improve location precision: standardized geocoding, building IDs (BIN/BBL), and parcel-level joins to enable hyper-local weather/impact modelling.
    - Use more expressive models: gradient-boosted trees or time-series models (Prophet, SARIMAX, ETS, or ensemble ML + GLM) with cross-validation and probabilistic forecasting to capture nonlinear/weather-interaction effects.
    - Automated evaluation pipeline: backtesting, rolling-origin validation, calibration metrics, and alert thresholds for operational use.
    """
)

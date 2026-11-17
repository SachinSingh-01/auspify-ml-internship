import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# 1. Load saved artifacts
BASE_DIR = Path(__file__).resolve().parent
artifacts = joblib.load(BASE_DIR / "netflix_inference_artifacts.pkl")

model = artifacts["model"]
encoder = artifacts["encoder"]
feature_columns = artifacts["feature_columns"]

# Extract learned categories directly from the encoder for dropdowns
countries = sorted(encoder.categories_[0].tolist())
ratings = sorted(encoder.categories_[1].tolist())

# 2. App UI Layout
st.set_page_config(page_title="Netflix Content Predictor", page_icon="🎬")
st.title("🎬 Netflix Content Type Predictor")
st.write("Predict whether a title is a **Movie** or a **TV Show** based on its metadata.")

col1, col2, col3 = st.columns(3)

with col1:
    selected_country = st.selectbox(
        "Production Country",
        countries,
        index=countries.index("United States") if "United States" in countries else 0
    )

with col2:
    selected_rating = st.selectbox(
        "Maturity Rating",
        ratings,
        index=ratings.index("TV-MA") if "TV-MA" in ratings else 0
    )

with col3:
    selected_year = st.number_input(
        "Release Year",
        min_value=1925,
        max_value=2026,
        value=2021,
        step=1
    )

# 3. Preprocess and Predict on Button Click
if st.button("Predict Content Type", type="primary"):
    input_df = pd.DataFrame({
        "country": [selected_country],
        "rating": [selected_rating],
        "release_year": [selected_year]
    })

    # Encode categorical features
    encoded_data = encoder.transform(input_df[["country", "rating"]])
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(["country", "rating"])
    )

    # Combine with release_year and align columns
    processed_df = pd.concat([encoded_df, input_df[["release_year"]]], axis=1)
    processed_df = processed_df.reindex(columns=feature_columns, fill_value=0)

    # Generate prediction and probabilities
    prediction = model.predict(processed_df)[0]
    probabilities = model.predict_proba(processed_df)[0]
    prob_dict = dict(zip(model.classes_, probabilities))

    # Display Results
    st.divider()
    st.subheader(f"Prediction: **{prediction}**")

    prob_col1, prob_col2 = st.columns(2)
    prob_col1.metric("Movie Probability", f"{prob_dict['Movie'] * 100:.1f}%")
    prob_col2.metric("TV Show Probability", f"{prob_dict['TV Show'] * 100:.1f}%")
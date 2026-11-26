import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
artifacts = joblib.load(BASE_DIR / "rating_inference_artifacts.pkl")

model = artifacts["model"]
encoder = artifacts["encoder"]
mlb = artifacts["mlb"]
top_countries = artifacts["top_countries"]
feature_columns = artifacts["feature_columns"]

st.set_page_config(page_title="Netflix Rating Predictor", page_icon="🎯")
st.title("🎯 Netflix Audience Rating Predictor")
st.write("Predict the audience maturity rating (`TV-MA`, `TV-14`, `PG-13`, `TV-Y7`, etc.) from title metadata.")

col1, col2 = st.columns(2)
with col1:
    content_type = st.selectbox("Content Type", ["Movie", "TV Show"])
    country = st.selectbox("Production Country", sorted(top_countries + ["Other"]))
    release_year = st.number_input("Release Year", min_value=1925, max_value=2026, value=2021)

with col2:
    duration_label = "Duration (Minutes)" if content_type == "Movie" else "Duration (Seasons)"
    duration_num = st.number_input(duration_label, min_value=1, max_value=300, value=95 if content_type == "Movie" else 1)
    selected_genres = st.multiselect("Genres", sorted(mlb.classes_.tolist()), default=["Dramas"])

if st.button("Predict Audience Rating", type="primary"):
    if not selected_genres:
        st.warning("Please select at least one genre.")
    else:
        country_clean = country if country in top_countries else "Other"
        cat_encoded = encoder.transform(pd.DataFrame({"type": [content_type], "country_clean": [country_clean]}))
        cat_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(["type", "country_clean"]))

        genre_encoded = mlb.transform([selected_genres])
        genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)

        num_df = pd.DataFrame({"release_year": [release_year], "duration_num": [float(duration_num)]})
        final_input = pd.concat([num_df, cat_df, genre_df], axis=1).reindex(columns=feature_columns, fill_value=0)

        pred = model.predict(final_input)[0]
        probs = model.predict_proba(final_input)[0]
        top3_idx = np.argsort(probs)[::-1][:3]

        st.divider()
        st.subheader(f"Predicted Rating: **{pred}**")
        cols = st.columns(3)
        for i, idx in enumerate(top3_idx):
            cols[i].metric(model.classes_[idx], f"{probs[idx] * 100:.1f}%")
            
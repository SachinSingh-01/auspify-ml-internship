
import joblib
import pandas as pd


# 1. Load saved inference artifacts
artifacts = joblib.load("netflix_inference_artifacts.pkl")

model = artifacts["model"]
encoder = artifacts["encoder"]
feature_columns = artifacts["feature_columns"]


# 2. Preprocessing function
def preprocess_new_data(data):

    required_columns = [
        "country",
        "rating",
        "release_year"
    ]

    # Check required columns
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    # Select required columns
    data = data[required_columns].copy()

    # Check missing values
    if data.isnull().any().any():
        raise ValueError("Input contains missing values.")

    # Convert release_year to numeric
    data["release_year"] = pd.to_numeric(
        data["release_year"],
        errors="raise"
    )

    # Encode categorical features
    encoded_data = encoder.transform(
        data[["country", "rating"]]
    )

    encoded_columns = encoder.get_feature_names_out(
        ["country", "rating"]
    )

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoded_columns,
        index=data.index
    )

    # Add numerical feature
    processed_data = pd.concat(
        [
            encoded_df,
            data[["release_year"]]
        ],
        axis=1
    )

    # Maintain original feature order
    processed_data = processed_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return processed_data


# 3. Create sample raw Netflix records
new_netflix_data = pd.DataFrame({
    "country": ["India", "South Korea"],
    "rating": ["TV-14", "TV-MA"],
    "release_year": [2024, 2023]
})


# 4. Preprocess raw records
processed_data = preprocess_new_data(
    new_netflix_data
)


# 5. Generate predictions
predictions = model.predict(processed_data)


# 6. Display results
print("\nNetflix Content Classification Results")
print("-" * 45)

for i, prediction in enumerate(predictions):

    print(
        f"Netflix Record {i + 1}: {prediction}"
    )

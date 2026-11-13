
import joblib
import pandas as pd
from pathlib import Path


# ==========================================
# 1. LOAD SAVED INFERENCE ARTIFACTS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

artifacts = joblib.load(
    BASE_DIR / "netflix_inference_artifacts.pkl"
)

model = artifacts["model"]
encoder = artifacts["encoder"]
feature_columns = artifacts["feature_columns"]

print("Inference artifacts loaded successfully!")
print("Number of features:", len(feature_columns))


# ==========================================
# 2. PREPROCESSING FUNCTION
# ==========================================

def preprocess_new_data(data):

    # Required input columns
    required_columns = [
        "country",
        "rating",
        "release_year"
    ]

    # Check missing columns
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(
                f"Missing required column: {col}"
            )

    # Select columns in correct order
    data = data[required_columns].copy()

    # Check missing values
    if data.isnull().any().any():
        raise ValueError(
            "Input data contains missing values."
        )

    # Convert release year to numeric
    data["release_year"] = pd.to_numeric(
        data["release_year"],
        errors="raise"
    )

    # Apply fitted encoder
    encoded_data = encoder.transform(
        data[["country", "rating"]]
    )

    # Get encoded feature names
    encoded_columns = encoder.get_feature_names_out(
        ["country", "rating"]
    )

    # Convert encoded data into DataFrame
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



# ==========================================
# 3. LOAD VALIDATION CSV
# ==========================================

validation_data = pd.read_csv(
    BASE_DIR / "validation_samples.csv"
)

# Separate actual labels from input features
actual_labels = validation_data.pop("actual")


# ==========================================
# 4. PREPROCESS VALIDATION DATA
# ==========================================

processed_data = preprocess_new_data(
    validation_data
)

print("\nPreprocessing completed!")
print("Processed data shape:", processed_data.shape)


# ==========================================
# 5. GENERATE PREDICTIONS
# ==========================================

predictions = model.predict(processed_data)


# ==========================================
# 6. COMPARE ACTUAL AND PREDICTED LABELS
# ==========================================

results = pd.DataFrame({
    "Actual": actual_labels.values,
    "Predicted": predictions
})

results["Match"] = (
    results["Actual"] == results["Predicted"]
)

print("\nValidation Results")
print("=" * 50)

print(results.to_string(index=False))

print("\nCorrect predictions:", results["Match"].sum())
print("Total records:", len(results))

print(
    "Sample Accuracy:",
    results["Match"].mean()
)

print("\nValidation completed!")

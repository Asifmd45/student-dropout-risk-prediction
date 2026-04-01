from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = Path("Models")
MODEL_PATH = MODEL_DIR / "logistic_regression_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"

RAW_FEATURES = [
    "school",
    "sex",
    "age",
    "address",
    "famsize",
    "Pstatus",
    "Medu",
    "Fedu",
    "Mjob",
    "Fjob",
    "reason",
    "guardian",
    "traveltime",
    "studytime",
    "failures",
    "schoolsup",
    "famsup",
    "paid",
    "activities",
    "nursery",
    "higher",
    "internet",
    "romantic",
    "famrel",
    "freetime",
    "goout",
    "Dalc",
    "Walc",
    "health",
    "absences",
    "G1",
]


DEFAULT_STUDENT = {
    "school": "GP",
    "sex": "F",
    "age": 17,
    "address": "U",
    "famsize": "GT3",
    "Pstatus": "T",
    "Medu": 2,
    "Fedu": 2,
    "Mjob": "other",
    "Fjob": "other",
    "reason": "course",
    "guardian": "mother",
    "traveltime": 2,
    "studytime": 2,
    "failures": 0,
    "schoolsup": "no",
    "famsup": "yes",
    "paid": "no",
    "activities": "yes",
    "nursery": "yes",
    "higher": "yes",
    "internet": "yes",
    "romantic": "no",
    "famrel": 3,
    "freetime": 3,
    "goout": 3,
    "Dalc": 1,
    "Walc": 1,
    "health": 3,
    "absences": 4,
    "G1": 10,
}


def risk_category(score: float) -> str:
    if score >= 0.7:
        return "High Risk"
    if score >= 0.4:
        return "Medium Risk"
    return "Low Risk"


def patch_model_compat(model):
    # Some sklearn version combinations can deserialize LogisticRegression
    # without this attribute, which breaks predict_proba.
    if model.__class__.__name__ == "LogisticRegression" and not hasattr(model, "multi_class"):
        model.multi_class = "auto"
    return model


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    model = patch_model_compat(model)
    return model, scaler, feature_columns


def preprocess_input(raw_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    encoded = pd.get_dummies(raw_df, drop_first=True)
    encoded = encoded.reindex(columns=feature_columns, fill_value=0)
    return encoded


def predict_risk(raw_df: pd.DataFrame, model, scaler, feature_columns: list[str]) -> pd.DataFrame:
    prepared = preprocess_input(raw_df, feature_columns)
    scaled = scaler.transform(prepared)
    probs = model.predict_proba(scaled)[:, 1]
    preds = model.predict(scaled)

    result = raw_df.copy()
    result["risk_score"] = probs.round(4)
    result["prediction"] = preds
    result["risk_level"] = result["risk_score"].apply(risk_category)
    return result


def build_single_input_form() -> dict:
    st.subheader("Single Student Prediction")
    st.caption("Using 5 key inputs. Remaining features are auto-filled with baseline values.")

    col1, col2 = st.columns(2)
    with col1:
        studytime = st.slider("Study Time (1-4)", min_value=1, max_value=4, value=DEFAULT_STUDENT["studytime"])
        absences = st.number_input("Absences", min_value=0, max_value=100, value=DEFAULT_STUDENT["absences"])
        g1 = st.slider("First Period Grade (G1)", 0, 20, DEFAULT_STUDENT["G1"])

    with col2:
        failures = st.slider("Past Class Failures", 0, 4, DEFAULT_STUDENT["failures"])
        goout = st.slider(
            "Going Out With Friends (1-5)",
            min_value=1,
            max_value=5,
            value=DEFAULT_STUDENT["goout"],
        )

    student = DEFAULT_STUDENT.copy()
    student.update(
        {
            "studytime": studytime,
            "absences": int(absences),
            "G1": g1,
            "failures": failures,
            "goout": goout,
        }
    )
    return student


def validate_batch_columns(batch_df: pd.DataFrame) -> tuple[bool, list[str]]:
    required = set(RAW_FEATURES)
    present = set(batch_df.columns)
    missing = sorted(required - present)
    return len(missing) == 0, missing


def main():
    st.set_page_config(page_title="Student Dropout Risk Predictor", layout="wide")
    st.title("Student Dropout Risk Predictor")
    st.caption("Predict at-risk students using your trained Logistic Regression model.")

    missing_files = [
        str(path) for path in [MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH] if not path.exists()
    ]
    if missing_files:
        st.error("Missing model artifact files:")
        for path in missing_files:
            st.write(f"- {path}")
        st.stop()

    model, scaler, feature_columns = load_artifacts()

    tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])

    with tab_single:
        student_data = build_single_input_form()
        if st.button("Predict Dropout Risk", type="primary"):
            input_df = pd.DataFrame([student_data])
            result = predict_risk(input_df, model, scaler, feature_columns)
            score = float(result.loc[0, "risk_score"])
            level = str(result.loc[0, "risk_level"])

            st.metric("Risk Score", f"{score:.2%}")
            if level == "High Risk":
                st.error(f"Risk Level: {level}")
            elif level == "Medium Risk":
                st.warning(f"Risk Level: {level}")
            else:
                st.success(f"Risk Level: {level}")

            st.dataframe(result[["risk_score", "prediction", "risk_level"]], use_container_width=True)

    with tab_batch:
        st.write("Upload a CSV containing raw student features for batch risk prediction.")
        st.write("Required columns must include all training features except G2 and G3.")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file, sep=None, engine="python")
            # Ignore label/unused fields if present.
            batch_df = batch_df.drop(columns=["dropout", "G2", "G3"], errors="ignore")

            is_valid, missing = validate_batch_columns(batch_df)
            if not is_valid:
                st.error("Missing required columns in uploaded CSV:")
                st.write(missing)
            else:
                predictions = predict_risk(batch_df[RAW_FEATURES], model, scaler, feature_columns)
                st.success(f"Predictions generated for {len(predictions)} records.")
                st.dataframe(predictions, use_container_width=True)
                csv_bytes = predictions.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv_bytes,
                    file_name="dropout_risk_predictions.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
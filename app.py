import pandas as pd
import streamlit as st
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="wide")
st.title("House Price Prediction using Decision Tree Regressor")
st.caption("A Streamlit app for predicting house prices.")

TARGET_COL = "price"

NUMERIC_FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking"]
CATEGORICAL_FEATURES = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DEFAULT_FILES = [
    "housing.csv",
    "house_price.csv",
    "data/house_price.csv",
    "data/housing.csv",
]


@st.cache_data
def load_data():
    for file_path in DEFAULT_FILES:
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            df.columns = [c.strip() for c in df.columns]
            return df
    return None


@st.cache_resource
def train_model(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in ALL_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError("Missing expected columns: " + ", ".join(missing))

    X = df[ALL_FEATURES]
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")

    valid_mask = y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", DecisionTreeRegressor(random_state=42, max_depth=6)),
        ]
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds) ** 0.5,
        "r2": r2_score(y_test, preds),
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
    }

    return model, metrics, df


df = load_data()

if df is None:
    st.error(
        "Dataset file not found. Place one of these files in the project folder: "
        + ", ".join(DEFAULT_FILES)
    )
    st.stop()

try:
    model, metrics, df = train_model(df)
except Exception as exc:
    st.error(str(exc))
    st.stop()

col1, col2 = st.columns([1.05, 0.95])

with col1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    with st.expander("Show dataset summary"):
        st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
        st.write(df.describe(include="all"))

with col2:
    st.subheader("Model Performance")
    st.metric("MAE", f"{metrics['mae']:.2f}")
    st.metric("RMSE", f"{metrics['rmse']:.2f}")
    st.metric("R² Score", f"{metrics['r2']:.3f}")
    st.write("Train shape:", metrics["train_shape"])
    st.write("Test shape:", metrics["test_shape"])

st.divider()
st.subheader("Predict House Price")

c1, c2 = st.columns(2)

with c1:
    area = st.number_input("Area", min_value=0.0, value=5000.0, step=100.0)
    bedrooms = st.number_input("Bedrooms", min_value=0, value=3, step=1)
    bathrooms = st.number_input("Bathrooms", min_value=0, value=2, step=1)
    stories = st.number_input("Stories", min_value=0, value=2, step=1)

with c2:
    parking = st.number_input("Parking", min_value=0, value=1, step=1)
    mainroad = st.selectbox("Main Road", sorted(df["mainroad"].dropna().astype(str).unique().tolist()))
    guestroom = st.selectbox("Guest Room", sorted(df["guestroom"].dropna().astype(str).unique().tolist()))
    basement = st.selectbox("Basement", sorted(df["basement"].dropna().astype(str).unique().tolist()))

c3, c4 = st.columns(2)

with c3:
    hotwaterheating = st.selectbox("Hot Water Heating", sorted(df["hotwaterheating"].dropna().astype(str).unique().tolist()))
    airconditioning = st.selectbox("Air Conditioning", sorted(df["airconditioning"].dropna().astype(str).unique().tolist()))
    prefarea = st.selectbox("Preferred Area", sorted(df["prefarea"].dropna().astype(str).unique().tolist()))

with c4:
    furnishingstatus = st.selectbox(
        "Furnishing Status",
        sorted(df["furnishingstatus"].dropna().astype(str).unique().tolist())
    )

input_df = pd.DataFrame(
    [[
        area, bedrooms, bathrooms, stories, parking,
        mainroad, guestroom, basement, hotwaterheating,
        airconditioning, prefarea, furnishingstatus
    ]],
    columns=ALL_FEATURES
)

if st.button("Predict Price"):
    predicted_price = float(model.predict(input_df)[0])
    st.success(f"Predicted house price: {predicted_price:,.2f}")

st.caption("Educational demo only.")
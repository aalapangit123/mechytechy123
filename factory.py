import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="FactoryMind AI", layout="wide")

st.title("🏭 FactoryMind AI Dashboard")
st.subheader("Intelligent Resource Optimization in Smart Factories")

# --------------------------------------------------
# Generate Sample Data
# --------------------------------------------------
def generate_data():
    np.random.seed(42)
    data = {
        "Air temperature": np.random.normal(300, 5, 500),
        "Process temperature": np.random.normal(310, 5, 500),
        "Rotational speed": np.random.normal(1500, 200, 500),
        "Torque": np.random.normal(40, 10, 500),
        "Tool wear": np.random.randint(0, 250, 500),
        "Machine failure": np.random.randint(0, 2, 500),
        "energy_consumption": np.random.normal(100, 20, 500),
        "runtime_hours": np.random.randint(1, 10, 500),
        "load_percentage": np.random.randint(10, 100, 500),
    }
    return pd.DataFrame(data)

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("Controls")

data_option = st.sidebar.radio(
    "Choose Data Source",
    ["Use Generated Sample Data", "Upload CSV File"]
)

if data_option == "Use Generated Sample Data":
    df = generate_data()
else:
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a CSV file.")
        st.stop()

# Clean column spaces
df.columns = df.columns.str.strip()

# Show actual columns (for debugging clarity)
st.write("### 📂 Available Columns")
st.write(df.columns.tolist())

# --------------------------------------------------
# Data Preview
# --------------------------------------------------
st.write("### 📊 Raw Data Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Dynamic Feature Selection
# --------------------------------------------------
st.write("### 🔎 Select Feature Columns")

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

feature_columns = st.multiselect(
    "Choose feature columns",
    numeric_columns
)

if len(feature_columns) < 2:
    st.warning("Please select at least 2 feature columns.")
    st.stop()

X = df[feature_columns]

target_column = st.selectbox(
    "Select Target Column (for prediction)",
    numeric_columns
)

y = df[target_column]

# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Classification or Regression Auto Detect
# --------------------------------------------------
if df[target_column].nunique() <= 10:
    model = RandomForestClassifier()
    model_type = "Classification"
else:
    model = RandomForestRegressor()
    model_type = "Regression"

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

st.write(f"### 🤖 {model_type} Model Performance")
st.success(f"Mean Absolute Error: {round(mae, 2)}")

# --------------------------------------------------
# Anomaly Detection
# --------------------------------------------------
st.write("### 🚨 Anomaly Detection")

iso = IsolationForest(contamination=0.05)
df["anomaly"] = iso.fit_predict(X)

anomalies = df[df["anomaly"] == -1]
st.write(f"Total Anomalies Detected: {len(anomalies)}")
st.dataframe(anomalies.head())

# --------------------------------------------------
# Energy Analysis (Only if Column Exists)
# --------------------------------------------------
if "energy_consumption" in df.columns:
    st.write("### 📈 Energy Consumption Graph")

    fig, ax = plt.subplots()
    ax.plot(df["energy_consumption"].values)
    ax.set_xlabel("Data Points")
    ax.set_ylabel("Energy Consumption")
    st.pyplot(fig)

    st.write("### 💰 Economic Loss Estimation")

    energy_model = RandomForestRegressor()
    energy_model.fit(X, df["energy_consumption"])

    df["predicted_energy"] = energy_model.predict(X)
    df["extra_energy"] = df["energy_consumption"] - df["predicted_energy"]
    df["extra_energy"] = df["extra_energy"].apply(lambda x: x if x > 0 else 0)

    electricity_rate = 8  # ₹ per unit
    total_extra_energy = df["extra_energy"].sum()
    total_loss = total_extra_energy * electricity_rate

    st.error(f"Estimated Energy Waste: {round(total_extra_energy,2)} units")
    st.error(f"Estimated Economic Loss: ₹ {round(total_loss,2)}")

# --------------------------------------------------
# Optimization Suggestions (if columns exist)
# --------------------------------------------------
if "runtime_hours" in df.columns and "load_percentage" in df.columns:

    st.write("### ⚙️ Optimization Suggestions")

    low_efficiency = df[
        (df["runtime_hours"] > 6) &
        (df["load_percentage"] < 40)
    ]

    if len(low_efficiency) > 0:
        st.warning("Machines running long hours with low load detected.")
        st.dataframe(low_efficiency.head())
        st.info("Suggestion: Reduce runtime or shut down low-load machines to save energy.")
    else:
        st.success("No major inefficiency detected.")

st.markdown("---")
st.markdown("FactoryMind AI | Smart Manufacturing Optimization System 🚀")
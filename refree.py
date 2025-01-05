import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
@st.cache
def load_data():
    team_stats = pd.read_csv('VAR_Team_Stats.csv')
    return team_stats

data = load_data()

# Streamlit app
def main():
    st.title("VAR Team Stats Analysis")

    st.sidebar.header("Model Configuration")
    target_variable = st.sidebar.selectbox("Select Target Variable", options=data.columns[1:])
    feature_variable = st.sidebar.selectbox("Select Feature Variable", options=data.columns[1:])

    if target_variable == feature_variable:
        st.warning("Feature and Target variables must be different.")
        return

    # Data preview
    st.subheader("Dataset Overview")
    st.write(data.head())

    # Splitting data
    X = data[[feature_variable]].values
    y = data[target_variable].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Results
    st.subheader("Model Results")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"R-squared: {r2_score(y_test, y_pred):.2f}")

    # Visualization
    st.subheader("Prediction Visualization")
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, y_pred, color='red', label='Prediction')
    plt.xlabel(feature_variable)
    plt.ylabel(target_variable)
    plt.title(f"{target_variable} vs {feature_variable}")
    plt.legend()
    st.pyplot(plt)

if __name__ == "__main__":
    main()

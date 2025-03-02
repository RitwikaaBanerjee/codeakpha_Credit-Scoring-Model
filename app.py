import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load(
    "/Users/prashant/Desktop/Credit Scoring Model/MODEL/credit_scoring_model.pkl"
)

# Streamlit UI
st.title("Loan Eligibility Predictor")

st.write(
    """
Enter your details to check whether you are eligible for a loan.
"""
)

# Collect user input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Account Balance", min_value=0, max_value=100000, value=5000)
duration = st.number_input(
    "Duration of Last Contact (in seconds)", min_value=0, max_value=5000, value=200
)
day = st.number_input("Day of the Month", min_value=1, max_value=31, value=15)
campaign = st.number_input(
    "Number of Contacts During Campaign", min_value=1, max_value=50, value=1
)
pdays = st.number_input(
    "Days Since Last Contact", min_value=-1, max_value=999, value=-1
)
previous = st.number_input(
    "Number of Contacts Before Campaign", min_value=0, max_value=50, value=0
)

# Categorical Inputs
job = st.selectbox(
    "Job Type",
    [
        "admin.",
        "technician",
        "services",
        "management",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "unemployed",
        "student",
        "retired",
        "self-employed",
    ],
)
marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
education = st.selectbox("Education Level", ["primary", "secondary", "tertiary"])
default = st.selectbox("Has Credit in Default?", ["yes", "no"])
housing = st.selectbox("Has Housing Loan?", ["yes", "no"])
loan = st.selectbox("Has Personal Loan?", ["yes", "no"])
contact = st.selectbox(
    "Contact Communication Type", ["unknown", "telephone", "cellular"]
)
month = st.selectbox(
    "Last Contact Month",
    [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ],
)
poutcome = st.selectbox(
    "Outcome of the Previous Marketing Campaign",
    ["unknown", "other", "failure", "success"],
)

# When button is pressed
if st.button("Check Loan Eligibility"):
    # Input array to match the order of columns in your dataset
    input_data = np.array(
        [
            [
                age,
                job,
                marital,
                education,
                default,
                balance,
                housing,
                loan,
                contact,
                day,
                month,
                duration,
                campaign,
                pdays,
                previous,
                poutcome,
            ]
        ],
        dtype=object,  # Ensure the dtype is object to handle mixed types
    )

    # Convert input data to DataFrame
    input_data_df = pd.DataFrame(
        input_data,
        columns=[
            "age",
            "job",
            "marital",
            "education",
            "default",
            "balance",
            "housing",
            "loan",
            "contact",
            "day",
            "month",
            "duration",
            "campaign",
            "pdays",
            "previous",
            "poutcome",
        ],
    )

    # Preprocess the input data using the model's preprocessing pipeline
    input_data_preprocessed = model.named_steps["preprocessor"].transform(input_data_df)

    # Predict with the trained model
    prediction = model.named_steps["classifier"].predict(input_data_preprocessed)

    # Output result
    if prediction[0] == 1:
        st.success("You are eligible for a loan!")
    else:
        st.error("You are not eligible for a loan.")

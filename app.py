import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATA AND MODEL
# ---------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("European_Bank.csv")


@st.cache_resource
def load_model():
    with open("random_forest_churn_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


df = load_data()
model, scaler = load_model()

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🏦 Customer Churn Analytics Dashboard")
st.markdown(
    "### European Bank — Customer Segmentation, Churn Analysis & Prediction"
)

st.markdown("---")

# ---------------------------------------------------
# KEY BUSINESS METRICS
# ---------------------------------------------------

total_customers = len(df)
churned_customers = int(df["Exited"].sum())
churn_rate = df["Exited"].mean() * 100
active_customers = total_customers - churned_customers

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{churned_customers:,}")
col3.metric("Overall Churn Rate", f"{churn_rate:.2f}%")
col4.metric("Active Customers", f"{active_customers:,}")

st.markdown("---")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("📌 Dashboard Menu")

section = st.sidebar.radio(
    "Choose a section:",
    [
        "Overview",
        "Churn Analysis",
        "Customer Prediction",
        "Business Recommendations"
    ]
)

# ---------------------------------------------------
# OVERVIEW
# ---------------------------------------------------

if section == "Overview":

    st.header("📊 Customer Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Churn Distribution")

        churn_counts = df["Exited"].value_counts()
        churn_table = pd.DataFrame({
            "Customer Status": ["Stayed", "Exited"],
            "Customers": [
                int(churn_counts.get(0, 0)),
                int(churn_counts.get(1, 0))
            ]
        })

        st.bar_chart(
            churn_table.set_index("Customer Status")
        )

    with col2:
        st.subheader("Customers by Geography")

        geography_counts = df["Geography"].value_counts()

        st.bar_chart(geography_counts)

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

# ---------------------------------------------------
# CHURN ANALYSIS
# ---------------------------------------------------

elif section == "Churn Analysis":

    st.header("📈 Churn Analysis")

    # Geography
    st.subheader("Churn Rate by Geography")

    geography_churn = (
        df.groupby("Geography")["Exited"]
        .mean()
        .mul(100)
        .round(2)
    )

    st.bar_chart(geography_churn)

    st.write(geography_churn.rename("Churn Rate (%)"))

    st.markdown("---")

    # Age
    st.subheader("Churn Rate by Age Group")

    df_analysis = df.copy()

    df_analysis["AgeGroup"] = pd.cut(
        df_analysis["Age"],
        bins=[17, 30, 40, 50, 60, 100],
        labels=["18-30", "31-40", "41-50", "51-60", "61+"]
    )

    age_churn = (
        df_analysis.groupby(
            "AgeGroup",
            observed=False
        )["Exited"]
        .mean()
        .mul(100)
        .round(2)
    )

    st.bar_chart(age_churn)

    st.write(age_churn.rename("Churn Rate (%)"))

    st.markdown("---")

    # Number of products
    st.subheader("Churn Rate by Number of Products")

    product_churn = (
        df.groupby("NumOfProducts")["Exited"]
        .mean()
        .mul(100)
        .round(2)
    )

    st.bar_chart(product_churn)

    st.write(product_churn.rename("Churn Rate (%)"))

    st.markdown("---")

    # Active member
    st.subheader("Churn Rate by Active Membership")

    active_churn = (
        df.groupby("IsActiveMember")["Exited"]
        .mean()
        .mul(100)
        .round(2)
    )

    active_churn.index = ["Inactive", "Active"]

    st.bar_chart(active_churn)

    st.write(active_churn.rename("Churn Rate (%)"))

# ---------------------------------------------------
# CUSTOMER PREDICTION
# ---------------------------------------------------

elif section == "Customer Prediction":

    st.header("🔮 Customer Churn Prediction")

    st.write(
        "Enter customer information below to estimate the probability "
        "that the customer will leave the bank."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=40
        )

        tenure = st.number_input(
            "Tenure (Years)",
            min_value=0,
            max_value=10,
            value=5
        )

        balance = st.number_input(
            "Account Balance",
            min_value=0.0,
            value=75000.0
        )

    with col2:

        num_products = st.selectbox(
            "Number of Products",
            [1, 2, 3, 4]
        )

        has_credit_card = st.selectbox(
            "Has Credit Card?",
            ["Yes", "No"]
        )

        active_member = st.selectbox(
            "Is Active Member?",
            ["Yes", "No"]
        )

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=75000.0
        )

    with col3:

        geography = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"]
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

    st.markdown("---")

    if st.button("🔍 Predict Churn", use_container_width=True):

        # Convert categorical variables into the same dummy variables
        # used during model training.

        geography_germany = 1 if geography == "Germany" else 0
        geography_spain = 1 if geography == "Spain" else 0
        gender_male = 1 if gender == "Male" else 0

        credit_card = 1 if has_credit_card == "Yes" else 0
        active = 1 if active_member == "Yes" else 0

        # IMPORTANT:
        # Keep the feature order exactly the same as the trained model.

        input_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_products],
            "HasCrCard": [credit_card],
            "IsActiveMember": [active],
            "EstimatedSalary": [estimated_salary],
            "Geography_Germany": [geography_germany],
            "Geography_Spain": [geography_spain],
            "Gender_Male": [gender_male],
            "CustomerId": [0],
            "year":[2025]
        })

        # Make prediction
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
       
        stay_probability = probability[0] * 100
        churn_probability = probability[1] * 100

        st.markdown("---")

        if prediction == 1:

            st.error(
                "⚠️ HIGH CHURN RISK — This customer is likely to leave the bank."
            )

        else:

            st.success(
                "✅ LOW CHURN RISK — This customer is likely to stay with the bank."
            )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Probability of Staying",
                f"{stay_probability:.2f}%"
            )

        with col2:
            st.metric(
                "Probability of Leaving",
                f"{churn_probability:.2f}%"
            )

        st.progress(
            int(churn_probability)
        )

        if churn_probability >= 70:

            st.warning(
                "Recommended action: Contact the customer proactively, "
                "offer retention benefits, and investigate dissatisfaction."
            )

        elif churn_probability >= 40:

            st.warning(
                "Recommended action: Monitor the customer and consider "
                "personalized engagement."
            )

        else:

            st.info(
                "Recommended action: Maintain regular customer engagement."
            )

# ---------------------------------------------------
# BUSINESS RECOMMENDATIONS
# ---------------------------------------------------

elif section == "Business Recommendations":

    st.header("💡 Business Recommendations")

    st.subheader("1. Focus on High-Risk Age Groups")

    st.write(
        "Customers aged 51–60 show a substantially higher churn rate. "
        "The bank should design targeted retention campaigns for older "
        "customer segments."
    )

    st.subheader("2. Investigate Germany")

    st.write(
        "Germany has a considerably higher churn rate than France and Spain. "
        "The bank should investigate customer experience, competition, "
        "pricing and service quality in the German market."
    )

    st.subheader("3. Monitor Customers with 3–4 Products")

    st.write(
        "Customers with three or four products show exceptionally high "
        "churn rates in this dataset. These customers should be investigated "
        "for dissatisfaction, product complexity or service issues."
    )

    st.subheader("4. Improve Customer Engagement")

    st.write(
        "Inactive customers have higher churn risk than active members. "
        "Personalized communication, loyalty programs and regular engagement "
        "can help reduce churn."
    )

    st.subheader("5. Use Predictive Analytics")

    st.write(
        "The churn prediction model can help the bank identify customers "
        "who are more likely to leave and prioritize retention efforts."
    )

    st.markdown("---")

    st.subheader("📌 Key Findings")

    findings = pd.DataFrame({
        "Finding": [
            "Overall churn rate",
            "Highest-risk geography",
            "Highest-risk age group",
            "Highest-risk product segment"
        ],
        "Result": [
            "20.37%",
            "Germany — 32.44%",
            "Age 51–60 — 56.21%",
            "4 products — 100%"
        ]
    })

    st.dataframe(
        findings,
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.caption(
    "Customer Churn Analytics Project | European Bank Dataset"
)
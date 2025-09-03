import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import os
import numpy as np

st.set_page_config(page_title="UPI Fraud Detection EDA & Prediction", layout="wide")
st.title("UPI Fraud Detection - EDA & Prediction App")

uploaded_file = st.file_uploader("Upload your CSV data file", type=["csv"])
df = None
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Data Cleaning
    columns_to_drop = ['Days_Since_Last_Transaction', 'Transaction_Frequency', 'Transaction_Amount_Deviation',
                      'Time', 'Date', 'Device_OS', 'Transaction_Channel', 'Transaction_Status', 'Transaction_City',
                      'Transaction_ID', 'Merchant_ID', 'Customer_ID', 'Device_ID', 'IP_Address']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head())

    # Fraud and Normal Data
    fraud = df[df['fraud'] == 1]
    normal = df[df['fraud'] == 0]

    st.markdown("---")
    st.header("Fraud Distribution Visualizations")

    # Plot 1: Fraud Distribution by Transaction Type
    if 'Transaction_Type' in fraud.columns:
        fig1 = px.bar(
            x=fraud['Transaction_Type'].value_counts().index,
            y=fraud['Transaction_Type'].value_counts().values,
            color=fraud['Transaction_Type'].value_counts().index,
            title='Fraud Distribution by Transaction Type',
            labels={'x': 'Transaction Type', 'y': 'Fraud Count'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig1.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig1, use_container_width=True)

    # Plot 2: Fraud Distribution by Payment Gateway
    if 'Payment_Gateway' in fraud.columns:
        fig2 = px.bar(
            x=fraud['Payment_Gateway'].value_counts().index,
            y=fraud['Payment_Gateway'].value_counts().values,
            color=fraud['Payment_Gateway'].value_counts().index,
            title='Fraud Distribution by Payment Gateway',
            labels={'x': 'Payment Gateway', 'y': 'Fraud Count'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig2.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig2, use_container_width=True)

    # Plot 3: Fraud Distribution by Merchant Category
    if 'Merchant_Category' in fraud.columns:
        fig3 = px.bar(
            x=fraud['Merchant_Category'].value_counts().index,
            y=fraud['Merchant_Category'].value_counts().values,
            color=fraud['Merchant_Category'].value_counts().index,
            title='Fraud Distribution by Merchant Category',
            labels={'x': 'Merchant Category', 'y': 'Fraud Count'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig3.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig3, use_container_width=True)

    # Plot 4: Transaction Amount Distribution
    if 'amount' in fraud.columns:
        fig4 = px.histogram(fraud, x='amount', nbins=20, title='Transaction Amount Distribution',
                           labels={'amount': 'Transaction Amount'})
        st.plotly_chart(fig4, use_container_width=True)

    # Plot 5: Transaction Frequency vs Fraud
    if 'Transaction_Frequency' in fraud.columns:
        fig5 = px.scatter(fraud, x='Transaction_Frequency', y='fraud',
                         title='Transaction Frequency vs Fraud',
                         labels={'Transaction_Frequency': 'Transaction Frequency', 'fraud': 'Fraud'},
                         color='fraud', color_discrete_map={0: 'blue', 1: 'red'})
        st.plotly_chart(fig5, use_container_width=True)

    # Plot 6: Days Since Last Transaction vs Fraud
    if 'Days_Since_Last_Transaction' in fraud.columns:
        fig6 = px.scatter(fraud, x='Days_Since_Last_Transaction', y='fraud',
                         title='Days Since Last Transaction vs Fraud',
                         labels={'Days_Since_Last_Transaction': 'Days Since Last Transaction', 'fraud': 'Fraud'},
                         color='fraud', color_discrete_map={0: 'blue', 1: 'red'})
        st.plotly_chart(fig6, use_container_width=True)

    # Plot 7: Fraud Distribution by Transaction State
    if 'Transaction_State' in fraud.columns:
        fig7 = px.bar(
            x=fraud['Transaction_State'].value_counts().index,
            y=fraud['Transaction_State'].value_counts().values,
            color=fraud['Transaction_State'].value_counts().index,
            title='Fraud Distribution by Transaction State',
            labels={'x': 'Transaction State', 'y': 'Fraud Count'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig7.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig7, use_container_width=True)

    # Plot 8: Fraud Distribution by Device OS
    if 'Device_OS' in fraud.columns:
        fig8 = px.bar(
            x=fraud['Device_OS'].value_counts().index,
            y=fraud['Device_OS'].value_counts().values,
            color=fraud['Device_OS'].value_counts().index,
            title='Fraud Distribution by Device OS',
            labels={'x': 'Device OS', 'y': 'Fraud Count'},
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig8.update_layout(xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig8, use_container_width=True)

    st.markdown("---")
    st.header("Insights from EDA")
    st.markdown("""
    - Transaction types: Bank transfer, purchase & bill payment, are highly contributing to fraudulent transactions.
    - Platforms like ICICI, HDFC and GooglePay have reported the highest number of fraudulent transactions.
    - Merchant category: home delivery, travel bookings, utility, have reported the highest number of fraudulent transactions.
    - Transaction amount ranging 250 to 750 are highly sensitive to fraudulent transactions.
    - Transaction amount ranging 0 to 1250 are highest contributors to fraudulent transactions between FY 23–24.
    - Transaction frequency: 0 to 10 is highly sensitive to fraudulent transactions.
    - Days since last transaction feature is not contributing to the analysis as no pattern identified, hence can be dropped.
    - States – Himachal Pradesh, Rajasthan, Meghalaya & Bihar are highly sensitive to fraudulent transactions.
    - Android OS reported highest number of fraudulent transactions.
    """)

    # --- Prediction UI ---
    st.markdown("---")
    st.header("UPI Fraud Prediction")
    model_path = "dt_model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        st.success("Model loaded successfully!")

        # Get feature columns (excluding 'fraud')
        feature_cols = [col for col in df.columns if col != 'fraud']
        st.subheader("Enter Transaction Details for Prediction:")
        input_data = {}
        for col in feature_cols:
            if df[col].dtype == 'object':
                options = list(df[col].unique())
                input_data[col] = st.selectbox(f"{col}", options)
            elif df[col].dtype in ['int64', 'float64']:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                input_data[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)
            else:
                input_data[col] = st.text_input(f"{col}")

        if st.button("Predict Fraud"):
            try:
                # Prepare input for model
                input_df = pd.DataFrame([input_data])
                input_df = pd.get_dummies(input_df)

                # Get expected columns from cleaned training data (excluding 'fraud')
                expected_cols = [col for col in df.columns if col != 'fraud']
                expected_df = pd.get_dummies(df[expected_cols])
                expected_cols = expected_df.columns.tolist()

                # Add missing columns to input_df
                for col in expected_cols:
                    if col not in input_df.columns:
                        input_df[col] = 0
                # Remove extra columns
                input_df = input_df[expected_cols]

                # Predict
                pred = model.predict(input_df)[0]
                prob = model.predict_proba(input_df)[0][1]
                st.markdown(f"## Prediction: {'Fraud' if pred==1 else 'Not Fraud'}")
                st.markdown(f"### Probability of Fraud: {prob:.2%}")
            except Exception:
                st.markdown("## Prediction: Fraud occur")
    else:
        st.warning("Model file (dt_model.pkl) not found. Please ensure the model is present in the app directory.")
else:
    st.info("Please upload a CSV file to begin analysis.")

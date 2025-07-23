import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

st.set_page_config(page_title="CLV Prediction", layout="centered")
st.title("🧮 Customer Lifetime Value (CLV) Prediction")

# Step 1: Upload the file
uploaded_file = st.file_uploader("📤 Upload the Online Retail Excel file (.xlsx)", type=["xlsx"])

if uploaded_file:
    with st.spinner("🔄 Loading and processing data... please wait"):
        # Step 2: Read and show raw data
        df = pd.read_excel(uploaded_file)
        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        # Step 3: Preprocess
        df = df[df['Invoice'].notnull() & df['Customer ID'].notnull()]
        df = df[~df['Invoice'].astype(str).str.startswith('C')]
        df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['TotalPrice'] = df['Quantity'] * df['Price']

        # Step 4: Summarize transactions
        summary = summary_data_from_transaction_data(
            df,
            customer_id_col='Customer ID',
            datetime_col='InvoiceDate',
            monetary_value_col='TotalPrice',
            observation_period_end=df['InvoiceDate'].max()
        )

        # Step 5: Filter valid rows
        summary = summary[(summary['monetary_value'] > 0) & (summary['frequency'] > 0)]

        # Step 6: Fit models
        bgf = BetaGeoFitter()
        bgf.fit(summary['frequency'], summary['recency'], summary['T'])

        ggf = GammaGammaFitter()
        ggf.fit(summary['frequency'], summary['monetary_value'])

        # Step 7: Predict CLV
        summary['CLV (6 months)'] = ggf.customer_lifetime_value(
            bgf,
            summary['frequency'],
            summary['recency'],
            summary['T'],
            summary['monetary_value'],
            time=6, freq='D', discount_rate=0.01
        )

        summary['CLV Segment'] = pd.qcut(summary['CLV (6 months)'], 3, labels=['Low', 'Medium', 'High'])

    st.success("✅ Processing Complete!")

    # Step 8: Show filter
    st.subheader("📌 Filter by CLV Segment")
    selected_segment = st.selectbox("Choose Segment", ['All', 'Low', 'Medium', 'High'])

    if selected_segment != 'All':
        filtered_summary = summary[summary['CLV Segment'] == selected_segment]
    else:
        filtered_summary = summary

    st.subheader("📊 CLV Summary")
    st.dataframe(filtered_summary[['frequency', 'recency', 'T', 'monetary_value', 'CLV (6 months)', 'CLV Segment']].head(20))

    # Step 9: Visualizations
    st.subheader("📈 Top 20 Customers by CLV")
    top_customers = filtered_summary.sort_values(by='CLV (6 months)', ascending=False).head(20)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_customers['CLV (6 months)'], y=top_customers.index)
    plt.xlabel("CLV (6 months)")
    plt.ylabel("Customer ID")
    st.pyplot(plt.gcf())

    st.subheader("📉 CLV Distribution")
    plt.figure(figsize=(10, 4))
    sns.histplot(filtered_summary['CLV (6 months)'], kde=True, bins=30, color='purple')
    plt.xlabel("CLV (6 months)")
    st.pyplot(plt.gcf())

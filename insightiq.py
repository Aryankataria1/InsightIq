import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
from google import genai
import re
from dotenv import load_dotenv
import os
load_dotenv()
from fpdf import FPDF
import io

st.title("InsightIQ 🧠")
st.write("AI-Powered Business Analyst for Small Businesses")
client = genai.Client(api_key=os.environ.get("API_KEY"))

uploaded_file = st.file_uploader("Please upload your csv data", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Your Data")
    st.write(df.head(10))
    st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    st.subheader("Sales Trend")
    fig = px.line(df, x="Month", y="Sales", title="Monthly Sales Trend")
    st.plotly_chart(fig)
    
    st.subheader("AI Business Insights")
    summary = df.describe().to_string()
    prompt = f"""
    Here is a sales data summary for a small business with {df.shape[0]} months of data:
    {summary}
    Give 3 short plain-language business insights the owner should act on.
    """
    
    with st.spinner("Generating insights..."):
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt
        )
    
    st.write(response.text)

    st.subheader("Sales Forecast")
    X=df[["Month"]]
    y=df["Sales"]

    model=LinearRegression()
    model.fit(X,y)

    last_month=df["Month"].max()

    future_months=np.array([last_month+1, last_month+2,last_month+3]).reshape(-1,1)
    predictions=model.predict(future_months)

    forecast_df=pd.DataFrame({
        "Month":[last_month+1,last_month+2,last_month+3],
        "Predicted Sales":predictions.round(2)
    })

    st.write(forecast_df)

    fig2=px.line(forecast_df, x="Month", y="Predicted Sales", title="Next 3 Months Sales Forecast")
    st.plotly_chart(fig2)

st.subheader("Download Report")

if st.button("Generate PDF Report"):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "InsightIQ Business Report", new_x="LMARGIN", new_y="NEXT", align="C")
    
    # Summary stats
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, f"Total Months Analyzed: {df.shape[0]}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"Average Sales: {df['Sales'].mean().round(2)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"Best Month Sales: {df['Sales'].max()}", new_x="LMARGIN", new_y="NEXT")
    
    # AI Insights
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "AI Insights:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    clean_insights = re.sub(r'[#*]', '', response.text)
    pdf.multi_cell(0, 8, clean_insights)
    
    # Forecast
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "Sales Forecast:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    for _, row in forecast_df.iterrows():
        pdf.cell(0, 8, f"Month {int(row['Month'])}: {row['Predicted Sales']}", new_x="LMARGIN", new_y="NEXT")
    
    # Save and download
    pdf_bytes = pdf.output()
    st.download_button(
        label="Download PDF",
        data=bytes(pdf_bytes),
        file_name="insightiq_report.pdf",
        mime="application/pdf"
    )
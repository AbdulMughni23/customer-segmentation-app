# Customer Segmentation App

This application performs customer segmentation using RFM analysis and K-Means clustering on retail transaction data.

## Features

- Data loading and cleaning
- RFM (Recency, Frequency, Monetary) analysis
- K-Means clustering for customer segmentation
- Interactive visualizations
- Targeted marketing strategy recommendations

## How to Use

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## Demo

The app is deployed and available on Streamlit Community Cloud at: [Customer-Segmentation-App](https://customer-segmentation-app.streamlit.app)

## Data

The app uses the Online Retail dataset from the UCI Machine Learning Repository, which contains transactions for a UK-based online retailer.

## Analysis Workflow

1. **Data Cleaning**: Remove missing values, canceled transactions, and filter for positive quantities and prices
2. **RFM Analysis**: Calculate Recency, Frequency, and Monetary value for each customer
3. **Clustering**: Apply K-Means clustering to identify customer segments
4. **Marketing Strategy**: Generate targeted strategies for each customer segment

## Requirements

- Python 3.8+
- See requirements.txt for package dependencies


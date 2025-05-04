# Financial Analysis & Forecasting App

## Course Information
**Course Name:** AF3005 Programming For Finance  
**Instructor:** Dr. Usama Arshad  

## App Overview

This Streamlit web application provides an interactive dashboard for financial data analysis and forecasting. Built as part of the AF3005 Programming For Finance course, the app empowers users to fetch, visualize, and analyze stock market data using modern data science techniques.

---

## 🔧 Key Features

- 📈 **Stock Data Fetching**  
  Users can enter a valid stock ticker (e.g., `AAPL`, `MSFT`, `GOOGL`) to fetch historical data using the Yahoo Finance API (`yfinance`).

- 📊 **Data Visualization**  
    -Numerous Charts and Visualisation techniques to show data characteristics.

- 📋 **Descriptive Statistics**  
  Displays useful financial statistics and correlation matrices to understand stock behavior.

- ⚙️ **Data Preprocessing**  
  Handles missing data and prepares features for machine learning modeling.

- 🤖 **Machine Learning Models**  
  Predict future adjusted closing prices using:
  - Linear Regression  
  - Logistic Regression 
  - K-Means Clustering 

- 🎯 **Model Evaluation**  
  Includes visual comparison of predicted vs actual values and performance metrics (MAE, MSE, RMSE).

---

## 🛠 Technologies Used

- Python 3.11+
- Streamlit
- Pandas
- Numpy
- Matplotlib
- Plotly
- Scikit-learn
- YFinance
- OpenPyXL

---

## 🚀 How to Run the App

If you're using [Replit](https://replit.com):

1. Import this GitHub repository to Replit.
2. Replit automatically installs dependencies and runs the app.
3. If needed, manually run the app using the shell:
   ```bash
   streamlit run app.py
Alternatively, to run locally:

git clone https://github.com/M-Abdullah-K/AF3005_Programming_For_Finance_Assignment_3.git
cd AF3005_Programming_For_Finance_Assignment_3
pip install -r requirements.txt
streamlit run app.py
📁 Project Structure
.
├── app.py                 # Main Streamlit app
├── data_processing.py     # Data cleaning and feature engineering
├── ml_utils.py            # ML model training and evaluation
├── visualization.py       # Plotting and charting utilities
├── requirements.txt       # Python dependencies
├── .replit                # Replit config
├── pyproject.toml         # Project metadata

import streamlit as st
import pandas as pd
import numpy as np
import base64
import time
import os
from PIL import Image
import io

# Custom modules
import ml_utils
import visualization
import data_processing
from assets.finance_gifs import get_encoded_gif

# Helper functions for timestamp handling
def is_timestamp_column(data_series):
    """Check if a column contains timestamp data"""
    return pd.api.types.is_datetime64_any_dtype(data_series) or (
        hasattr(data_series, 'dtype') and 
        isinstance(data_series.dtype, pd.DatetimeTZDtype)
    )

def safe_numeric_conversion(feature_series):
    """
    Safely get min, max, and mean values for a feature, handling timestamps and other non-numeric data.

    Parameters:
    -----------
    feature_series : pandas.Series
        The feature data

    Returns:
    --------
    tuple : (min_val, max_val, mean_val) if conversion is possible, None otherwise
    """
    # Check if feature contains timestamp data
    if is_timestamp_column(feature_series):
        return None

    # Try to safely convert values to float
    try:
        min_val = float(feature_series.min())
        max_val = float(feature_series.max())
        mean_val = float(feature_series.mean())
        return (min_val, max_val, mean_val)
    except (TypeError, ValueError):
        return None

def handle_timestamp_columns(df):
    """
    Identify timestamp columns and convert them to numeric features.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe

    Returns:
    --------
    df_processed : pandas.DataFrame
        Dataframe with timestamp columns converted to numeric features
    timestamp_cols : list
        List of detected timestamp columns
    """
    timestamp_cols = []

    for col in df.columns:
        if is_timestamp_column(df[col]):
            timestamp_cols.append(col)

    # If no timestamp columns found, return original dataframe
    if not timestamp_cols:
        return df, []

    # Process timestamp columns
    df_processed = df.copy()

    for col in timestamp_cols:
        # Extract useful components from datetime
        df_processed[f"{col}_year"] = df_processed[col].dt.year
        df_processed[f"{col}_month"] = df_processed[col].dt.month
        df_processed[f"{col}_day"] = df_processed[col].dt.day
        df_processed[f"{col}_dayofweek"] = df_processed[col].dt.dayofweek

        # Try to extract hour if available
        if hasattr(df_processed[col].dt, 'hour'):
            df_processed[f"{col}_hour"] = df_processed[col].dt.hour

        # Drop the original datetime column
        df_processed = df_processed.drop(columns=[col])

    return df_processed, timestamp_cols

# Page configuration
st.set_page_config(
    page_title="Financial ML Pipeline",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'y' not in st.session_state:
    st.session_state.y = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'evaluation_metrics' not in st.session_state:
    st.session_state.evaluation_metrics = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'missing_values_info' not in st.session_state:
    st.session_state.missing_values_info = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'last_data_update' not in st.session_state:
    st.session_state.last_data_update = None
if 'timestamp_columns' not in st.session_state:
    st.session_state.timestamp_columns = []
if 'X_train_processed' not in st.session_state:
    st.session_state.X_train_processed = None
if 'X_test_processed' not in st.session_state:
    st.session_state.X_test_processed = None

# Function to display welcome page
def welcome_page():
    st.markdown(f"""
    <div style='text-align: center;'>
        <h1 style='color: #2e6db4;'>Welcome to Financial ML Pipeline</h1>
        <p style='font-size: 20px;'>An interactive machine learning application for financial datasets</p>
    </div>
    """, unsafe_allow_html=True)

    # Display the finance GIF
    welcome_gif = get_encoded_gif('welcome')
    st.markdown(f"""
    <div style='text-align: center;'>
        <img src="data:image/gif;base64,{welcome_gif}" alt="Welcome" width="500">
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🚀 Features:
    - Upload your own financial datasets or fetch real-time stock market data
    - Process and analyze financial data with interactive visualizations
    - Train and evaluate machine learning models (Linear Regression, Logistic Regression, K-Means Clustering)
    - Visualize predictions and model performance

    ### 🔍 How to Use:
    1. Select your data source in the sidebar
    2. Follow the step-by-step machine learning workflow
    3. Interact with the visualizations and explore your data
    4. Download your results

    ### 📊 Let's get started!
    Use the sidebar to select your data source and begin your ML journey.
    """)

    # Start button to proceed to the ML workflow
    if st.button("Start ML Pipeline", key="start_pipeline", use_container_width=True):
        st.session_state.current_step = 1
        st.rerun()

# Sidebar for data selection and model configuration
def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Configuration")

        # Data Source Selection
        st.header("1. Data Source")
        data_source = st.radio(
            "Select Data Source:",
            options=["Upload Kragle Dataset", "Yahoo Finance"],
            index=0
        )

        # Handle data source selection
        if data_source == "Upload Kragle Dataset":
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.data = df
                    st.session_state.data_source = "Kragle"
                    st.success("Dataset successfully loaded!")
                    # Display data preview in the sidebar
                    st.write("Data Preview:")
                    st.dataframe(df.head(3), use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading the dataset: {e}")

        else:  # Yahoo Finance
            st.session_state.data_source = "Yahoo"
            ticker_symbol = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT):", "AAPL")
            period = st.selectbox("Select Time Period:", 
                                 options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
                                 index=3)
            interval = st.selectbox("Select Interval:", 
                                   options=["1d", "1wk", "1mo"],
                                   index=0)

            if st.button("Fetch Stock Data", key="fetch_data"):
                with st.spinner("Fetching data from Yahoo Finance..."):
                    try:
                        df = data_processing.fetch_yahoo_finance_data(ticker_symbol, period, interval)
                        st.session_state.data = df
                        st.session_state.last_data_update = pd.Timestamp.now()
                        st.success(f"Successfully fetched data for {ticker_symbol}")
                        # Display data preview
                        st.write("Data Preview:")
                        st.dataframe(df.head(3), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error fetching stock data: {e}")

        # Model Selection (once data is loaded)
        if st.session_state.data is not None:
            st.header("2. Model Selection")
            model_type = st.selectbox(
                "Select Machine Learning Model:",
                options=["Linear Regression", "Logistic Regression", "K-Means Clustering"],
                index=0
            )
            st.session_state.model_type = model_type

            # Show different configuration options based on model type
            if model_type == "Linear Regression":
                st.info("Linear Regression predicts continuous values based on linear relationships.")
            elif model_type == "Logistic Regression":
                st.info("Logistic Regression predicts binary outcomes like buy/sell decisions.")
            elif model_type == "K-Means Clustering":
                st.info("K-Means Clustering groups similar data points to find patterns.")
                st.session_state.n_clusters = st.slider("Number of Clusters:", min_value=2, max_value=10, value=3)

        # Real-time data updates (Bonus feature)
        if st.session_state.data_source == "Yahoo":
            st.header("3. Real-time Updates")
            auto_update = st.checkbox("Enable Automatic Updates", value=False)

            if auto_update:
                update_interval = st.slider("Update Interval (minutes):", 
                                           min_value=5, max_value=60, value=15, step=5)

                # Show last update time if available
                if st.session_state.last_data_update:
                    st.write(f"Last updated: {st.session_state.last_data_update.strftime('%Y-%m-%d %H:%M:%S')}")

                    # Check if it's time to update
                    current_time = pd.Timestamp.now()
                    elapsed_minutes = (current_time - st.session_state.last_data_update).total_seconds() / 60

                    if elapsed_minutes >= update_interval:
                        st.warning("Updating data...")
                        try:
                            df = data_processing.fetch_yahoo_finance_data(ticker_symbol, period, interval)
                            st.session_state.data = df
                            st.session_state.last_data_update = current_time
                            st.success("Data updated successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error updating data: {e}")

# ML Pipeline Steps
def load_data_step():
    st.header("Step 1: Load and Explore Data 📊")

    if st.session_state.data is None:
        st.warning("Please load data using the sidebar options before proceeding.")
        return

    # Display data source info
    if st.session_state.data_source == "Kragle":
        st.success("✅ Kragle dataset loaded successfully")
    else:
        st.success("✅ Yahoo Finance data loaded successfully")

    # Display data exploration options
    st.subheader("Data Overview")

    tab1, tab2, tab3 = st.tabs(["Preview", "Summary Statistics", "Visualization"])

    with tab1:
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        st.write(f"Dataset Shape: {st.session_state.data.shape[0]} rows, {st.session_state.data.shape[1]} columns")

    with tab2:
        st.write("Summary Statistics:")
        st.dataframe(st.session_state.data.describe(), use_container_width=True)

        st.write("Data Types:")
        st.dataframe(pd.DataFrame(st.session_state.data.dtypes, columns=["Data Type"]), use_container_width=True)

        st.write("Missing Values:")
        missing_values = st.session_state.data.isnull().sum()
        missing_values = pd.DataFrame(missing_values, columns=["Missing Values"])
        missing_values["Percentage"] = (missing_values["Missing Values"] / len(st.session_state.data)) * 100
        st.dataframe(missing_values, use_container_width=True)

    with tab3:
        st.subheader("Data Visualization")

        # Select columns to visualize
        numeric_columns = st.session_state.data.select_dtypes(include=['int64', 'float64']).columns.tolist()

        if len(numeric_columns) > 0:
            selected_columns = st.multiselect("Select columns to visualize:", numeric_columns, default=numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns)

            if selected_columns:
                chart_type = st.selectbox("Select chart type:", ["Line Chart", "Bar Chart", "Histogram", "Box Plot", "Scatter Plot", "Correlation Heatmap"])

                if chart_type == "Line Chart":
                    st.plotly_chart(visualization.create_line_chart(st.session_state.data, selected_columns), use_container_width=True)
                elif chart_type == "Bar Chart":
                    st.plotly_chart(visualization.create_bar_chart(st.session_state.data, selected_columns), use_container_width=True)
                elif chart_type == "Histogram":
                    st.plotly_chart(visualization.create_histogram(st.session_state.data, selected_columns), use_container_width=True)
                elif chart_type == "Box Plot":
                    st.plotly_chart(visualization.create_box_plot(st.session_state.data, selected_columns), use_container_width=True)
                elif chart_type == "Scatter Plot":
                    if len(selected_columns) >= 2:
                        x_column = st.selectbox("Select X-axis column:", selected_columns)
                        y_column = st.selectbox("Select Y-axis column:", [col for col in selected_columns if col != x_column])
                        st.plotly_chart(visualization.create_scatter_plot(st.session_state.data, x_column, y_column), use_container_width=True)
                    else:
                        st.warning("Select at least 2 columns for scatter plot")
                elif chart_type == "Correlation Heatmap":
                    st.plotly_chart(visualization.create_correlation_heatmap(st.session_state.data[selected_columns]), use_container_width=True)
        else:
            st.warning("No numeric columns available for visualization")

    # Proceed button
    if st.button("Proceed to Preprocessing", key="proceed_to_preprocessing", use_container_width=True):
        st.session_state.current_step = 2
        st.rerun()

def preprocessing_step():
    st.header("Step 2: Data Preprocessing 🧹")

    if st.session_state.data is None:
        st.warning("Please load data first.")
        return

    # Make a copy of the original data for preprocessing
    if st.session_state.processed_data is None:
        st.session_state.processed_data = st.session_state.data.copy()

    # Display missing values information
    st.subheader("Missing Values")
    missing_values = st.session_state.processed_data.isnull().sum()
    missing_values_df = pd.DataFrame(missing_values, columns=["Missing Values"])
    missing_values_df["Percentage"] = (missing_values_df["Missing Values"] / len(st.session_state.processed_data)) * 100

    # Store missing values info in session state
    st.session_state.missing_values_info = missing_values_df

    st.dataframe(missing_values_df, use_container_width=True)

    # Handle missing values
    st.subheader("Handle Missing Values")

    columns_with_missing = missing_values[missing_values > 0].index.tolist()

    if not columns_with_missing:
        st.success("No missing values to handle!")
    else:
        st.write("Columns with missing values:", ", ".join(columns_with_missing))

        missing_value_strategy = st.selectbox(
            "Select strategy for handling missing values:",
            options=["Remove columns", "Remove rows with missing values", "Fill with mean/median/mode", "Fill with specific value"],
            index=2
        )

        if missing_value_strategy == "Remove columns":
            if st.button("Remove columns with missing values", key="remove_columns"):
                st.session_state.processed_data = st.session_state.processed_data.drop(columns=columns_with_missing)
                st.success(f"Removed {len(columns_with_missing)} columns with missing values")
                st.rerun()

        elif missing_value_strategy == "Remove rows with missing values":
            if st.button("Remove rows with any missing values", key="remove_rows"):
                original_len = len(st.session_state.processed_data)
                st.session_state.processed_data = st.session_state.processed_data.dropna()
                new_len = len(st.session_state.processed_data)
                st.success(f"Removed {original_len - new_len} rows with missing values")
                st.rerun()

        elif missing_value_strategy == "Fill with mean/median/mode":
            col1, col2 = st.columns(2)

            with col1:
                columns_to_fill = st.multiselect(
                    "Select columns to fill:",
                    options=columns_with_missing,
                    default=columns_with_missing
                )

            with col2:
                fill_method = st.selectbox(
                    "Fill method:",
                    options=["Mean", "Median", "Mode"],
                    index=0
                )

            if st.button("Fill missing values", key="fill_missing"):
                for column in columns_to_fill:
                    if st.session_state.processed_data[column].dtype in ['int64', 'float64']:
                        if fill_method == "Mean":
                            st.session_state.processed_data[column].fillna(st.session_state.processed_data[column].mean(), inplace=True)
                        elif fill_method == "Median":
                            st.session_state.processed_data[column].fillna(st.session_state.processed_data[column].median(), inplace=True)
                        else:  # Mode
                            st.session_state.processed_data[column].fillna(st.session_state.processed_data[column].mode()[0], inplace=True)
                    else:
                        # For non-numeric columns, use mode
                        st.session_state.processed_data[column].fillna(st.session_state.processed_data[column].mode()[0], inplace=True)

                st.success(f"Successfully filled missing values in {len(columns_to_fill)} columns using {fill_method.lower()}")
                st.rerun()

        elif missing_value_strategy == "Fill with specific value":
            col1, col2 = st.columns(2)

            with col1:
                column_to_fill = st.selectbox(
                    "Select column to fill:",
                    options=columns_with_missing
                )

            with col2:
                fill_value = st.text_input("Fill value:")

            if st.button("Fill with value", key="fill_value"):
                try:
                    # Try to convert fill_value to the column's data type
                    if st.session_state.processed_data[column_to_fill].dtype in ['int64', 'float64']:
                        fill_value = float(fill_value)

                    st.session_state.processed_data[column_to_fill].fillna(fill_value, inplace=True)
                    st.success(f"Successfully filled missing values in '{column_to_fill}' with '{fill_value}'")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error filling values: {e}")

    # Handle outliers
    st.subheader("Handle Outliers")

    numeric_columns = st.session_state.processed_data.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if not numeric_columns:
        st.warning("No numeric columns available for outlier detection")
    else:
        col1, col2 = st.columns(2)

        with col1:
            outlier_columns = st.multiselect(
                "Select columns for outlier detection:",
                options=numeric_columns,
                default=numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns
            )

        with col2:
            outlier_method = st.selectbox(
                "Outlier detection method:",
                options=["IQR (Interquartile Range)", "Z-Score"],
                index=0
            )

        if outlier_columns:
            # Display box plots for selected columns to visualize outliers
            st.write("Box plots to visualize outliers:")
            st.plotly_chart(visualization.create_box_plot(st.session_state.processed_data, outlier_columns), use_container_width=True)

            outlier_action = st.selectbox(
                "Action for outliers:",
                options=["No action", "Remove outliers", "Cap outliers"],
                index=0
            )

            if outlier_action != "No action" and st.button("Process outliers", key="process_outliers"):
                if outlier_method == "IQR (Interquartile Range)":
                    for column in outlier_columns:
                        Q1 = st.session_state.processed_data[column].quantile(0.25)
                        Q3 = st.session_state.processed_data[column].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        if outlier_action == "Remove outliers":
                            # Keep only the rows where this column is not an outlier
                            st.session_state.processed_data = st.session_state.processed_data[
                                (st.session_state.processed_data[column] >= lower_bound) & 
                                (st.session_state.processed_data[column] <= upper_bound)
                            ]
                        else:  # Cap outliers
                            # Cap the values
                            st.session_state.processed_data[column] = np.where(
                                st.session_state.processed_data[column] < lower_bound,
                                lower_bound,
                                st.session_state.processed_data[column]
                            )
                            st.session_state.processed_data[column] = np.where(
                                st.session_state.processed_data[column] > upper_bound,
                                upper_bound,
                                st.session_state.processed_data[column]
                            )

                elif outlier_method == "Z-Score":
                    for column in outlier_columns:
                        z_scores = np.abs((st.session_state.processed_data[column] - st.session_state.processed_data[column].mean()) / st.session_state.processed_data[column].std())

                        if outlier_action == "Remove outliers":
                            # Keep only the rows where z-score is less than 3
                            st.session_state.processed_data = st.session_state.processed_data[z_scores < 3]
                        else:  # Cap outliers
                            # Identify outliers
                            outliers = z_scores > 3

                            # Get the values at which to cap
                            mean = st.session_state.processed_data[column].mean()
                            std = st.session_state.processed_data[column].std()

                            # Cap positive outliers
                            upper_cap = mean + 3 * std
                            # Cap negative outliers
                            lower_cap = mean - 3 * std

                            # Apply capping
                            st.session_state.processed_data.loc[
                                (outliers) & (st.session_state.processed_data[column] > mean), 
                                column
                            ] = upper_cap

                            st.session_state.processed_data.loc[
                                (outliers) & (st.session_state.processed_data[column] < mean), 
                                column
                            ] = lower_cap

                st.success(f"Successfully processed outliers using {outlier_method}")
                st.rerun()

    # Compare original and processed data
    st.subheader("Original vs. Processed Data")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Original Data Shape:", st.session_state.data.shape)

    with col2:
        st.write("Processed Data Shape:", st.session_state.processed_data.shape)

    # Display sample of processed data
    st.write("Processed Data Preview:")
    st.dataframe(st.session_state.processed_data.head(5), use_container_width=True)

    # Notification of preprocessing completion
    st.info("Preprocessing completed! You can now proceed to feature engineering.")

    # Proceed button
    if st.button("Proceed to Feature Engineering", key="proceed_to_feature_engineering", use_container_width=True):
        st.session_state.current_step = 3
        st.rerun()

def feature_engineering_step():
    st.header("Step 3: Feature Engineering 🔧")

    if st.session_state.processed_data is None:
        st.warning("Please complete the preprocessing step first.")
        return

    # Make a copy for feature engineering
    if 'feature_data' not in st.session_state:
        st.session_state.feature_data = st.session_state.processed_data.copy()

    # Display current features
    st.subheader("Current Features")
    st.dataframe(st.session_state.feature_data.head(5), use_container_width=True)

    # Feature selection
    st.subheader("Feature Selection")

    # Select target variable for supervised learning models
    if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
        st.write("For supervised learning models, select a target variable:")

        # Get appropriate columns for the target based on the model type
        if st.session_state.model_type == "Linear Regression":
            target_cols = st.session_state.feature_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            target_type = "continuous"
        else:
            # For logistic regression, ideally we need binary columns, but let's include all
            target_cols = st.session_state.feature_data.columns.tolist()
            target_type = "binary"

        if not target_cols:
            st.error(f"No suitable {target_type} columns available for {st.session_state.model_type}")
            return

        target_column = st.selectbox(
            "Select target variable:",
            options=target_cols
        )

        st.session_state.target_column = target_column

        # For logistic regression, check if target is binary or can be converted to binary
        if st.session_state.model_type == "Logistic Regression":
            unique_values = st.session_state.feature_data[target_column].nunique()

            if unique_values > 2:
                st.warning(f"The selected target has {unique_values} unique values, but Logistic Regression works best with binary targets.")

                binarize_method = st.selectbox(
                    "How would you like to binarize the target?",
                    options=["Convert based on threshold", "One-vs-Rest (keep as is for now)"],
                    index=0
                )

                if binarize_method == "Convert based on threshold":
                    if st.session_state.feature_data[target_column].dtype in ['int64', 'float64']:
                        threshold = st.slider(
                            "Threshold value:",
                            min_value=float(st.session_state.feature_data[target_column].min()),
                            max_value=float(st.session_state.feature_data[target_column].max()),
                            value=float(st.session_state.feature_data[target_column].median())
                        )

                        if st.button("Binarize target", key="binarize_target"):
                            st.session_state.feature_data[f"{target_column}_binary"] = (st.session_state.feature_data[target_column] > threshold).astype(int)
                            st.session_state.target_column = f"{target_column}_binary"
                            st.success(f"Created binary target '{target_column}_binary' using threshold {threshold}")
                            st.rerun()
                    else:
                        st.error("Cannot binarize non-numeric column using threshold. Please select a different target or method.")

    # Feature selection for both supervised and unsupervised learning
    st.subheader("Select Features for Model")

    available_columns = st.session_state.feature_data.columns.tolist()

    # Remove target column from feature list for supervised models
    if st.session_state.model_type in ["Linear Regression", "Logistic Regression"] and st.session_state.target_column in available_columns:
        available_columns.remove(st.session_state.target_column)

    selected_features = st.multiselect(
        "Select features to include in the model:",
        options=available_columns,
        default=available_columns[:5] if len(available_columns) > 5 else available_columns
    )

    st.session_state.selected_features = selected_features

    if not selected_features:
        st.warning("Please select at least one feature.")
        return

    # Feature transformation options
    st.subheader("Feature Transformation")

    transformation_options = st.multiselect(
        "Select transformations to apply:",
        options=["Standardization (Z-score)", "Normalization (Min-Max)", "Log Transform", "One-Hot Encoding"],
        default=[]
    )

    transformed_features = selected_features.copy()

    if transformation_options:
        for transformation in transformation_options:
            if transformation == "Standardization (Z-score)":
                numeric_features = st.session_state.feature_data[selected_features].select_dtypes(include=['int64', 'float64']).columns.tolist()

                if numeric_features:
                    selected_for_standardization = st.multiselect(
                        "Select numeric features to standardize:",
                        options=numeric_features,
                        default=numeric_features
                    )

                    if st.button("Apply Standardization", key="apply_standardization"):
                        for feature in selected_for_standardization:
                            new_col_name = f"{feature}_standardized"
                            mean_val = st.session_state.feature_data[feature].mean()
                            std_val = st.session_state.feature_data[feature].std()
                            st.session_state.feature_data[new_col_name] = (st.session_state.feature_data[feature] - mean_val) / std_val

                            # Update transformed features list
                            transformed_features.remove(feature)
                            transformed_features.append(new_col_name)

                        st.success("Standardization applied successfully!")
                        st.session_state.selected_features = transformed_features
                        st.rerun()
                else:
                    st.warning("No numeric features selected for standardization.")

            elif transformation == "Normalization (Min-Max)":
                numeric_features = st.session_state.feature_data[selected_features].select_dtypes(include=['int64', 'float64']).columns.tolist()

                if numeric_features:
                    selected_for_normalization = st.multiselect(
                        "Select numeric features to normalize:",
                        options=numeric_features,
                        default=numeric_features
                    )

                    if st.button("Apply Normalization", key="apply_normalization"):
                        for feature in selected_for_normalization:
                            new_col_name = f"{feature}_normalized"
                            min_val = st.session_state.feature_data[feature].min()
                            max_val = st.session_state.feature_data[feature].max()
                            st.session_state.feature_data[new_col_name] = (st.session_state.feature_data[feature] - min_val) / (max_val - min_val)

                            # Update transformed features list
                            transformed_features.remove(feature)
                            transformed_features.append(new_col_name)

                        st.success("Normalization applied successfully!")
                        st.session_state.selected_features = transformed_features
                        st.rerun()
                else:
                    st.warning("No numeric features selected for normalization.")

            elif transformation == "Log Transform":
                numeric_features = st.session_state.feature_data[selected_features].select_dtypes(include=['int64', 'float64']).columns.tolist()

                if numeric_features:
                    selected_for_log = st.multiselect(
                        "Select numeric features for log transformation:",
                        options=numeric_features,
                        default=[]
                    )

                    if st.button("Apply Log Transform", key="apply_log"):
                        for feature in selected_for_log:
                            # Check if all values are positive
                            if (st.session_state.feature_data[feature] <= 0).any():
                                min_val = st.session_state.feature_data[feature].min()
                                if min_val <= 0:
                                    offset = abs(min_val) + 1  # Add 1 to ensure positive values
                                    st.warning(f"Adding offset of {offset} to ensure positive values for log transform of {feature}")
                                    new_col_name = f"{feature}_log"
                                    st.session_state.feature_data[new_col_name] = np.log(st.session_state.feature_data[feature] + offset)
                            else:
                                new_col_name = f"{feature}_log"
                                st.session_state.feature_data[new_col_name] = np.log(st.session_state.feature_data[feature])

                            # Update transformed features list
                            transformed_features.remove(feature)
                            transformed_features.append(new_col_name)

                        st.success("Log transformation applied successfully!")
                        st.session_state.selected_features = transformed_features
                        st.rerun()
                else:
                    st.warning("No numeric features selected for log transformation.")

            elif transformation == "One-Hot Encoding":
                categorical_features = st.session_state.feature_data[selected_features].select_dtypes(exclude=['int64', 'float64']).columns.tolist()

                if not categorical_features:
                    # Also include numeric features with low cardinality as potential categorical features
                    for feature in st.session_state.feature_data[selected_features].select_dtypes(include=['int64']).columns:
                        if st.session_state.feature_data[feature].nunique() < 10:
                            categorical_features.append(feature)

                if categorical_features:
                    selected_for_onehot = st.multiselect(
                        "Select categorical features for one-hot encoding:",
                        options=categorical_features,
                        default=categorical_features
                    )

                    if st.button("Apply One-Hot Encoding", key="apply_onehot"):
                        # Store original feature data before transformation
                        original_feature_data = st.session_state.feature_data.copy()

                        # Apply one-hot encoding
                        for feature in selected_for_onehot:
                            # Get dummies for this feature
                            dummies = pd.get_dummies(st.session_state.feature_data[feature], prefix=feature, drop_first=False)

                            # Add to feature data
                            st.session_state.feature_data = pd.concat([st.session_state.feature_data, dummies], axis=1)

                            # Update transformed features list
                            transformed_features.remove(feature)
                            transformed_features.extend(dummies.columns.tolist())

                        # Remove original categorical columns
                        st.session_state.feature_data = st.session_state.feature_data.drop(columns=selected_for_onehot)

                        st.success("One-hot encoding applied successfully!")
                        st.session_state.selected_features = transformed_features
                        st.rerun()
                else:
                    st.warning("No categorical features identified for one-hot encoding.")

    # Feature creation (bonus)
    st.subheader("Feature Creation (Optional)")

    if st.checkbox("Create polynomial features", key="create_poly"):
        numeric_features = st.session_state.feature_data[selected_features].select_dtypes(include=['int64', 'float64']).columns.tolist()

        if numeric_features:
            selected_for_poly = st.multiselect(
                "Select numeric features for polynomial transformation:",
                options=numeric_features,
                default=[]
            )

            if selected_for_poly:
                poly_degree = st.slider("Polynomial degree:", min_value=2, max_value=5, value=2)

                if st.button("Create Polynomial Features", key="apply_poly"):
                    for feature in selected_for_poly:
                        for degree in range(2, poly_degree + 1):
                            new_col_name = f"{feature}^{degree}"
                            st.session_state.feature_data[new_col_name] = st.session_state.feature_data[feature] ** degree

                            # Add to transformed features list
                            transformed_features.append(new_col_name)

                    st.success("Polynomial features created successfully!")
                    st.session_state.selected_features = transformed_features
                    st.rerun()
        else:
            st.warning("No numeric features available for polynomial transformation.")

    if st.checkbox("Create interaction terms", key="create_interaction"):
        numeric_features = st.session_state.feature_data[selected_features].select_dtypes(include=['int64', 'float64']).columns.tolist()

        if len(numeric_features) >= 2:
            col1 = st.selectbox("Select first feature:", options=numeric_features, key="interact_col1")
            col2 = st.selectbox("Select second feature:", options=[f for f in numeric_features if f != col1], key="interact_col2")

            if st.button("Create Interaction Term", key="apply_interaction"):
                new_col_name = f"{col1}_x_{col2}"
                st.session_state.feature_data[new_col_name] = st.session_state.feature_data[col1] * st.session_state.feature_data[col2]

                # Add to transformed features list
                transformed_features.append(new_col_name)

                st.success("Interaction term created successfully!")
                st.session_state.selected_features = transformed_features
                st.rerun()
        else:
            st.warning("Need at least 2 numeric features to create interaction terms.")

    # Save the final feature set
    st.subheader("Final Feature Set")

    # Display the selected features and their preview
    if selected_features:
        st.write("Selected Features:", ", ".join(st.session_state.selected_features))

        if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
            st.write("Target Variable:", st.session_state.target_column)

        st.write("Feature Preview:")
        feature_preview = st.session_state.feature_data[st.session_state.selected_features].head(5)
        st.dataframe(feature_preview, use_container_width=True)

        # Visualize feature distributions
        st.subheader("Feature Distributions")

        numeric_features = st.session_state.feature_data[st.session_state.selected_features].select_dtypes(include=['int64', 'float64']).columns.tolist()

        if numeric_features:
            selected_feature = st.selectbox("Select feature to visualize:", options=numeric_features)
            st.plotly_chart(visualization.create_feature_distribution(st.session_state.feature_data, selected_feature), use_container_width=True)

        # Display correlation between features
        if len(numeric_features) > 1:
            st.subheader("Feature Correlations")
            st.plotly_chart(visualization.create_correlation_heatmap(st.session_state.feature_data[numeric_features]), use_container_width=True)

        # Save feature engineering results
        if st.button("Save Features and Proceed", key="save_features", use_container_width=True):
            # Prepare X and y for modeling
            if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
                st.session_state.X = st.session_state.feature_data[st.session_state.selected_features]
                st.session_state.y = st.session_state.feature_data[st.session_state.target_column]
            else:  # K-Means
                st.session_state.X = st.session_state.feature_data[st.session_state.selected_features]
                st.session_state.y = None

            st.success("Feature engineering completed! Ready for train/test split.")
            st.session_state.current_step = 4
            st.rerun()
    else:
        st.warning("Please select at least one feature to proceed.")

def train_test_split_step():
    st.header("Step 4: Train/Test Split 🔪")

    if st.session_state.X is None:
        st.warning("Please complete the feature engineering step first.")
        return

    st.write("Splitting the data into training and testing sets is crucial for evaluating model performance.")

    # Check for datetime columns and display info
    datetime_cols = []
    for col in st.session_state.X.columns:
        if is_timestamp_column(st.session_state.X[col]):
            datetime_cols.append(col)

    if datetime_cols:
        st.info(f"Datetime columns detected: {datetime_cols}")

        # Auto-convert datetime columns to numeric features
        if st.checkbox("Automatically convert datetime columns to numeric features", value=True, key="auto_convert_datetime"):
            with st.spinner("Converting datetime features to numeric..."):
                # Process timestamp columns
                X_processed, _ = handle_timestamp_columns(st.session_state.X)
                st.session_state.X = X_processed

                st.success("Datetime features converted to numeric features")
                st.dataframe(st.session_state.X.head(), use_container_width=True)

                # Store timestamp columns in session state for reference
                st.session_state.timestamp_columns = datetime_cols
        st.write("For machine learning models, datetime columns will be automatically converted to numeric features during training.")

    # Different instructions based on model type
    if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
        st.info("For supervised learning, we'll split both features (X) and target (y) into training and testing sets.")
    else:  # K-Means
        st.info("For unsupervised learning like K-Means, we'll split only the feature data for validation purposes.")

    # Display feature data shape
    st.write(f"Feature data shape: {st.session_state.X.shape}")

    if st.session_state.y is not None:
        st.write(f"Target data shape: {st.session_state.y.shape}")

    # Split configuration
    st.subheader("Configure Split")

    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider("Test Size (%):", min_value=10, max_value=50, value=20) / 100

    with col2:
        random_state = st.number_input("Random State (for reproducibility):", min_value=0, max_value=999, value=42)

    stratify = False
    if st.session_state.model_type == "Logistic Regression" and st.session_state.y is not None:
        unique_values = st.session_state.y.nunique()

        if unique_values < 10:  # Only offer stratification for categorical targets with reasonable number of classes
            stratify = st.checkbox("Use stratified sampling (maintains class distribution)?", value=True)

    # Button to perform the split
    if st.button("Perform Train/Test Split", key="do_split", use_container_width=True):
        if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
            try:
                # For supervised models
                if stratify:
                    st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = data_processing.perform_train_test_split(
                        st.session_state.X, 
                        st.session_state.y, 
                        test_size=test_size, 
                        random_state=random_state,
                        stratify=st.session_state.y
                    )
                else:
                    st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = data_processing.perform_train_test_split(
                        st.session_state.X, 
                        st.session_state.y, 
                        test_size=test_size, 
                        random_state=random_state
                    )

                st.success("Train/test split completed successfully!")

            except Exception as e:
                st.error(f"Error performing train/test split: {e}")
                return
        else:
            # For unsupervised models (K-Means)
            try:
                st.session_state.X_train, st.session_state.X_test = data_processing.perform_train_test_split(
                    st.session_state.X, 
                    None, 
                    test_size=test_size, 
                    random_state=random_state
                )

                st.success("Train/test split completed successfully!")

            except Exception as e:
                st.error(f"Error performing train/test split: {e}")
                return

    # Display split results if available
    if st.session_state.X_train is not None and st.session_state.X_test is not None:
        st.subheader("Split Results")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Training Set:")
            st.write(f"X_train shape: {st.session_state.X_train.shape}")
            if st.session_state.y_train is not None:
                st.write(f"y_train shape: {st.session_state.y_train.shape}")

        with col2:
            st.write("Testing Set:")
            st.write(f"X_test shape: {st.session_state.X_test.shape}")
            if st.session_state.y_test is not None:
                st.write(f"y_test shape: {st.session_state.y_test.shape}")

        # Visualize the split
        st.subheader("Train/Test Split Visualization")

        # Pie chart showing the split
        train_size = len(st.session_state.X_train)
        test_size = len(st.session_state.X_test)
        total_size = train_size + test_size

        st.plotly_chart(visualization.create_train_test_split_chart(train_size, test_size), use_container_width=True)

        # For classification, show class distribution in train/test
        if st.session_state.model_type == "Logistic Regression" and st.session_state.y_train is not None:
            st.subheader("Class Distribution Across Train/Test Sets")

            # Get class counts
            train_classes = st.session_state.y_train.value_counts()
            test_classes = st.session_state.y_test.value_counts()

            # Create a DataFrame for display
            class_dist = pd.DataFrame({
                'Training Set': train_classes / len(st.session_state.y_train),
                'Testing Set': test_classes / len(st.session_state.y_test)
            })

            # Normalize to get percentages
            class_dist = class_dist * 100

            st.plotly_chart(visualization.create_class_distribution_chart(class_dist), use_container_width=True)

        # Proceed button
        if st.button("Proceed to Model Training", key="proceed_to_training", use_container_width=True):
            st.session_state.current_step = 5
            st.rerun()

def model_training_step():
    st.header("Step 5: Model Training 🧠")

    if st.session_state.X_train is None or (st.session_state.model_type in ["Linear Regression", "Logistic Regression"] and st.session_state.y_train is None):
        st.warning("Please complete the train/test split step first.")
        return

    st.subheader(f"Training a {st.session_state.model_type} Model")

    # Configure model hyperparameters based on model type
    if st.session_state.model_type == "Linear Regression":
        st.write("Linear Regression predicts continuous target values based on linear relationships with features.")

        with st.expander("Model Configuration"):
            fit_intercept = st.checkbox("Fit Intercept", value=True)
            normalize = st.checkbox("Normalize", value=False)

        # Train button
        if st.button("Train Linear Regression Model", key="train_linear", use_container_width=True):
            with st.spinner("Training model..."):
                try:
                    # Check for and handle datetime columns in the training data
                    X_train_processed = st.session_state.X_train.copy()
                    X_test_processed = st.session_state.X_test.copy()

                    # Find timestamp/datetime columns
                    datetime_cols = []
                    for col in X_train_processed.columns:
                        if pd.api.types.is_datetime64_any_dtype(X_train_processed[col]) or (
                            hasattr(X_train_processed[col], 'dtype') and 
                            isinstance(X_train_processed[col].dtype, pd.DatetimeTZDtype)):
                            datetime_cols.append(col)

                    # Handle datetime columns by extracting useful numeric features
                    if datetime_cols:
                        st.info(f"Converting datetime features to numeric: {datetime_cols}")
                        for col in datetime_cols:
                            # Extract useful components from the datetime
                            if pd.api.types.is_datetime64_any_dtype(X_train_processed[col]) or (
                                hasattr(X_train_processed[col], 'dtype') and 
                                isinstance(X_train_processed[col].dtype, pd.DatetimeTZDtype)):

                                # For training set
                                X_train_processed[f"{col}_year"] = X_train_processed[col].dt.year
                                X_train_processed[f"{col}_month"] = X_train_processed[col].dt.month
                                X_train_processed[f"{col}_day"] = X_train_processed[col].dt.day
                                X_train_processed[f"{col}_dayofweek"] = X_train_processed[col].dt.dayofweek

                                # For test set
                                X_test_processed[f"{col}_year"] = X_test_processed[col].dt.year
                                X_test_processed[f"{col}_month"] = X_test_processed[col].dt.month
                                X_test_processed[f"{col}_day"] = X_test_processed[col].dt.day
                                X_test_processed[f"{col}_dayofweek"] = X_test_processed[col].dt.dayofweek

                                # Drop the original datetime column
                                X_train_processed = X_train_processed.drop(columns=[col])
                                X_test_processed = X_test_processed.drop(columns=[col])

                    # Check for other non-numeric columns (like object/string types)
                    non_numeric_cols = X_train_processed.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
                    if non_numeric_cols:
                        st.warning(f"Removing non-numeric features that can't be used in Linear Regression: {non_numeric_cols}")
                        X_train_processed = X_train_processed.select_dtypes(include=['int64', 'float64'])
                        X_test_processed = X_test_processed.select_dtypes(include=['int64', 'float64'])

                    # Train the model with the processed data
                    st.session_state.model, st.session_state.feature_importance = ml_utils.train_linear_regression(
                        X_train_processed, 
                        st.session_state.y_train,
                        fit_intercept=fit_intercept,
                        normalize=normalize
                    )

                    st.success("Linear Regression model trained successfully!")

                    # Make predictions on the processed test set
                    st.session_state.predictions = st.session_state.model.predict(X_test_processed)

                    # Evaluate the model
                    st.session_state.evaluation_metrics = ml_utils.evaluate_regression(
                        st.session_state.y_test, 
                        st.session_state.predictions
                    )

                    # Store the processed features for later use
                    st.session_state.X_train_processed = X_train_processed
                    st.session_state.X_test_processed = X_test_processed

                except Exception as e:
                    st.error(f"Error training model: {e}")
                    return

    elif st.session_state.model_type == "Logistic Regression":
        st.write("Logistic Regression predicts categorical target values, especially effective for binary classification.")

        with st.expander("Model Configuration"):
            solver = st.selectbox(
                "Solver:",
                options=["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
                index=0
            )

            C = st.slider(
                "Regularization Strength (C):",
                min_value=0.01, 
                max_value=10.0,
                value=1.0,
                step=0.01
            )

            max_iter = st.slider(
                "Maximum Iterations:",
                min_value=100,
                max_value=2000,
                value=1000,
                step=100
            )

        # Train button
        if st.button("Train Logistic Regression Model", key="train_logistic", use_container_width=True):
            with st.spinner("Training model..."):
                try:
                    st.session_state.model, st.session_state.feature_importance = ml_utils.train_logistic_regression(
                        st.session_state.X_train, 
                        st.session_state.y_train,
                        C=C,
                        solver=solver,
                        max_iter=max_iter
                    )

                    st.success("Logistic Regression model trained successfully!")

                    # Make predictions on the test set
                    st.session_state.predictions = st.session_state.model.predict(st.session_state.X_test)

                    # Get probability predictions if it's binary classification
                    if len(np.unique(st.session_state.y_train)) == 2:
                        st.session_state.prediction_probs = st.session_state.model.predict_proba(st.session_state.X_test)[:, 1]

                    # Evaluate the model
                    st.session_state.evaluation_metrics = ml_utils.evaluate_classification(
                        st.session_state.y_test, 
                        st.session_state.predictions
                    )

                except Exception as e:
                    st.error(f"Error training model: {e}")
                    return

    elif st.session_state.model_type == "K-Means Clustering":
        st.write("K-Means Clustering groups similar data points together, revealing patterns in the data.")

        with st.expander("Model Configuration"):
            n_clusters = st.session_state.n_clusters if 'n_clusters' in st.session_state else 3
            n_clusters = st.slider(
                "Number of Clusters (k):",
                min_value=2,
                max_value=10,
                value=n_clusters
            )

            init = st.selectbox(
                "Initialization Method:",
                options=["k-means++", "random"],
                index=0
            )

            max_iter = st.slider(
                "Maximum Iterations:",
                min_value=100,
                max_value=1000,
                value=300,
                step=50
            )

            n_init = st.slider(
                "Number of Initializations:",
                min_value=1,
                max_value=20,
                value=10
            )

        # Train button
        if st.button("Train K-Means Clustering Model", key="train_kmeans", use_container_width=True):
            with st.spinner("Training model..."):
                try:
                    st.session_state.model, st.session_state.cluster_centers = ml_utils.train_kmeans_clustering(
                        st.session_state.X_train,
                        n_clusters=n_clusters,
                        init=init,
                        max_iter=max_iter,
                        n_init=n_init
                    )

                    st.success("K-Means Clustering model trained successfully!")

                    # Make predictions on the test set
                    st.session_state.predictions = st.session_state.model.predict(st.session_state.X_test)

                    # For K-Means, store the silhouette score and inertia
                    st.session_state.evaluation_metrics = ml_utils.evaluate_clustering(
                        st.session_state.X_test, 
                        st.session_state.predictions
                    )

                except Exception as e:
                    st.error(f"Error training model: {e}")
                    return

    # Display training results if model is trained
    if st.session_state.model is not None:
        st.subheader("Training Complete")

        # Display appropriate training info based on model type
        if st.session_state.model_type == "Linear Regression":
            st.write("Model Parameters:")

            # Check if we're using the processed data (which we should be)
            if hasattr(st.session_state, 'X_train_processed'):
                # Use the processed columns that were actually used for training
                feature_columns = st.session_state.X_train_processed.columns
            else:
                # Fallback to original columns
                feature_columns = st.session_state.X_train.columns

            # Make sure the length of columns matches the coefficients
            if len(feature_columns) == len(st.session_state.model.coef_):
                params = pd.DataFrame({
                    'Feature': feature_columns,
                    'Coefficient': st.session_state.model.coef_
                })
                st.dataframe(params, use_container_width=True)
            else:
                st.warning("Feature columns and coefficients have different lengths. Showing coefficients only.")
                params = pd.DataFrame({
                    'Coefficient': st.session_state.model.coef_
                })
                st.dataframe(params, use_container_width=True)

            st.write(f"Intercept: {st.session_state.model.intercept_:.4f}")

            # Display feature importance
            if st.session_state.feature_importance is not None:
                st.subheader("Feature Importance")

                # Use the same columns as above for consistency
                if hasattr(st.session_state, 'X_train_processed') and len(st.session_state.X_train_processed.columns) == len(st.session_state.feature_importance):
                    feature_cols_for_plot = st.session_state.X_train_processed.columns
                elif len(st.session_state.X_train.columns) == len(st.session_state.feature_importance):
                    feature_cols_for_plot = st.session_state.X_train.columns
                else:
                    # If columns don't match, use generic feature names
                    feature_cols_for_plot = [f"Feature {i+1}" for i in range(len(st.session_state.feature_importance))]

                st.plotly_chart(visualization.create_feature_importance_chart(
                    feature_cols_for_plot,
                    st.session_state.feature_importance
                ), use_container_width=True)

        elif st.session_state.model_type == "Logistic Regression":
            st.write("Model Parameters:")

            # For multiclass, coefficients are different
            if len(st.session_state.model.coef_) == 1:
                params = pd.DataFrame({
                    'Feature': st.session_state.X_train.columns,
                    'Coefficient': st.session_state.model.coef_[0]
                })
            else:
                params_data = []
                for i, coef in enumerate(st.session_state.model.coef_):
                    for j, feature in enumerate(st.session_state.X_train.columns):
                        params_data.append({
                            'Class': i,
                            'Feature': feature,
                            'Coefficient': coef[j]
                        })
                params = pd.DataFrame(params_data)

            st.dataframe(params, use_container_width=True)

            # Display feature importance
            if st.session_state.feature_importance is not None:
                st.subheader("Feature Importance")
                st.plotly_chart(visualization.create_feature_importance_chart(
                    st.session_state.X_train.columns,
                    st.session_state.feature_importance
                ), use_container_width=True)

        elif st.session_state.model_type == "K-Means Clustering":
            st.write("Cluster Centers:")
            centers = pd.DataFrame(
                st.session_state.cluster_centers,
                columns=st.session_state.X_train.columns
            )
            centers.index.name = "Cluster"
            st.dataframe(centers, use_container_width=True)

            # Display cluster distribution
            cluster_counts = pd.Series(st.session_state.predictions).value_counts().sort_index()

            st.subheader("Cluster Distribution")
            st.plotly_chart(visualization.create_cluster_distribution_chart(cluster_counts), use_container_width=True)

        # Notification of model training completion
        st.success("Model training completed! You can now proceed to evaluation.")

        # Proceed button
        if st.button("Proceed to Model Evaluation", key="proceed_to_evaluation", use_container_width=True):
            st.session_state.current_step = 6
            st.rerun()

def model_evaluation_step():
    st.header("Step 6: Model Evaluation 📈")

    if st.session_state.model is None or st.session_state.predictions is None:
        st.warning("Please complete the model training step first.")
        return

    st.subheader(f"Evaluating the {st.session_state.model_type} Model")

    # Display evaluation metrics based on model type
    if st.session_state.model_type == "Linear Regression":
        if st.session_state.evaluation_metrics is not None:
            mae, mse, rmse, r2 = st.session_state.evaluation_metrics

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
                st.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
            with col2:
                st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.4f}")
                st.metric("R² Score", f"{r2:.4f}")

            # Visualization: Actual vs Predicted
            st.subheader("Actual vs. Predicted Values")

            actual_vs_predicted_df = pd.DataFrame({
                'Actual': st.session_state.y_test,
                'Predicted': st.session_state.predictions
            })

            st.plotly_chart(visualization.create_actual_vs_predicted_chart(
                actual_vs_predicted_df, 
                model_type="regression"
            ), use_container_width=True)

            # Residuals plot
            st.subheader("Residual Analysis")

            residuals = st.session_state.y_test - st.session_state.predictions
            residual_df = pd.DataFrame({
                'Predicted': st.session_state.predictions,
                'Residuals': residuals
            })

            st.plotly_chart(visualization.create_residual_plot(residual_df), use_container_width=True)

            # Residual distribution
            st.subheader("Residual Distribution")
            st.plotly_chart(visualization.create_residual_distribution(residuals), use_container_width=True)

    elif st.session_state.model_type == "Logistic Regression":
        if st.session_state.evaluation_metrics is not None:
            accuracy, precision, recall, f1 = st.session_state.evaluation_metrics

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
                st.metric("Precision", f"{precision:.4f}")
            with col2:
                st.metric("Recall", f"{recall:.4f}")
                st.metric("F1 Score", f"{f1:.4f}")

            # Confusion Matrix
            st.subheader("Confusion Matrix")

            st.plotly_chart(visualization.create_confusion_matrix(
                st.session_state.y_test, 
                st.session_state.predictions
            ), use_container_width=True)

            # ROC Curve (for binary classification)
            if len(np.unique(st.session_state.y_test)) == 2 and hasattr(st.session_state, 'prediction_probs'):
                st.subheader("ROC Curve")

                st.plotly_chart(visualization.create_roc_curve(
                    st.session_state.y_test, 
                    st.session_state.prediction_probs
                ), use_container_width=True)

    elif st.session_state.model_type == "K-Means Clustering":
        if st.session_state.evaluation_metrics is not None:
            silhouette, inertia = st.session_state.evaluation_metrics

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Silhouette Score", f"{silhouette:.4f}")
            with col2:
                st.metric("Inertia", f"{inertia:.2f}")

            # Visualization: Cluster assignments (for 2D data)
            if st.session_state.X_test.shape[1] >= 2:
                st.subheader("Cluster Visualization")

                # For higher dimensional data, use dimensionality reduction
                if st.session_state.X_test.shape[1] > 2:
                    st.info("Using PCA to visualize high-dimensional data in 2D")

                # Create a dataframe with the cluster assignments
                cluster_df = pd.DataFrame(st.session_state.X_test.copy())
                cluster_df['Cluster'] = st.session_state.predictions

                # Visualize clusters
                st.plotly_chart(visualization.create_cluster_plot(
                    cluster_df, 
                    st.session_state.cluster_centers
                ), use_container_width=True)

    # Model interpretation and insights
    st.subheader("Model Insights")

    if st.session_state.model_type == "Linear Regression":
        st.write("Key insights from the Linear Regression model:")

        # Feature importance-based insights
        if st.session_state.feature_importance is not None:
            # Get the appropriate feature columns
            if hasattr(st.session_state, 'X_train_processed') and len(st.session_state.X_train_processed.columns) == len(st.session_state.feature_importance):
                feature_cols = st.session_state.X_train_processed.columns
            elif len(st.session_state.X_train.columns) == len(st.session_state.feature_importance):
                feature_cols = st.session_state.X_train.columns
            else:
                feature_cols = [f"Feature {i+1}" for i in range(len(st.session_state.feature_importance))]

            feature_impact = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': np.abs(st.session_state.feature_importance)
            }).sort_values('Importance', ascending=False)

            top_features = feature_impact.head(3)

            st.write(f"1. The most important features for predicting {st.session_state.target_column} are: {', '.join(top_features['Feature'].values)}")

            # Direction of influence
            positive_features = pd.DataFrame({
                'Feature': feature_cols,
                'Coefficient': st.session_state.feature_importance
            }).query("Coefficient > 0").sort_values('Coefficient', ascending=False).head(3)

            negative_features = pd.DataFrame({
                'Feature': feature_cols,
                'Coefficient': st.session_state.feature_importance
            }).query("Coefficient < 0").sort_values('Coefficient').head(3)

            if not positive_features.empty:
                st.write(f"2. Features with positive influence (increase in feature leads to increase in target): {', '.join(positive_features['Feature'].values)}")

            if not negative_features.empty:
                st.write(f"3. Features with negative influence (increase in feature leads to decrease in target): {', '.join(negative_features['Feature'].values)}")

        # R-squared interpretation
        if 'evaluation_metrics' in st.session_state and st.session_state.evaluation_metrics is not None:
            r2 = st.session_state.evaluation_metrics[3]

            if r2 >= 0.8:
                st.write(f"4. The model has strong predictive power with R² of {r2:.2f}, explaining {r2*100:.1f}% of the variance in the target variable.")
            elif r2 >= 0.5:
                st.write(f"4. The model has moderate predictive power with R² of {r2:.2f}, explaining {r2*100:.1f}% of the variance in the target variable.")
            else:
                st.write(f"4. The model has limited predictive power with R² of {r2:.2f}, explaining only {r2*100:.1f}% of the variance in the target variable.")

    elif st.session_state.model_type == "Logistic Regression":
        st.write("Key insights from the Logistic Regression model:")

        # Feature importance-based insights
        if st.session_state.feature_importance is not None:
            feature_impact = pd.DataFrame({
                'Feature': st.session_state.X_train.columns,
                'Importance': np.abs(st.session_state.feature_importance)
            }).sort_values('Importance', ascending=False)

            top_features = feature_impact.head(3)

            st.write(f"1. The most important features for classifying {st.session_state.target_column} are: {', '.join(top_features['Feature'].values)}")

            # Direction of influence
            positive_features = pd.DataFrame({
                'Feature': st.session_state.X_train.columns,
                'Coefficient': st.session_state.feature_importance
            }).query("Coefficient > 0").sort_values('Coefficient', ascending=False).head(3)

            negative_features = pd.DataFrame({
                'Feature': st.session_state.X_train.columns,
                'Coefficient': st.session_state.feature_importance
            }).query("Coefficient < 0").sort_values('Coefficient').head(3)

            if not positive_features.empty:
                st.write(f"2. Features that increase probability of positive class: {', '.join(positive_features['Feature'].values)}")

            if not negative_features.empty:
                st.write(f"3. Features that decrease probability of positive class: {', '.join(negative_features['Feature'].values)}")

        # Accuracy interpretation
        if 'evaluation_metrics' in st.session_state and st.session_state.evaluation_metrics is not None:
            accuracy = st.session_state.evaluation_metrics[0]

            if accuracy >= 0.8:
                st.write(f"4. The model has strong classification performance with an accuracy of {accuracy:.2f} ({accuracy*100:.1f}%).")
            elif accuracy >= 0.6:
                st.write(f"4. The model has moderate classification performance with an accuracy of {accuracy:.2f} ({accuracy*100:.1f}%).")
            else:
                st.write(f"4. The model has limited classification performance with an accuracy of {accuracy:.2f} ({accuracy*100:.1f}%).")

    elif st.session_state.model_type == "K-Means Clustering":
        st.write("Key insights from the K-Means Clustering model:")

        # Cluster interpretation
        if st.session_state.cluster_centers is not None:
            centers = pd.DataFrame(
                st.session_state.cluster_centers,
                columns=st.session_state.X_train.columns
            )

            # Identify distinguishing features for each cluster
            cluster_insights = []

            for i in range(len(centers)):
                cluster_center = centers.iloc[i]

                # Find the top distinguishing features (furthest from the global mean)
                distinguishing_features = []

                for col in centers.columns:
                    global_mean = st.session_state.X.loc[:, col].mean()
                    distance_from_mean = abs(cluster_center[col] - global_mean)
                    distinguishing_features.append((col, distance_from_mean, cluster_center[col]))

                # Sort by distance from global mean
                distinguishing_features.sort(key=lambda x: x[1], reverse=True)

                # Get top features
                top_features = distinguishing_features[:3]

                feature_descriptions = []
                for feature, _, value in top_features:
                    global_mean = st.session_state.X.loc[:, feature].mean()
                    if value > global_mean:
                        feature_descriptions.append(f"high {feature}")
                    else:
                        feature_descriptions.append(f"low {feature}")

                cluster_insights.append(f"Cluster {i} is characterized by: {', '.join(feature_descriptions)}")

            for i, insight in enumerate(cluster_insights, 1):
                st.write(f"{i}. {insight}")

        # Silhouette score interpretation
        if 'evaluation_metrics' in st.session_state and st.session_state.evaluation_metrics is not None:
            silhouette = st.session_state.evaluation_metrics[0]

            if silhouette >= 0.5:
                st.write(f"{len(cluster_insights) + 1}. The clustering has good separation with a silhouette score of {silhouette:.2f}.")
            elif silhouette >= 0.3:
                st.write(f"{len(cluster_insights) + 1}. The clustering has moderate separation with a silhouette score of {silhouette:.2f}.")
            else:
                st.write(f"{len(cluster_insights) + 1}. The clustering has poor separation with a silhouette score of {silhouette:.2f}.")

    # Downloadable results (Bonus)
    st.subheader("Download Results")

    if st.button("Generate Downloadable Results", key="generate_results"):
        try:
            # Create a buffer to store the results
            buffer = io.BytesIO()

            # Create a writer to write to Excel
            with pd.ExcelWriter(buffer) as writer:
                # Convert timezone-aware datetimes to naive before writing
                data_to_write = convert_tz_aware_to_naive(st.session_state.data.head(100))
                # Write data
                data_to_write.to_excel(writer, sheet_name="Original Data", index=False)

                # Write model results
                results_df = pd.DataFrame()

                if st.session_state.model_type in ["Linear Regression", "Logistic Regression"]:
                    # Create results dataframe with actual and predicted values
                    results_df['Actual'] = st.session_state.y_test.reset_index(drop=True)
                    results_df['Predicted'] = st.session_state.predictions

                    if st.session_state.model_type == "Logistic Regression" and hasattr(st.session_state, 'prediction_probs'):
                        results_df['Probability'] = st.session_state.prediction_probs

                    results_df.to_excel(writer, sheet_name="Model Results", index=False)

                    # Feature importances
                    if st.session_state.feature_importance is not None:
                        importance_df = pd.DataFrame({
                            'Feature': st.session_state.X_train.columns,
                            'Importance': st.session_state.feature_importance
                        })
                        importance_df.to_excel(writer, sheet_name="Feature Importance", index=False)

                    # Metrics
                    if st.session_state.evaluation_metrics is not None:
                        if st.session_state.model_type == "Linear Regression":
                            metrics_df = pd.DataFrame({
                                'Metric': ['MAE', 'MSE', 'RMSE', 'R²'],
                                'Value': st.session_state.evaluation_metrics
                            })
                        else:
                            metrics_df = pd.DataFrame({
                                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                                'Value': st.session_state.evaluation_metrics
                            })

                        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

                elif st.session_state.model_type == "K-Means Clustering":
                    # Create results dataframe with cluster assignments
                    cluster_results = pd.DataFrame(st.session_state.X_test.copy())
                    cluster_results['Cluster'] = st.session_state.predictions

                    cluster_results.to_excel(writer, sheet_name="Clusters", index=False)

                    # Cluster centers
                    if st.session_state.cluster_centers is not None:
                        centers_df = pd.DataFrame(
                            st.session_state.cluster_centers,
                            columns=st.session_state.X_train.columns
                        )
                        centers_df.index.name = "Cluster"

                        centers_df.to_excel(writer, sheet_name="Cluster Centers")

                    # Metrics
                    if st.session_state.evaluation_metrics is not None:
                        metrics_df = pd.DataFrame({
                            'Metric': ['Silhouette Score', 'Inertia'],
                            'Value': st.session_state.evaluation_metrics
                        })

                        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

            # Get the value of the buffer
            buffer.seek(0)

            # Generate download link
            b64 = base64.b64encode(buffer.read()).decode()

            # Generate filename based on model type
            if st.session_state.data_source == "Kragle":
                filename = f"{st.session_state.model_type.replace(' ', '_')}_Results.xlsx"
            else:
                ticker = st.session_state.data.index.name if st.session_state.data.index.name else "Stock"
                filename = f"{ticker}_{st.session_state.model_type.replace(' ', '_')}_Results.xlsx"

            # Create download link
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Results as Excel</a>'

            st.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error generating downloadable results: {e}")

    # Proceed button
    if st.button("Proceed to Results Visualization", key="proceed_to_visualization", use_container_width=True):
        st.session_state.current_step = 7
        st.rerun()

def results_visualization_step():
    st.header("Step 7: Results Visualization 🌟")

    if st.session_state.model is None or st.session_state.predictions is None:
        st.warning("Please complete the model training step first.")
        return

    st.subheader(f"Visualizing Results for {st.session_state.model_type}")

    # Different visualizations based on model type
    if st.session_state.model_type == "Linear Regression":
        # Interactive scatter plot with tooltips
        st.subheader("Interactive Prediction Plot")

        prediction_df = pd.DataFrame({
            'Actual': st.session_state.y_test,
            'Predicted': st.session_state.predictions,
            'Error': np.abs(st.session_state.y_test - st.session_state.predictions)
        }).reset_index(drop=True)

        # Add some of the original features for context
        for col in st.session_state.X_test.columns[:3]:  # Add a few columns for context
            prediction_df[col] = st.session_state.X_test[col].values

        st.plotly_chart(visualization.create_interactive_prediction_plot(prediction_df), use_container_width=True)

        # Feature effect plot
        st.subheader("Feature Effect Plot")

        if len(st.session_state.X_train.columns) > 0:
            feature_to_plot = st.selectbox(
                "Select feature to analyze:",
                options=st.session_state.X_train.columns
            )

            feature_effect_df = pd.DataFrame({
                feature_to_plot: st.session_state.X_test[feature_to_plot],
                'Actual': st.session_state.y_test,
                'Predicted': st.session_state.predictions
            })

            st.plotly_chart(visualization.create_feature_effect_plot(feature_effect_df, feature_to_plot), use_container_width=True)

        # Error distribution
        st.subheader("Error Distribution")

        errors = st.session_state.y_test - st.session_state.predictions

        st.plotly_chart(visualization.create_error_distribution(errors), use_container_width=True)

        # What-if analysis (interactive prediction)
        st.subheader("What-If Analysis: Interactive Prediction")

        st.write("Adjust feature values to see how they affect the prediction:")

        col1, col2 = st.columns(2)

        # Create input fields for features
        feature_values = {}

        for i, feature in enumerate(st.session_state.X_train.columns):
            # Alternate between columns
            with col1 if i % 2 == 0 else col2:
                # Use our helper function to safely get numeric values
                values = safe_numeric_conversion(st.session_state.X[feature])
                if values is None:
                    # Skip this feature if it's not numeric
                    st.info(f"{feature} is not suitable for what-if analysis (non-numeric data).")
                    continue

                min_val, max_val, mean_val = values

                # Ensure min and max values are different
                if min_val == max_val:
                    min_val = min_val - 1 if min_val != 0 else 0
                    max_val = max_val + 1
                # Use slider for numeric features with reasonable range
                if max_val - min_val < 1000:
                    feature_values[feature] = st.slider(
                        f"{feature}:",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(mean_val)
                    )
                else:
                    # Use number input for features with large ranges
                    feature_values[feature] = st.number_input(
                        f"{feature}:",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val
                    )

        # Create input dataframe
        input_df = pd.DataFrame([feature_values])

        # Make prediction
        if st.button("Predict", key="predict_button"):
            try:
                prediction = st.session_state.model.predict(input_df)[0]

                st.success(f"Predicted {st.session_state.target_column}: {prediction:.4f}")

                # Context for the prediction
                mean_target = st.session_state.y.mean()

                if prediction > mean_target:
                    st.info(f"This prediction is {(prediction - mean_target) / mean_target * 100:.2f}% above the average {st.session_state.target_column} of {mean_target:.4f}")
                else:
                    st.info(f"This prediction is {(mean_target - prediction) / mean_target * 100:.2f}% below the average {st.session_state.target_column} of {mean_target:.4f}")

            except Exception as e:
                st.error(f"Error making prediction: {e}")

    elif st.session_state.model_type == "Logistic Regression":
        # ROC curve
        if len(np.unique(st.session_state.y_test)) == 2 and hasattr(st.session_state, 'prediction_probs'):
            st.subheader("ROC Curve")

            st.plotly_chart(visualization.create_roc_curve(
                st.session_state.y_test, 
                st.session_state.prediction_probs
            ), use_container_width=True)

        # Confusion matrix
        st.subheader("Interactive Confusion Matrix")

        st.plotly_chart(visualization.create_confusion_matrix(
            st.session_state.y_test, 
            st.session_state.predictions
        ), use_container_width=True)

        # Classification report visualization
        st.subheader("Classification Report")

        report = ml_utils.get_classification_report(st.session_state.y_test, st.session_state.predictions)

        if report is not None:
            st.plotly_chart(visualization.create_classification_report_vis(report), use_container_width=True)

        # What-if analysis (interactive prediction)
        st.subheader("What-If Analysis: Interactive Prediction")

        st.write("Adjust feature values to see how they affect the classification:")

        col1, col2 = st.columns(2)

        # Create input fields for features
        feature_values = {}

        for i, feature in enumerate(st.session_state.X_train.columns):
            # Alternate between columns
            with col1 if i % 2 == 0 else col2:
                # Use our helper function to safely get numeric values
                values = safe_numeric_conversion(st.session_state.X[feature])
                if values is None:
                    # Skip this feature if it's not numeric
                    st.info(f"{feature} is not suitable for what-if analysis (non-numeric data).")
                    continue

                min_val, max_val, mean_val = values

                # Ensure min and max values are different
                if min_val == max_val:
                    min_val = min_val - 1 if min_val != 0 else 0
                    max_val = max_val + 1
                # Use slider for numeric features with reasonable range
                if max_val - min_val < 1000:
                    feature_values[feature] = st.slider(
                        f"{feature}:",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(mean_val)
                    )
                else:
                    # Use number input for features with large ranges
                    feature_values[feature] = st.number_input(
                        f"{feature}:",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val
                    )

        # Create input dataframe
        input_df = pd.DataFrame([feature_values])

        # Make prediction
        if st.button("Predict", key="predict_button"):
            try:
                prediction_class = st.session_state.model.predict(input_df)[0]

                if hasattr(st.session_state.model, 'predict_proba'):
                    prediction_prob = st.session_state.model.predict_proba(input_df)[0]

                    st.success(f"Predicted class: {prediction_class}")

                    # Display probabilities
                    prob_df = pd.DataFrame({
                        'Class': st.session_state.model.classes_,
                        'Probability': prediction_prob
                    })

                    st.plotly_chart(visualization.create_prediction_probability_chart(prob_df), use_container_width=True)
                else:
                    st.success(f"Predicted class: {prediction_class}")

            except Exception as e:
                st.error(f"Error making prediction: {e}")

    elif st.session_state.model_type == "K-Means Clustering":
        # 3D cluster visualization if we have enough features
        if st.session_state.X_test.shape[1] >= 3:
            st.subheader("3D Cluster Visualization")

            # Select features for visualization
            feature_options = st.session_state.X_test.columns.tolist()

            col1, col2, col3 = st.columns(3)

            with col1:
                x_feature = st.selectbox("X-axis feature:", options=feature_options, index=0)

            with col2:
                remaining_features = [f for f in feature_options if f != x_feature]
                y_feature = st.selectbox("Y-axis feature:", options=remaining_features, index=0)

            with col3:
                final_features = [f for f in feature_options if f != x_feature and f != y_feature]
                z_feature = st.selectbox("Z-axis feature:", options=final_features, index=0)

            # Create dataframe for visualization
            cluster_data = pd.DataFrame({
                'x': st.session_state.X_test[x_feature],
                'y': st.session_state.X_test[y_feature],
                'z': st.session_state.X_test[z_feature],
                'cluster': st.session_state.predictions
            })

            # Get cluster centers for these features
            centers = pd.DataFrame(
                st.session_state.cluster_centers,
                columns=st.session_state.X_train.columns
            )

            centers_3d = centers[[x_feature, y_feature, z_feature]].values

            st.plotly_chart(visualization.create_3d_cluster_plot(
                cluster_data, 
                centers_3d
            ), use_container_width=True)

        # Cluster radar chart
        st.subheader("Cluster Profile Radar Chart")

        # Get cluster centers and scale them
        centers = pd.DataFrame(
            st.session_state.cluster_centers,
            columns=st.session_state.X_train.columns
        )

        # Select features for radar chart (limit to 10 for readability)
        if len(centers.columns) > 10:
            radar_features = st.multiselect(
                "Select features for radar chart:",
                options=centers.columns.tolist(),
                default=centers.columns.tolist()[:5]
            )
        else:
            radar_features = centers.columns.tolist()

        if radar_features:
            st.plotly_chart(visualization.create_cluster_radar_chart(
                centers[radar_features]
            ), use_container_width=True)

        # Feature distribution by cluster
        st.subheader("Feature Distribution by Cluster")

        # Create dataframe with cluster assignments
        cluster_df = pd.DataFrame(st.session_state.X_test.copy())
        cluster_df['Cluster'] = st.session_state.predictions

        # Select feature to visualize
        feature_to_viz = st.selectbox(
            "Select feature to visualize distribution:",
            options=st.session_state.X_test.columns.tolist()
        )

        st.plotly_chart(visualization.create_cluster_feature_distribution(
            cluster_df, 
            feature_to_viz
        ), use_container_width=True)

        # Cluster assignment tool
        st.subheader("Cluster Assignment Tool")

        st.write("Input feature values to find the closest cluster:")

        col1, col2 = st.columns(2)

        # Create input fields for features
        feature_values = {}

        for i, feature in enumerate(st.session_state.X_train.columns):
            # Alternate between columns
            with col1 if i % 2 == 0 else col2:
                # Use our helper function to safely get numeric values
                values = safe_numeric_conversion(st.session_state.X[feature])
                if values is None:
                    # Skip this feature if it's not numeric
                    st.info(f"{feature} is not suitable for what-if analysis (non-numeric data).")
                    continue

                min_val, max_val, mean_val = values

                # Ensure min and max values are different
                if min_val == max_val:
                    min_val = min_val - 1 if min_val != 0 else 0
                    max_val = max_val + 1
                # Use slider for numeric features with reasonable range
                if max_val - min_val < 1000:
                    feature_values[feature] = st.slider(
                        f"{feature}:",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(mean_val)
                    )
                else:
                    # Use number input for features with large ranges
                    feature_values[feature] = st.number_input(
                        f"{feature}:",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val
                    )

        # Create input dataframe
        input_df = pd.DataFrame([feature_values])

        # Make prediction
        if st.button("Find Cluster", key="find_cluster_button"):
            try:
                cluster = st.session_state.model.predict(input_df)[0]

                st.success(f"Assigned to Cluster {cluster}")

                # Show distances to all cluster centers
                distances = []

                for i, center in enumerate(st.session_state.cluster_centers):
                    dist = np.sqrt(np.sum((input_df.values[0] - center) ** 2))
                    distances.append({"Cluster": i, "Distance": dist})

                distances_df = pd.DataFrame(distances)

                st.write("Distances to cluster centers:")
                st.plotly_chart(visualization.create_cluster_distances_chart(distances_df), use_container_width=True)

            except Exception as e:
                st.error(f"Error finding cluster: {e}")

    # Celebrating completion with a finance GIF
    st.subheader("🎉 Congratulations on Completing the ML Pipeline! 🎉")

    completion_gif = get_encoded_gif('completion')
    st.markdown(f"""
    <div style='text-align: center;'>
        <img src="data:image/gif;base64,{completion_gif}" alt="Completion" width="500">
        <p style='font-size: 18px; margin-top: 20px;'>You've successfully built and analyzed a {st.session_state.model_type} model for financial data!</p>
    </div>
    """, unsafe_allow_html=True)

    # Summary of the workflow
    st.subheader("Workflow Summary")

    st.write("Here's a summary of what you've accomplished:")

    st.write("1. **Data Loading**: Loaded data from " + ("an uploaded Kragle dataset" if st.session_state.data_source == "Kragle" else "Yahoo Finance API"))
    st.write("2. **Preprocessing**: Cleaned and prepared the data for analysis")
    st.write("3. **Feature Engineering**: Selected and transformed features for optimal model performance")
    st.write("4. **Train/Test Split**: Split the data to evaluate model performance")
    st.write(f"5. **Model Training**: Trained a {st.session_state.model_type} model")
    st.write("6. **Evaluation**: Assessed model performance using appropriate metrics")
    st.write("7. **Visualization**: Created interactive visualizations to understand the results")

    # Reset button to start over
    if st.button("Start Over with New Data", key="start_over", use_container_width=True):
        # Reset session state
        for key in list(st.session_state.keys()):
            if key != 'current_step':  # Keep the current step for now
                del st.session_state[key]

        # Go back to the welcome page
        st.session_state.current_step = 0
        st.rerun()

# Main application
def main():
    # Render sidebar
    render_sidebar()

    # Determine which step to show based on session state
    if st.session_state.current_step == 0:
        welcome_page()
    elif st.session_state.current_step == 1:
        load_data_step()
    elif st.session_state.current_step == 2:
        preprocessing_step()
    elif st.session_state.current_step == 3:
        feature_engineering_step()
    elif st.session_state.current_step == 4:
        train_test_split_step()
    elif st.session_state.current_step == 5:
        model_training_step()
    elif st.session_state.current_step == 6:
        model_evaluation_step()
    elif st.session_state.current_step == 7:
        results_visualization_step()

if __name__ == "__main__":
    main()

def convert_tz_aware_to_naive(df):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)
    return df
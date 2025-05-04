import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split as sklearn_train_test_split

def fetch_yahoo_finance_data(ticker_symbol, period="1y", interval="1d"):
    """
    Fetch stock market data from Yahoo Finance.
    
    Parameters:
    -----------
    ticker_symbol : str
        Stock ticker symbol (e.g., "AAPL", "MSFT")
    period : str, default="1y"
        Period to download data for (e.g., "1d", "1mo", "1y", "max")
    interval : str, default="1d"
        Interval between data points (e.g., "1d", "1wk", "1mo")
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame containing stock market data
    """
    try:
        # Download data from Yahoo Finance
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period, interval=interval)
        
        # Reset index to convert Date to a column
        df = df.reset_index()
        
        # Add additional columns for analysis
        if len(df) > 1:
            # Add daily returns
            df['Daily_Return'] = df['Close'].pct_change()
            
            # Add moving averages
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            
            # Add volatility (standard deviation of returns over 20 days)
            df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
            
            # Add relative strength index (RSI)
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    except Exception as e:
        raise Exception(f"Error fetching data from Yahoo Finance: {e}")

def perform_train_test_split(X, y=None, test_size=0.2, random_state=42, stratify=None):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature data
    y : pandas.Series, optional
        Target data
    test_size : float, default=0.2
        Proportion of the data to include in the test split
    random_state : int, default=42
        Random seed for reproducibility
    stratify : array-like, optional
        Data to use for stratified sampling
        
    Returns:
    --------
    splits : tuple
        For supervised learning: (X_train, X_test, y_train, y_test)
        For unsupervised learning: (X_train, X_test)
    """
    # Check if we're doing supervised or unsupervised learning
    if y is not None:
        # Supervised learning
        X_train, X_test, y_train, y_test = sklearn_train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )
        
        return X_train, X_test, y_train, y_test
    else:
        # Unsupervised learning
        X_train, X_test = sklearn_train_test_split(
            X, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test

def preprocess_data(df, handle_missing='mean', handle_outliers=None, z_threshold=3):
    """
    Preprocess data by handling missing values and outliers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data to preprocess
    handle_missing : str, default='mean'
        Method to handle missing values ('mean', 'median', 'mode', 'drop')
    handle_outliers : str, optional
        Method to handle outliers ('clip', 'remove', None)
    z_threshold : float, default=3
        Z-score threshold for outlier detection
        
    Returns:
    --------
    processed_df : pandas.DataFrame
        Preprocessed data
    """
    # Make a copy of the dataframe
    processed_df = df.copy()
    
    # Handle missing values
    if handle_missing == 'mean':
        for col in processed_df.select_dtypes(include=['int64', 'float64']).columns:
            processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
    elif handle_missing == 'median':
        for col in processed_df.select_dtypes(include=['int64', 'float64']).columns:
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())
    elif handle_missing == 'mode':
        for col in processed_df.columns:
            processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
    elif handle_missing == 'drop':
        processed_df = processed_df.dropna()
    
    # Handle outliers for numeric columns
    if handle_outliers:
        for col in processed_df.select_dtypes(include=['int64', 'float64']).columns:
            # Calculate z-scores
            z_scores = np.abs((processed_df[col] - processed_df[col].mean()) / processed_df[col].std())
            
            if handle_outliers == 'clip':
                # Clip outliers
                processed_df.loc[z_scores > z_threshold, col] = np.nan
                processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
            
            elif handle_outliers == 'remove':
                # Remove outliers
                processed_df = processed_df[z_scores <= z_threshold]
    
    return processed_df

def is_timestamp_column(data_series):
    """Check if a column contains timestamp data"""
    return pd.api.types.is_datetime64_any_dtype(data_series) or (
        hasattr(data_series, 'dtype') and 
        isinstance(data_series.dtype, pd.DatetimeTZDtype)
    )

def safe_numeric_conversion(series):
    """Convert a series to numeric, handling categorical values with label encoding"""
    try:
        # Handle Int64 type explicitly
        if hasattr(series.dtype, 'name') and series.dtype.name == 'Int64':
            series = series.astype('float64')
            
        # For datetime types
        if pd.api.types.is_datetime64_any_dtype(series):
            series_numeric = series.astype(np.int64) // 10**9
            return float(series_numeric.min()), float(series_numeric.max()), float(series_numeric.mean())
            
        # For numeric types
        if pd.api.types.is_numeric_dtype(series):
            return float(series.min()), float(series.max()), float(series.mean())
            
        # For non-numeric types
        try:
            numeric_series = pd.to_numeric(series, errors='coerce')
            if not numeric_series.isna().all():
                return float(numeric_series.min()), float(numeric_series.max()), float(numeric_series.mean())
        except:
            pass
            
        # For categorical, use label encoding
        encoded = pd.Categorical(series).codes
        return float(encoded.min()), float(encoded.max()), float(encoded.mean())
    except:
        return 0.0, 1.0, 0.5

def convert_tz_aware_to_naive(df):
    """Convert timezone-aware datetime columns to naive datetime columns"""
    df_copy = df.copy()
    for col in df_copy.columns:
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            if hasattr(df_copy[col].dt, 'tz') and df_copy[col].dt.tz is not None:
                df_copy[col] = df_copy[col].dt.tz_localize(None)
    return df_copy

def handle_timestamp_columns(df):
    """Handle timestamp columns by converting them to numeric features"""
    timestamp_cols = []
    df_processed = df.copy()
    
    for col in df.columns:
        if is_timestamp_column(df[col]):
            timestamp_cols.append(col)
            df_processed[f"{col}_year"] = df_processed[col].dt.year
            df_processed[f"{col}_month"] = df_processed[col].dt.month
            df_processed[f"{col}_day"] = df_processed[col].dt.day
            df_processed[f"{col}_dayofweek"] = df_processed[col].dt.dayofweek
            if hasattr(df_processed[col].dt, 'hour'):
                df_processed[f"{col}_hour"] = df_processed[col].dt.hour
            df_processed = df_processed.drop(columns=[col])
    
    return df_processed, timestamp_cols

def encode_categorical_features(df, encoding_method='one-hot', drop_first=False):
    """
    Encode categorical features in the data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data containing categorical features
    encoding_method : str, default='one-hot'
        Method to encode categorical features ('one-hot', 'label')
    drop_first : bool, default=False
        Whether to drop the first category in one-hot encoding
        
    Returns:
    --------
    encoded_df : pandas.DataFrame
        Data with encoded categorical features
    """
    # Make a copy of the dataframe
    encoded_df = df.copy()
    
    # Get categorical columns
    categorical_cols = encoded_df.select_dtypes(include=['object', 'category']).columns
    
    # Also include numeric columns with few unique values
    for col in encoded_df.select_dtypes(include=['int64']).columns:
        if encoded_df[col].nunique() < 10:
            categorical_cols = pd.Index(list(categorical_cols) + [col])
    
    if encoding_method == 'one-hot':
        # One-hot encode categorical variables
        encoded_df = pd.get_dummies(encoded_df, columns=categorical_cols, drop_first=drop_first)
    
    elif encoding_method == 'label':
        # Label encode categorical variables
        for col in categorical_cols:
            encoded_df[col] = encoded_df[col].astype('category').cat.codes
    
    return encoded_df

def normalize_features(df, method='z-score'):
    """
    Normalize numeric features in the data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data containing numeric features
    method : str, default='z-score'
        Method to normalize features ('z-score', 'min-max')
        
    Returns:
    --------
    normalized_df : pandas.DataFrame
        Data with normalized numeric features
    """
    # Make a copy of the dataframe
    normalized_df = df.copy()
    
    # Get numeric columns
    numeric_cols = normalized_df.select_dtypes(include=['int64', 'float64']).columns
    
    if method == 'z-score':
        # Z-score normalization
        for col in numeric_cols:
            mean = normalized_df[col].mean()
            std = normalized_df[col].std()
            
            if std > 0:  # Avoid division by zero
                normalized_df[col] = (normalized_df[col] - mean) / std
    
    elif method == 'min-max':
        # Min-max normalization
        for col in numeric_cols:
            min_val = normalized_df[col].min()
            max_val = normalized_df[col].max()
            
            if max_val > min_val:  # Avoid division by zero
                normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
    
    return normalized_df

def create_time_features(df, date_column):
    """
    Create time-based features from a date column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data containing a date column
    date_column : str
        Name of the date column
        
    Returns:
    --------
    df_with_time : pandas.DataFrame
        Data with additional time-based features
    """
    # Make a copy of the dataframe
    df_with_time = df.copy()
    
    # Convert date column to datetime if it's not already
    if df_with_time[date_column].dtype != 'datetime64[ns]':
        df_with_time[date_column] = pd.to_datetime(df_with_time[date_column])
    
    # Extract time features
    df_with_time['year'] = df_with_time[date_column].dt.year
    df_with_time['month'] = df_with_time[date_column].dt.month
    df_with_time['day'] = df_with_time[date_column].dt.day
    df_with_time['day_of_week'] = df_with_time[date_column].dt.dayofweek
    df_with_time['day_of_year'] = df_with_time[date_column].dt.dayofyear
    df_with_time['quarter'] = df_with_time[date_column].dt.quarter
    df_with_time['is_month_start'] = df_with_time[date_column].dt.is_month_start.astype(int)
    df_with_time['is_month_end'] = df_with_time[date_column].dt.is_month_end.astype(int)
    
    return df_with_time

def create_lag_features(df, columns, lag_periods=[1, 2, 3]):
    """
    Create lag features for time series data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Time series data
    columns : list
        Columns to create lag features for
    lag_periods : list, default=[1, 2, 3]
        List of lag periods to create
        
    Returns:
    --------
    df_with_lags : pandas.DataFrame
        Data with additional lag features
    """
    # Make a copy of the dataframe
    df_with_lags = df.copy()
    
    # Create lag features
    for col in columns:
        for lag in lag_periods:
            df_with_lags[f'{col}_lag_{lag}'] = df_with_lags[col].shift(lag)
    
    # Drop rows with NaN values created by the lag
    df_with_lags = df_with_lags.dropna()
    
    return df_with_lags

def create_polynomial_features(df, columns, degree=2):
    """
    Create polynomial features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data containing numeric features
    columns : list
        Columns to create polynomial features for
    degree : int, default=2
        Degree of the polynomial features
        
    Returns:
    --------
    df_with_poly : pandas.DataFrame
        Data with additional polynomial features
    """
    # Make a copy of the dataframe
    df_with_poly = df.copy()
    
    # Create polynomial features
    for col in columns:
        for d in range(2, degree + 1):
            df_with_poly[f'{col}^{d}'] = df_with_poly[col] ** d
    
    return df_with_poly

def create_interaction_features(df, columns):
    """
    Create interaction features between columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data containing numeric features
    columns : list
        Columns to create interaction features for
        
    Returns:
    --------
    df_with_interact : pandas.DataFrame
        Data with additional interaction features
    """
    # Make a copy of the dataframe
    df_with_interact = df.copy()
    
    # Create interaction features
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]
            df_with_interact[f'{col1}_{col2}_interaction'] = df_with_interact[col1] * df_with_interact[col2]
    
    return df_with_interact

def ensure_numeric_features(X):
    """
    Ensure all features in X are numeric, encoding categorical features if needed.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature data
        
    Returns:
    --------
    X_numeric : pandas.DataFrame
        DataFrame with all numeric features
    """
    X_numeric = X.copy()
    
    for col in X_numeric.columns:
        # Handle Int64 type explicitly
        if hasattr(X_numeric[col].dtype, 'name') and X_numeric[col].dtype.name == 'Int64':
            X_numeric[col] = X_numeric[col].astype('float64')
            continue
            
        # Handle datetime
        if pd.api.types.is_datetime64_any_dtype(X_numeric[col]):
            X_numeric[col] = X_numeric[col].astype(np.int64) // 10**9
            continue
            
        # Handle non-numeric types
        if not pd.api.types.is_numeric_dtype(X_numeric[col]):
            # Try converting to numeric first
            try:
                X_numeric[col] = pd.to_numeric(X_numeric[col], errors='raise')
                continue
            except:
                pass
                
            # Check if binary categorical
            if X_numeric[col].astype(str).str.lower().isin(['yes', 'no', 'true', 'false', 'y', 'n']).all():
                bool_map = {'yes': 1, 'no': 0, 'true': 1, 'false': 0, 'y': 1, 'n': 0}
                X_numeric[col] = X_numeric[col].astype(str).str.lower().map(bool_map).astype('float64')
            else:
                # For other categorical features, use one-hot encoding
                dummies = pd.get_dummies(X_numeric[col], prefix=col, drop_first=True)
                X_numeric = pd.concat([X_numeric.drop(col, axis=1), dummies], axis=1)
    
    # Final conversion to float64 for all columns
    for col in X_numeric.columns:
        X_numeric[col] = X_numeric[col].astype('float64')
    
    return X_numeric
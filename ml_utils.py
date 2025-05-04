import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import silhouette_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

def train_linear_regression(X_train, y_train, fit_intercept=True, normalize=False):
    # Convert categorical target to numeric if needed
    if y_train.dtype == 'object' or y_train.dtype.name == 'category':
        y_train = pd.Categorical(y_train).codes
    """
    Train a Linear Regression model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature data
    y_train : pandas.Series
        Training target data
    fit_intercept : bool, default=True
        Whether to fit an intercept
    normalize : bool, default=False
        Whether to normalize the data (deprecated, only kept for backward compatibility)
        
    Returns:
    --------
    model : LinearRegression
        Trained model
    feature_importance : array
        Feature importance values (coefficients)
    """
    # Normalize data if requested (manually since the parameter was deprecated)
    if normalize:
        # Create a copy of the data to avoid modifying the original
        X_scaled = X_train.copy()
        for column in X_scaled.columns:
            if X_scaled[column].std() > 0:  # Avoid division by zero
                X_scaled[column] = (X_scaled[column] - X_scaled[column].mean()) / X_scaled[column].std()
    else:
        X_scaled = X_train
    
    # Train the model (without the deprecated 'normalize' parameter)
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_scaled, y_train)
    
    # Get feature importance (coefficients)
    feature_importance = model.coef_
    
    return model, feature_importance

def train_logistic_regression(X, y):
    """
    Train a logistic regression model.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature data
    y : pandas.Series
        Target variable
        
    Returns:
    --------
    model : sklearn.linear_model.LogisticRegression
        Trained logistic regression model
    feature_importance : pandas.Series
        Feature importance
    """
    # Check if target variable is categorical and needs encoding
    if not pd.api.types.is_numeric_dtype(y):
        # Handle binary categorical targets (yes/no, true/false, etc.)
        if y.astype(str).str.lower().isin(['yes', 'no', 'true', 'false', 'y', 'n', '0', '1']).all():
            bool_map = {'yes': 1, 'no': 0, 'true': 1, 'false': 0, 'y': 1, 'n': 0, '1': 1, '0': 0}
            y = y.astype(str).str.lower().map(bool_map)
        else:
            # For other categorical targets, use LabelEncoder
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
    
    # Create and train logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    
    # Get feature importance
    feature_importance = pd.Series(
        np.abs(model.coef_[0]),
        index=X.columns
    ).sort_values(ascending=False)
    
    return model, feature_importance

def train_kmeans_clustering(X_train, n_clusters=3, init='k-means++', max_iter=300, n_init=10):
    """
    Train a K-means clustering model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training feature data
    n_clusters : int, default=3
        Number of clusters
    init : {'k-means++', 'random'}, default='k-means++'
        Method for initialization
    max_iter : int, default=300
        Maximum number of iterations
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different centroid seeds
        
    Returns:
    --------
    model : KMeans
        Trained model
    cluster_centers : array
        Cluster centers
    """
    # Train the model
    model = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, n_init=n_init, random_state=42)
    model.fit(X_train)
    
    # Get cluster centers
    cluster_centers = model.cluster_centers_
    
    return model, cluster_centers

def evaluate_regression(y_true, y_pred):
    """
    Evaluate regression model performance.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
        
    Returns:
    --------
    metrics : tuple
        (MAE, MSE, RMSE, R²)
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return mae, mse, rmse, r2

def evaluate_classification(y_true, y_pred):
    """
    Evaluate classification model performance.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
        
    Returns:
    --------
    metrics : tuple
        (Accuracy, Precision, Recall, F1 Score)
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # If binary classification
    if len(np.unique(y_true)) == 2:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1

def evaluate_clustering(X, cluster_labels):
    """
    Evaluate clustering model performance.
    
    Parameters:
    -----------
    X : array-like
        Feature data
    cluster_labels : array-like
        Cluster assignments
        
    Returns:
    --------
    metrics : tuple
        (Silhouette Score, Inertia)
    """
    # Calculate silhouette score (only if we have more than one cluster and more than one sample)
    n_clusters = len(np.unique(cluster_labels))
    
    if n_clusters > 1 and len(X) > n_clusters:
        silhouette = silhouette_score(X, cluster_labels)
    else:
        silhouette = 0  # Not applicable
    
    # Calculate inertia
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    inertia = kmeans.inertia_
    
    return silhouette, inertia

def get_classification_report(y_true, y_pred):
    """
    Get classification report as a dataframe.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
        
    Returns:
    --------
    report_df : pandas.DataFrame
        Classification report as a dataframe
    """
    try:
        # Get the report as a string
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Convert to dataframe
        report_df = pd.DataFrame(report).transpose()
        
        return report_df
    except Exception as e:
        print(f"Error generating classification report: {e}")
        return None

def standardize_features(X):
    """
    Standardize features to have mean=0 and variance=1.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature data
        
    Returns:
    --------
    X_scaled : pandas.DataFrame
        Standardized feature data
    scaler : StandardScaler
        Fitted scaler
    """
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit and transform the data
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    return X_scaled, scaler

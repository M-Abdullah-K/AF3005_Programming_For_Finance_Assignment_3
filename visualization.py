import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

def create_line_chart(data, columns):
    """
    Create a line chart for selected columns.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to visualize
    columns : list
        Columns to include in the line chart
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Line chart figure
    """
    fig = go.Figure()
    
    for column in columns:
        fig.add_trace(go.Scatter(
            x=data.index if data.index.name is not None else data.index,
            y=data[column],
            mode='lines',
            name=column
        ))
    
    fig.update_layout(
        title="Line Chart",
        xaxis_title="Index",
        yaxis_title="Value",
        legend_title="Variables",
        template="plotly_white"
    )
    
    return fig

def create_bar_chart(data, columns):
    """
    Create a bar chart for selected columns.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to visualize
    columns : list
        Columns to include in the bar chart
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Bar chart figure
    """
    # For multiple columns, create a grouped bar chart
    if len(columns) > 1:
        # Take a sample of data if it's too large
        if len(data) > 50:
            sample_data = data.sample(50)
        else:
            sample_data = data
            
        fig = go.Figure()
        
        for column in columns:
            fig.add_trace(go.Bar(
                x=sample_data.index if sample_data.index.name is not None else sample_data.index,
                y=sample_data[column],
                name=column
            ))
            
        fig.update_layout(
            title="Grouped Bar Chart",
            xaxis_title="Index",
            yaxis_title="Value",
            legend_title="Variables",
            barmode='group',
            template="plotly_white"
        )
    
    # For a single column, create a simple bar chart
    else:
        column = columns[0]
        
        # Get value counts if categorical, else use histogram
        if data[column].dtype == 'object' or data[column].nunique() < 15:
            value_counts = data[column].value_counts().sort_index()
            
            fig = go.Figure(go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                text=value_counts.values,
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"Bar Chart - {column}",
                xaxis_title=column,
                yaxis_title="Count",
                template="plotly_white"
            )
        
        else:
            fig = px.histogram(
                data,
                x=column,
                nbins=20,
                title=f"Histogram - {column}"
            )
            
            fig.update_layout(
                xaxis_title=column,
                yaxis_title="Count",
                template="plotly_white"
            )
    
    return fig

def create_histogram(data, columns):
    """
    Create histograms for selected columns.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to visualize
    columns : list
        Columns to create histograms for
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Histogram figure
    """
    # Create subplots for multiple columns
    if len(columns) > 1:
        fig = make_subplots(
            rows=len(columns), 
            cols=1,
            subplot_titles=[f"Histogram - {column}" for column in columns]
        )
        
        for i, column in enumerate(columns, 1):
            fig.add_trace(
                go.Histogram(
                    x=data[column],
                    name=column,
                    nbinsx=20
                ),
                row=i,
                col=1
            )
            
        fig.update_layout(
            height=300 * len(columns),
            showlegend=False,
            template="plotly_white"
        )
    
    # Single histogram for one column
    else:
        column = columns[0]
        
        fig = px.histogram(
            data,
            x=column,
            nbins=20,
            title=f"Histogram - {column}"
        )
        
        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Count",
            template="plotly_white"
        )
    
    return fig

def create_box_plot(data, columns):
    """
    Create box plots for selected columns.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to visualize
    columns : list
        Columns to create box plots for
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Box plot figure
    """
    # Create box plot with all columns
    fig = go.Figure()
    
    for column in columns:
        fig.add_trace(go.Box(
            y=data[column],
            name=column,
            boxpoints='outliers'
        ))
    
    fig.update_layout(
        title="Box Plot",
        yaxis_title="Value",
        template="plotly_white"
    )
    
    return fig

def create_scatter_plot(data, x_column, y_column):
    """
    Create a scatter plot of two columns.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to visualize
    x_column : str
        Column for x-axis
    y_column : str
        Column for y-axis
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Scatter plot figure
    """
    fig = px.scatter(
        data,
        x=x_column,
        y=y_column,
        title=f"Scatter Plot: {x_column} vs {y_column}",
        opacity=0.7
    )
    
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        template="plotly_white"
    )
    
    # Add a trend line
    if data[x_column].dtype in ['int64', 'float64'] and data[y_column].dtype in ['int64', 'float64']:
        fig.add_trace(go.Scatter(
            x=data[x_column],
            y=data[y_column],
            mode='lines',
            name='Trend',
            line=dict(color='red', dash='dash'),
            opacity=0.5
        ))
    
    return fig

def create_correlation_heatmap(data):
    """
    Create a correlation heatmap.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data to create correlation heatmap from
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Correlation heatmap figure
    """
    corr = data.corr()
    
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap"
    )
    
    fig.update_layout(
        template="plotly_white"
    )
    
    return fig

def create_feature_distribution(data, feature):
    """
    Create distribution plot for a feature.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data containing the feature
    feature : str
        Feature to visualize
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Distribution plot
    """
    # Check if the feature is categorical or has few unique values
    if data[feature].dtype == 'object' or data[feature].nunique() < 10:
        # Create a bar chart for categorical features
        value_counts = data[feature].value_counts().sort_index()
        
        fig = px.bar(
            x=value_counts.index, 
            y=value_counts.values,
            labels={'x': feature, 'y': 'Count'},
            title=f"Distribution of {feature}"
        )
        
        fig.update_traces(
            text=value_counts.values,
            textposition='outside'
        )
    else:
        # Create a histogram for continuous features
        fig = px.histogram(
            data,
            x=feature,
            nbins=30,
            title=f"Distribution of {feature}"
        )
        
        # Add a KDE curve
        kde_x = np.linspace(data[feature].min(), data[feature].max(), 1000)
        kde_y = data[feature].plot.kde().get_lines()[0].get_data()[1]
        
        # Scale the KDE to match the histogram height
        hist_max = np.histogram(data[feature], bins=30)[0].max()
        kde_max = kde_y.max()
        kde_y = kde_y * (hist_max / kde_max)
        
        fig.add_trace(go.Scatter(
            x=kde_x,
            y=kde_y,
            mode='lines',
            name='Density',
            line=dict(color='red', width=2)
        ))
    
    fig.update_layout(
        xaxis_title=feature,
        yaxis_title="Count",
        template="plotly_white"
    )
    
    return fig

def create_train_test_split_chart(train_size, test_size):
    """
    Create a pie chart showing train/test split.
    
    Parameters:
    -----------
    train_size : int
        Size of training set
    test_size : int
        Size of test set
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Pie chart figure
    """
    labels = ['Training Set', 'Testing Set']
    values = [train_size, test_size]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=['#2E86C1', '#E74C3C']
    )])
    
    fig.update_layout(
        title_text="Train/Test Split",
        template="plotly_white"
    )
    
    # Add percentage annotations
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+value'
    )
    
    return fig

def create_class_distribution_chart(class_dist):
    """
    Create a bar chart comparing class distributions in train/test sets.
    
    Parameters:
    -----------
    class_dist : pandas.DataFrame
        DataFrame with class distributions for train and test sets
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Bar chart figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=class_dist.index,
        y=class_dist['Training Set'],
        name='Training Set',
        marker_color='#2E86C1'
    ))
    
    fig.add_trace(go.Bar(
        x=class_dist.index,
        y=class_dist['Testing Set'],
        name='Testing Set',
        marker_color='#E74C3C'
    ))
    
    fig.update_layout(
        title_text="Class Distribution in Train/Test Sets",
        xaxis_title="Class",
        yaxis_title="Percentage (%)",
        barmode='group',
        template="plotly_white"
    )
    
    return fig

def create_feature_importance_chart(features, importance):
    """
    Create a bar chart of feature importances.
    
    Parameters:
    -----------
    features : array-like
        Feature names
    importance : array-like
        Feature importance values
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Feature importance bar chart
    """
    # Create a DataFrame for sorting
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': np.abs(importance)  # Use absolute values for importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Create the figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=importance_df['Feature'],
        y=importance_df['Importance'],
        marker_color='#3498DB',
        text=importance_df['Importance'].round(3),
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Feature",
        yaxis_title="Importance",
        template="plotly_white"
    )
    
    return fig

def create_actual_vs_predicted_chart(df, model_type="regression"):
    """
    Create a scatter plot of actual vs predicted values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'Actual' and 'Predicted' columns
    model_type : str, default="regression"
        Type of model ('regression' or 'classification')
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Scatter plot
    """
    if model_type == "regression":
        fig = px.scatter(
            df,
            x="Actual",
            y="Predicted",
            title="Actual vs. Predicted Values",
            opacity=0.7
        )
        
        # Add a diagonal line (perfect prediction)
        x_range = [df["Actual"].min(), df["Actual"].max()]
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=x_range,
            mode="lines",
            name="Perfect Prediction",
            line=dict(color="red", dash="dash")
        ))
        
        fig.update_layout(
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            template="plotly_white"
        )
    
    else:  # Classification
        fig = px.scatter(
            df,
            x="Actual",
            y="Predicted",
            title="Actual vs. Predicted Classes",
            opacity=0.7
        )
        
        fig.update_layout(
            xaxis_title="Actual Class",
            yaxis_title="Predicted Class",
            template="plotly_white"
        )
    
    return fig

def create_residual_plot(df):
    """
    Create a residual plot (predicted vs residuals).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'Predicted' and 'Residuals' columns
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Residual plot
    """
    fig = px.scatter(
        df,
        x="Predicted",
        y="Residuals",
        title="Residual Plot",
        opacity=0.7
    )
    
    # Add a horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=df["Predicted"].min(),
        y0=0,
        x1=df["Predicted"].max(),
        y1=0,
        line=dict(color="red", width=2, dash="dash")
    )
    
    fig.update_layout(
        xaxis_title="Predicted Values",
        yaxis_title="Residuals",
        template="plotly_white"
    )
    
    return fig

def create_residual_distribution(residuals):
    """
    Create a histogram of residuals.
    
    Parameters:
    -----------
    residuals : array-like
        Residual values
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Histogram
    """
    fig = px.histogram(
        x=residuals,
        nbins=30,
        title="Residual Distribution"
    )
    
    fig.update_layout(
        xaxis_title="Residuals",
        yaxis_title="Count",
        template="plotly_white"
    )
    
    # Add a KDE curve
    kde_x = np.linspace(min(residuals), max(residuals), 1000)
    
    # Calculate KDE
    try:
        kde_y = pd.Series(residuals).plot.kde().get_lines()[0].get_data()[1]
        
        # Scale the KDE to match the histogram height
        hist_max = np.histogram(residuals, bins=30)[0].max()
        kde_max = kde_y.max()
        kde_y = kde_y * (hist_max / kde_max)
        
        fig.add_trace(go.Scatter(
            x=kde_x,
            y=kde_y,
            mode='lines',
            name='Density',
            line=dict(color='red', width=2)
        ))
    except Exception as e:
        # Skip KDE if it fails
        pass
    
    return fig

def create_confusion_matrix(y_true, y_pred):
    """
    Create a confusion matrix visualization.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Confusion matrix heatmap
    """
    # Calculate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize the confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Get unique classes
    classes = np.unique(np.concatenate((y_true, y_pred)))
    
    # Create heatmap
    fig = px.imshow(
        cm,
        x=classes,
        y=classes,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        title="Confusion Matrix"
    )
    
    # Add text annotations
    annotations = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            annotations.append(
                dict(
                    x=classes[j],
                    y=classes[i],
                    text=f"{cm[i, j]}<br>({cm_norm[i, j]:.1%})",
                    showarrow=False,
                    font=dict(color="white" if cm[i, j] > cm.max() / 2 else "black")
                )
            )
    
    fig.update_layout(annotations=annotations)
    
    return fig

def create_roc_curve(y_true, y_scores):
    """
    Create a ROC curve.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_scores : array-like
        Target scores (probabilities)
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        ROC curve
    """
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig = go.Figure()
    
    # Add ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='#2E86C1', width=2)
    ))
    
    # Add diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='Receiver Operating Characteristic (ROC) Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        template="plotly_white",
        legend=dict(x=0.1, y=0, orientation='h')
    )
    
    return fig

def create_cluster_plot(df, centers):
    """
    Create a cluster visualization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with features and 'Cluster' column
    centers : array
        Cluster center coordinates
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Cluster visualization
    """
    # If data has more than 2 dimensions, use PCA to reduce to 2D
    if df.shape[1] - 1 > 2:  # Subtract 1 for the 'Cluster' column
        pca = PCA(n_components=2)
        features = df.drop('Cluster', axis=1)
        feature_names = features.columns
        
        # Apply PCA
        pca_result = pca.fit_transform(features)
        
        # Create DataFrame with PCA results
        pca_df = pd.DataFrame(
            data=pca_result,
            columns=['Principal Component 1', 'Principal Component 2']
        )
        pca_df['Cluster'] = df['Cluster']
        
        # Transform centers
        centers_2d = pca.transform(centers)
        
        # Create figure
        fig = px.scatter(
            pca_df,
            x='Principal Component 1',
            y='Principal Component 2',
            color='Cluster',
            title='Cluster Visualization (PCA)',
            color_continuous_scale=px.colors.diverging.RdYlBu,
            opacity=0.7
        )
        
        # Add cluster centers
        for i, center in enumerate(centers_2d):
            fig.add_trace(go.Scatter(
                x=[center[0]],
                y=[center[1]],
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=15,
                    color='black',
                    line=dict(width=2)
                ),
                name=f'Cluster {i} Center'
            ))
        
        # Add explained variance as annotation
        explained_variance = pca.explained_variance_ratio_
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            text=f"Explained Variance:<br>PC1: {explained_variance[0]:.2%}<br>PC2: {explained_variance[1]:.2%}",
            showarrow=False,
            xref="paper",
            yref="paper",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1
        )
    
    # For 2D data, plot directly
    else:
        features = df.drop('Cluster', axis=1).columns.tolist()
        
        # Create figure
        fig = px.scatter(
            df,
            x=features[0],
            y=features[1],
            color='Cluster',
            title='Cluster Visualization',
            color_continuous_scale=px.colors.diverging.RdYlBu,
            opacity=0.7
        )
        
        # Add cluster centers
        for i, center in enumerate(centers):
            fig.add_trace(go.Scatter(
                x=[center[0]],
                y=[center[1]],
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=15,
                    color='black',
                    line=dict(width=2)
                ),
                name=f'Cluster {i} Center'
            ))
    
    fig.update_layout(
        template="plotly_white"
    )
    
    return fig

def create_cluster_distribution_chart(cluster_counts):
    """
    Create a bar chart of cluster distributions.
    
    Parameters:
    -----------
    cluster_counts : pandas.Series
        Counts for each cluster
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Bar chart
    """
    fig = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        labels={'x': 'Cluster', 'y': 'Count'},
        title="Cluster Size Distribution",
        color=cluster_counts.index,
        color_continuous_scale=px.colors.diverging.RdYlBu
    )
    
    fig.update_traces(
        text=cluster_counts.values,
        textposition='outside'
    )
    
    fig.update_layout(
        xaxis_title="Cluster",
        yaxis_title="Count",
        template="plotly_white",
        coloraxis_showscale=False
    )
    
    return fig

def create_3d_cluster_plot(df, centers):
    """
    Create a 3D scatter plot of clusters.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with x, y, z coordinates and cluster assignments
    centers : array
        Cluster center coordinates
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        3D scatter plot
    """
    fig = px.scatter_3d(
        df,
        x='x',
        y='y',
        z='z',
        color='cluster',
        title="3D Cluster Visualization",
        opacity=0.7,
        color_continuous_scale=px.colors.diverging.RdYlBu
    )
    
    # Add cluster centers
    for i, center in enumerate(centers):
        fig.add_trace(go.Scatter3d(
            x=[center[0]],
            y=[center[1]],
            z=[center[2]],
            mode='markers',
            marker=dict(
                symbol='diamond',
                size=8,
                color='black'
            ),
            name=f'Cluster {i} Center'
        ))
    
    fig.update_layout(
        template="plotly_white"
    )
    
    return fig

def create_cluster_radar_chart(centers):
    """
    Create a radar chart showing cluster profiles.
    
    Parameters:
    -----------
    centers : pandas.DataFrame
        DataFrame with cluster centers
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Radar chart
    """
    # Normalize the centers for better visualization
    normalized_centers = centers.copy()
    
    for column in centers.columns:
        min_val = centers[column].min()
        max_val = centers[column].max()
        
        if max_val > min_val:
            normalized_centers[column] = (centers[column] - min_val) / (max_val - min_val)
    
    # Create radar chart
    fig = go.Figure()
    
    # Add each cluster as a separate trace
    for i, center in enumerate(normalized_centers.values):
        fig.add_trace(go.Scatterpolar(
            r=center,
            theta=centers.columns,
            fill='toself',
            name=f'Cluster {i}'
        ))
    
    fig.update_layout(
        title="Cluster Profiles",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        template="plotly_white"
    )
    
    return fig

def create_cluster_feature_distribution(df, feature):
    """
    Create a box plot showing feature distribution by cluster.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with feature and cluster assignments
    feature : str
        Feature to visualize
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Box plot
    """
    fig = px.box(
        df,
        x='Cluster',
        y=feature,
        color='Cluster',
        title=f"{feature} Distribution by Cluster",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    fig.update_layout(
        xaxis_title="Cluster",
        yaxis_title=feature,
        template="plotly_white",
        showlegend=False
    )
    
    return fig

def create_cluster_distances_chart(df):
    """
    Create a bar chart showing distances to cluster centers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'Cluster' and 'Distance' columns
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Bar chart
    """
    fig = px.bar(
        df,
        x='Cluster',
        y='Distance',
        color='Cluster',
        title="Distances to Cluster Centers",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    fig.update_traces(
        text=df['Distance'].round(2),
        textposition='outside'
    )
    
    fig.update_layout(
        xaxis_title="Cluster",
        yaxis_title="Distance",
        template="plotly_white",
        showlegend=False
    )
    
    return fig

def create_interactive_prediction_plot(df):
    """
    Create an interactive scatter plot with tooltips for predictions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'Actual', 'Predicted', and other columns
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Interactive scatter plot
    """
    # Create a custom tooltip with additional information
    hover_data = list(df.columns)
    hover_data.remove('Actual')
    hover_data.remove('Predicted')
    
    fig = px.scatter(
        df,
        x='Actual',
        y='Predicted',
        color='Error',
        hover_data=hover_data,
        title="Interactive Prediction Results",
        color_continuous_scale='RdYlBu_r',
        opacity=0.7
    )
    
    # Add diagonal line (perfect prediction)
    x_range = [df['Actual'].min(), df['Actual'].max()]
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=x_range,
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='black', dash='dash')
    ))
    
    fig.update_layout(
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        template="plotly_white"
    )
    
    return fig

def is_timestamp_column(data_series):
    """Check if a column contains timestamp data"""
    return pd.api.types.is_datetime64_any_dtype(data_series) or (
        hasattr(data_series, 'dtype') and 
        isinstance(data_series.dtype, pd.DatetimeTZDtype)
    )

def is_timestamp_column(data_series):
    """Check if a column contains timestamp data"""
    return pd.api.types.is_datetime64_any_dtype(data_series) or (
        hasattr(data_series, 'dtype') and 
        isinstance(data_series.dtype, pd.DatetimeTZDtype)
    )

def create_feature_effect_plot(df, feature):
    """
    Create a scatter plot showing the effect of a feature on predictions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with feature, 'Actual', and 'Predicted' columns
    feature : str
        Feature to analyze
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Scatter plot
    """
    fig = go.Figure()
    
    # Check if feature is a timestamp
    if is_timestamp_column(df[feature]):
        # For timestamp features, convert to numerical representation (timestamp)
        x_values = df[feature].astype(np.int64) // 10**9  # Convert to unix timestamp
        title = f"Effect of {feature} (Time Series) on Target"
    else:
        x_values = df[feature]
        title = f"Effect of {feature} on Target"
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=x_values,
        y=df['Actual'],
        mode='markers',
        name='Actual',
        marker=dict(color='blue', size=8)
    ))
    
    # Add predicted values
    fig.add_trace(go.Scatter(
        x=x_values,
        y=df['Predicted'],
        mode='markers',
        name='Predicted',
        marker=dict(color='red', size=8)
    ))
    
    # Add trend lines
    x_sorted = np.sort(x_values)
    
    # Linear regression for actual values
    actual_coeffs = np.polyfit(x_values, df['Actual'], 1)
    actual_trend = np.polyval(actual_coeffs, x_sorted)
    
    fig.add_trace(go.Scatter(
        x=x_sorted,
        y=actual_trend,
        mode='lines',
        name='Actual Trend',
        line=dict(color='blue', dash='dash')
    ))
    
    # Linear regression for predicted values
    pred_coeffs = np.polyfit(x_values, df['Predicted'], 1)
    pred_trend = np.polyval(pred_coeffs, x_sorted)
    
    fig.add_trace(go.Scatter(
        x=x_sorted,
        y=pred_trend,
        mode='lines',
        name='Predicted Trend',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=feature,
        yaxis_title="Value",
        template="plotly_white"
    )
    
    return fig

def create_error_distribution(errors):
    """
    Create a histogram of prediction errors.
    
    Parameters:
    -----------
    errors : array-like
        Prediction errors
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Histogram
    """
    fig = px.histogram(
        x=errors,
        nbins=30,
        title="Error Distribution",
        color_discrete_sequence=['#3498DB']
    )
    
    # Add mean and median lines
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    
    fig.add_vline(
        x=mean_error,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_error:.2f}",
        annotation_position="top"
    )
    
    fig.add_vline(
        x=median_error,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Median: {median_error:.2f}",
        annotation_position="bottom"
    )
    
    fig.update_layout(
        xaxis_title="Error",
        yaxis_title="Count",
        template="plotly_white"
    )
    
    return fig

def create_classification_report_vis(report):
    """
    Create a heatmap visualization of the classification report.
    
    Parameters:
    -----------
    report : pandas.DataFrame
        Classification report as a DataFrame
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Heatmap
    """
    # Filter out the 'support' column and the 'accuracy' row
    metrics_df = report.drop('support', axis=1)
    metrics_df = metrics_df.drop('accuracy', errors='ignore')
    
    fig = px.imshow(
        metrics_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="Classification Report"
    )
    
    fig.update_layout(
        xaxis_title="Metric",
        yaxis_title="Class",
        template="plotly_white"
    )
    
    return fig

def create_prediction_probability_chart(df):
    """
    Create a bar chart of prediction probabilities.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'Class' and 'Probability' columns
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Bar chart
    """
    fig = px.bar(
        df,
        x='Class',
        y='Probability',
        title="Prediction Probabilities",
        color='Probability',
        color_continuous_scale="Blues"
    )
    
    fig.update_traces(
        text=df['Probability'].round(3),
        textposition='outside'
    )
    
    fig.update_layout(
        xaxis_title="Class",
        yaxis_title="Probability",
        template="plotly_white",
        coloraxis_showscale=False
    )
    
    return fig

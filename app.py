# Full code with all fixes applied

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta
import time
import gc
import io
import base64
from tqdm import tqdm
import os
import re
import glob
from pathlib import Path

# Set page configuration
st.set_page_config(
    layout="wide", 
    page_title="Multi-Step Time Series Forecasting with LSTM",
    page_icon="📈"
)

# Check for GPU availability (without showing any message)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
MODEL_CONFIG = {
    "default_lstm_layers": 2,
    "default_lstm_neurons": 50,
    "default_linear_layers": 2,
    "default_linear_neurons": 50,
    "max_epochs": 200,
    "default_learning_rate": 0.001
}

# Define the LSTM model
class LSTMForecasting(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, linear_hidden_size, lstm_num_layers, linear_num_layers, output_size):
        super(LSTMForecasting, self).__init__()
        self.linear_hidden_size = linear_hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.linear_num_layers = linear_num_layers
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, lstm_num_layers, batch_first=True)
        self.linear_layers = nn.ModuleList()
        self.linear_num_layers -= 1
        self.linear_layers.append(nn.Linear(self.lstm_hidden_size, self.linear_hidden_size))

        for _ in range(linear_num_layers):
            self.linear_layers.append(nn.Linear(self.linear_hidden_size, int(self.linear_hidden_size / 1.5)))
            self.linear_hidden_size = int(self.linear_hidden_size / 1.5)

        self.fc = nn.Linear(self.linear_hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        for linear_layer in self.linear_layers:
            out = linear_layer(out)

        out = self.fc(out[:, -1, :])
        return out

# Helper functions
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

def split_sequences_optimized(sequences, n_steps_in, n_steps_out):
    # Use numpy array operations instead of loops
    total_length = len(sequences) - n_steps_in - n_steps_out + 1
    if total_length <= 0:
        return torch.tensor([]), torch.tensor([])
    
    indices = np.arange(total_length)
    X_indices = indices[:, None] + np.arange(n_steps_in)
    y_indices = indices[:, None] + np.arange(n_steps_in, n_steps_in + n_steps_out)
    
    X = sequences[X_indices, :-1]
    y = sequences[y_indices, -1]
    
    return torch.from_numpy(X).float(), torch.from_numpy(y).float()

def check_date_frequency(date_series):
    dates = pd.to_datetime(date_series)
    differences = (dates - dates.shift(1)).dropna()

    daily_count = (differences == timedelta(days=1)).sum()
    hourly_count = (differences == timedelta(hours=1)).sum()
    weekly_count = (differences == timedelta(weeks=1)).sum()
    monthly_count = (differences >= timedelta(days=28, hours=23, minutes=59)).sum()

    if daily_count > max(monthly_count, hourly_count, weekly_count):
        return 365, "Daily"
    elif monthly_count > max(daily_count, hourly_count, weekly_count):
        return 12, "Monthly"
    elif weekly_count > max(daily_count, hourly_count, monthly_count):
        return 52, "Weekly"
    elif hourly_count > max(daily_count, weekly_count, monthly_count):
        return 24 * 365, "Hourly"
    else:
        return 1, "Unknown"

def detect_date_columns(df):
    date_columns = []
    for col in df.columns:
        try:
            pd.to_datetime(df[col])
            date_columns.append(col)
        except:
            continue
    return date_columns

def preprocess_data(df, date_col, input_cols, output_col, missing_value_method, encode_categorical):
    try:
        # Create a copy of the dataframe
        processed_df = df.copy()
        
        # Handle missing values
        if missing_value_method == "Interpolation":
            # Only interpolate numeric columns
            numeric_cols = processed_df.select_dtypes(include=['number']).columns
            processed_df[numeric_cols] = processed_df[numeric_cols].interpolate(method='linear')
        elif missing_value_method == "Forward Fill":
            processed_df = processed_df.ffill()
        elif missing_value_method == "Backward Fill":
            processed_df = processed_df.bfill()
        
        # Ensure date column is datetime
        processed_df[date_col] = pd.to_datetime(processed_df[date_col])
        
        # Sort by date
        processed_df = processed_df.sort_values(date_col)
        
        # Encode categorical features if requested
        if encode_categorical:
            categorical_cols = processed_df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    # Use LabelEncoder for simplicity
                    le = LabelEncoder()
                    processed_df[col] = le.fit_transform(processed_df[col].astype(str))
        
        # Select relevant columns
        selected_cols = [date_col] + input_cols + [output_col]
        processed_df = processed_df[selected_cols]
        
        # Initialize scaler for numeric columns only
        numeric_cols = processed_df.select_dtypes(include=['number']).columns
        if date_col in numeric_cols:
            numeric_cols = numeric_cols.drop(date_col)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])
        
        return processed_df, scaler
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None, None

@st.cache_data
def seasonal_decomposition_cached(data, date_col, output_col, period):
    # Check if dataset is large enough for seasonal decomposition
    if len(data) < 2 * period:
        return None, None, None, None
    
    try:
        result = seasonal_decompose(
            data.set_index(date_col)[output_col], 
            model='additive', 
            period=period
        )
        return result.trend, result.seasonal, result.resid, result.observed
    except Exception as e:
        return None, None, None, None

def save_model(model, scaler, filepath, epoch=None):
    if epoch is not None:
        base, ext = os.path.splitext(filepath)
        filepath = f"{base}_epoch_{epoch}{ext}"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'model_params': st.session_state.model_params,
        'training_params': st.session_state.training_params,
        'epoch': epoch
    }, filepath)
    return filepath

def load_model(filepath):
    try:
        # Simple fix: use weights_only=False to handle PyTorch 2.6+ security changes
        checkpoint = torch.load(filepath, weights_only=False)
        
        # Check if the checkpoint has the required keys
        required_keys = ['model_state_dict', 'scaler', 'model_params']
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Missing required key in checkpoint: {key}")
        
        # Create model with the saved parameters
        model = LSTMForecasting(**checkpoint['model_params'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint['scaler'], checkpoint['model_params'], checkpoint.get('training_params', {})
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def check_model_compatibility(model_params, input_cols, output_col):
    """Check if loaded model is compatible with current data"""
    try:
        # Check if input size matches
        expected_input_size = len(input_cols)
        if model_params["input_size"] != expected_input_size:
            return False, f"Input size mismatch: model expects {model_params['input_size']}, data has {expected_input_size}"
        
        return True, "Model is compatible"
    except Exception as e:
        return False, f"Compatibility check error: {str(e)}"

def get_saved_models():
    """Get list of saved models with proper metadata"""
    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        return []
    
    model_files = glob.glob(os.path.join(model_dir, "*.pt"))
    models_info = []
    
    for model_file in model_files:
        try:
            basename = os.path.basename(model_file)
            # Extract model name and epoch
            if '_epoch_' in basename:
                name_part = basename.split('_epoch_')[0]
                epoch_part = basename.split('_epoch_')[1].replace('.pt', '')
            else:
                name_part = basename.replace('.pt', '')
                epoch_part = "final"
            
            # Get file stats
            stat = os.stat(model_file)
            size_mb = stat.st_size / (1024 * 1024)
            mod_time = datetime.fromtimestamp(stat.st_mtime)
            
            models_info.append({
                "filename": model_file,
                "display_name": f"{name_part} (Epoch {epoch_part})",
                "epoch": epoch_part,
                "size_mb": f"{size_mb:.2f}",
                "mod_time": mod_time.strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp": mod_time
            })
        except Exception as e:
            continue
    
    # Sort by modification time (newest first)
    models_info.sort(key=lambda x: x["timestamp"], reverse=True)
    return models_info

def train_model(X_train, y_train, X_val, y_val, model_params, training_params, 
                progress_bar, status_text, train_loss_metric, val_loss_metric, save_model_name):
    # Initialize model
    model = LSTMForecasting(
        input_size=model_params["input_size"],
        lstm_hidden_size=model_params["lstm_hidden_size"],
        linear_hidden_size=model_params["linear_hidden_size"],
        lstm_num_layers=model_params["lstm_num_layers"],
        linear_num_layers=model_params["linear_num_layers"],
        output_size=model_params["output_size"]
    ).to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params["learning_rate"])
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=training_params["batch_size"], shuffle=True)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=training_params["batch_size"], shuffle=False)
    
    # Training history
    train_losses = []
    val_losses = []
    
    # Create directory for saved models if it doesn't exist
    if save_model_name:
        model_dir = "saved_models"
        os.makedirs(model_dir, exist_ok=True)
    
    # Training loop with progress bar
    with tqdm(total=training_params["epochs"], desc="Training") as pbar:
        for epoch in range(training_params["epochs"]):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Gradient clipping to prevent explosions
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                
                # Clear memory
                del inputs, targets, outputs, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    
                    # Clear memory
                    del inputs, targets, outputs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            
            # Update UI elements
            progress = (epoch + 1) / training_params["epochs"]
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch+1}/{training_params['epochs']}")
            train_loss_metric.metric("Training Loss", f"{train_loss:.4f}")
            val_loss_metric.metric("Validation Loss", f"{val_loss:.4f}")
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({"Train Loss": f"{train_loss:.4f}", "Val Loss": f"{val_loss:.4f}"})
            
            # Save model after each epoch if a model name is provided
            if save_model_name:
                saved_path = save_model(
                    model, 
                    st.session_state.scaler, 
                    os.path.join(model_dir, f"{save_model_name}.pt"),
                    epoch=epoch+1
                )
                st.session_state[f"model_path_epoch_{epoch+1}"] = saved_path
    
    return model, train_losses, val_losses

def generate_predictions(model, X_test, scaler, forecast_steps, feature_cols):
    model.eval()
    with torch.no_grad():
        # Move test data to the same device as the model
        X_test = X_test.to(device)
        predictions = model(X_test).cpu().numpy()
    
    # Reshape predictions for inverse transformation
    # Create array with same number of features as original data
    predictions_reshaped = np.zeros((predictions.shape[0] * predictions.shape[1], len(feature_cols)))
    
    # Place predictions in the last column (output column position)
    predictions_reshaped[:, -1] = predictions.flatten()
    
    # Inverse transform to get actual values
    try:
        actual_predictions = scaler.inverse_transform(predictions_reshaped)[:, -1]
        
        # Reshape back to (n_sequences, forecast_steps)
        actual_predictions = actual_predictions.reshape(predictions.shape[0], predictions.shape[1])
        
    except Exception as e:
        # Fallback: return normalized predictions
        actual_predictions = predictions
    
    return actual_predictions

def update_validation_results(processed_df, model, scaler, lag_steps, forecast_steps, train_size=0.8):
    """
    Generate predictions and update validation results for the loaded model
    """
    try:
        # Prepare data for prediction
        feature_cols = st.session_state.input_cols + [st.session_state.output_col]
        data_array = processed_df[feature_cols].values
        
        # Split into sequences
        X, y = split_sequences_optimized(data_array, lag_steps, forecast_steps)
        
        if len(X) == 0:
            st.error("Not enough data for validation")
            return False
        
        # Split into train and validation sets (same split as during training)
        split_idx = int(len(X) * train_size)
        X_val = X[split_idx:]
        y_val = y[split_idx:]
        
        if len(X_val) == 0:
            st.error("No validation data available")
            return False
        
        # Generate predictions
        model.eval()
        with torch.no_grad():
            X_val_tensor = X_val.to(device)
            predictions_tensor = model(X_val_tensor)
            predictions = predictions_tensor.cpu().numpy()
        
        # Handle single-output vs multi-output predictions
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        # Inverse transform predictions
        try:
            # Create dummy array for inverse transform
            dummy_pred = np.zeros((predictions.size, len(feature_cols)))
            dummy_pred[:, -1] = predictions.flatten()
            predictions_inv = scaler.inverse_transform(dummy_pred)[:, -1]
            predictions = predictions_inv.reshape(predictions.shape)
        except:
            # If inverse transform fails, use normalized predictions
            st.warning("Using normalized predictions (inverse transform failed)")
        
        # Get actual values (inverse transformed)
        actual_values = []
        for i in range(len(y_val)):
            actual_idx = split_idx + i + lag_steps
            if actual_idx + forecast_steps <= len(processed_df):
                # Get actual values and inverse transform
                actual_slice = processed_df[feature_cols].iloc[actual_idx:actual_idx + forecast_steps]
                actual_inv = scaler.inverse_transform(actual_slice)[:, -1]
                actual_values.append(actual_inv)
        
        if not actual_values:
            st.error("Could not extract actual values for validation")
            return False
        
        actual_values = np.array(actual_values)
        
        # Ensure shapes match
        min_len = min(len(actual_values), len(predictions))
        actual_values = actual_values[:min_len]
        predictions = predictions[:min_len]
        
        # Store in session state
        st.session_state.predictions = predictions
        st.session_state.actual_values = actual_values
        
        # Calculate residuals
        residuals = actual_values.flatten() - predictions.flatten()
        st.session_state.residuals = residuals
        
        return True
        
    except Exception as e:
        st.error(f"Error in update_validation_results: {str(e)}")
        return False

def calculate_confidence_intervals(predictions, std_dev, confidence=0.95):
    z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99% confidence
    margin = z_score * std_dev
    lower_bound = predictions - margin
    upper_bound = predictions + margin
    return lower_bound, upper_bound

def create_plotly_chart(x_data, y_data, title, x_label, y_label, mode='lines', name=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data, y=y_data, mode=mode, name=name))
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    return fig

# Main app
def main():
    # Header
    st.title("📈 Multi-Step Time Series Forecasting with LSTM")
    st.markdown("An interactive tool for time series forecasting using Long Short-Term Memory (LSTM) networks.")
    
    # Initialize all session state variables
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'train_losses' not in st.session_state:
        st.session_state.train_losses = []
    if 'val_losses' not in st.session_state:
        st.session_state.val_losses = []
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'actual_values' not in st.session_state:
        st.session_state.actual_values = None
    if 'model_params' not in st.session_state:
        st.session_state.model_params = {}
    if 'training_params' not in st.session_state:
        st.session_state.training_params = {}
    if 'residuals' not in st.session_state:
        st.session_state.residuals = None
    if 'date_col' not in st.session_state:
        st.session_state.date_col = None
    if 'input_cols' not in st.session_state:
        st.session_state.input_cols = None
    if 'output_col' not in st.session_state:
        st.session_state.output_col = None
    if 'lag_steps' not in st.session_state:
        st.session_state.lag_steps = 10
    
    # File upload section
    st.header("Data Upload")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Read the CSV file
        try:
            df = load_data(uploaded_file)
            st.session_state.data = df
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Display data info
            st.subheader("Data Information")
            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Detect date columns (without showing warning if none found)
            date_columns = detect_date_columns(df)
            if date_columns:
                st.success(f"Detected date columns: {', '.join(date_columns)}")
            
        except Exception as e:
            st.error(f"Error reading the CSV file: {str(e)}")
    
    # Feature selection section
    if st.session_state.data is not None:
        st.header("Feature Selection")
        df = st.session_state.data
        
        # Date column selection
        date_columns = detect_date_columns(df)
        if date_columns:
            date_col = st.selectbox("Select date column:", date_columns)
        else:
            date_col = st.selectbox("Select date column:", df.columns)
        
        # Input and output feature selection
        feature_cols = [col for col in df.columns if col != date_col]
        input_cols = st.multiselect("Select input features (X):", feature_cols)
        output_col = st.selectbox("Select output feature (Y):", feature_cols)
        
        # Validation for feature selection
        if not input_cols:
            st.error("Please select at least one input feature")
            return
        
        # Data preprocessing options
        st.header("Data Preprocessing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            missing_value_method = st.selectbox(
                "Missing value handling:",
                ["Interpolation", "Forward Fill", "Backward Fill", "Drop Rows"]
            )
            
            encode_categorical = st.checkbox(
                "Automatically encode non-numeric features",
                value=True,
                help="Convert categorical features to numeric using label encoding"
            )
        
        with col2:
            st.write("### Data Types")
            st.dataframe(df.dtypes.to_frame().rename(columns={0: "Data Type"}))
        
        # Preprocess button
        if st.button("Preprocess Data", type="primary"):
            with st.spinner("Preprocessing data..."):
                processed_df, scaler = preprocess_data(
                    df, date_col, input_cols, output_col, missing_value_method, encode_categorical
                )
                
                if processed_df is not None:
                    st.session_state.processed_data = processed_df
                    st.session_state.scaler = scaler
                    st.session_state.date_col = date_col
                    st.session_state.input_cols = input_cols
                    st.session_state.output_col = output_col
                    
                    st.success("Data preprocessed successfully!")
    
    # Data exploration section
    if st.session_state.processed_data is not None:
        st.header("Data Exploration")
        processed_df = st.session_state.processed_data
        date_col = st.session_state.date_col
        input_cols = st.session_state.input_cols
        output_col = st.session_state.output_col
        
        # Display preprocessed data
        st.subheader("Preprocessed Data")
        st.dataframe(processed_df.head(10), use_container_width=True)
        
        # Data summary statistics
        st.subheader("Data Summary")
        st.dataframe(processed_df.describe(), use_container_width=True)
        
        # Data visualization
        st.subheader("Data Visualization")
        
        # Time series plot of output variable
        fig = create_plotly_chart(
            processed_df[date_col], processed_df[output_col],
            f"Time Series of {output_col}", "Date", output_col
        )
        st.plotly_chart(fig, use_container_width=True, key="time_series_plot")
        
        # Distribution plot
        fig_hist = px.histogram(
            processed_df,
            x=output_col,
            title=f"Distribution of {output_col}"
        )
        st.plotly_chart(fig_hist, use_container_width=True, key="distribution_plot")
        
        # Correlation matrix
        numeric_cols = processed_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            corr_matrix = processed_df[numeric_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix"
            )
            st.plotly_chart(fig_corr, use_container_width=True, key="correlation_matrix")
        
        # Seasonal decomposition
        st.subheader("Seasonal Decomposition")
        
        # Detect frequency
        period, freq_type = check_date_frequency(processed_df[date_col])
        st.write(f"Detected data frequency: {freq_type}")
        
        # Allow user to adjust period
        user_period = st.number_input(
            "Seasonal period (leave 0 for auto-detection):",
            min_value=0,
            value=0,
            step=1
        )
        
        if user_period > 0:
            period = user_period
        
        # Check if dataset is large enough
        if len(processed_df) < 2 * period:
            st.warning(f"Dataset too small for seasonal decomposition with period={period}. Need at least {2 * period} data points.")
        else:
            # Perform seasonal decomposition
            with st.spinner("Performing seasonal decomposition..."):
                trend, seasonal, resid, observed = seasonal_decomposition_cached(
                    processed_df, date_col, output_col, period
                )
            
            if trend is not None:
                # Create plots
                st.subheader("Seasonal Decomposition Results")
                
                # Original data
                fig_observed = create_plotly_chart(
                    processed_df[date_col], observed,
                    "Original Data", "Date", "Value"
                )
                st.plotly_chart(fig_observed, use_container_width=True, key="original_data")
                
                # Trend
                fig_trend = create_plotly_chart(
                    processed_df[date_col], trend,
                    "Trend Component", "Date", "Value"
                )
                st.plotly_chart(fig_trend, use_container_width=True, key="trend_component")
                
                # Seasonal
                fig_seasonal = create_plotly_chart(
                    processed_df[date_col], seasonal,
                    "Seasonal Component", "Date", "Value"
                )
                st.plotly_chart(fig_seasonal, use_container_width=True, key="seasonal_component")
                
                # Residual
                fig_resid = create_plotly_chart(
                    processed_df[date_col], resid,
                    "Residual Component", "Date", "Value"
                )
                st.plotly_chart(fig_resid, use_container_width=True, key="residual_component")
    
    # Model training section
    if st.session_state.processed_data is not None:
        st.header("Model Training")
        processed_df = st.session_state.processed_data
        date_col = st.session_state.date_col
        input_cols = st.session_state.input_cols
        output_col = st.session_state.output_col
        
        st.subheader("Model Configuration")
        
        # Lag and forecast steps
        col1, col2 = st.columns(2)
        
        with col1:
            lag_steps = st.number_input(
                "Lag steps (look-back window):",
                min_value=1,
                max_value=len(processed_df) // 2,
                value=10,
                step=1,
                help="Number of previous time steps to use for prediction"
            )
            # Store lag steps in session state
            st.session_state.lag_steps = lag_steps
        
        with col2:
            forecast_steps = st.number_input(
                "Forecast steps (prediction horizon):",
                min_value=1,
                max_value=len(processed_df) // 3,
                value=5,
                step=1,
                help="Number of future time steps to predict"
            )
        
        # Check if lag + forecast steps are valid
        if lag_steps + forecast_steps >= len(processed_df):
            st.error(f"Lag steps + forecast steps must be less than the total number of data points ({len(processed_df)}).")
        else:
            # Train-test split
            train_size = st.slider(
                "Training data proportion:",
                min_value=0.5,
                max_value=0.9,
                value=0.8,
                step=0.05,
                help="Proportion of data to use for training"
            )
            
            # LSTM parameters
            st.subheader("LSTM Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                lstm_layers = st.slider(
                    "LSTM layers:",
                    min_value=1,
                    max_value=5,
                    value=MODEL_CONFIG["default_lstm_layers"],
                    step=1,
                    help="Number of LSTM layers in the model"
                )
                
                lstm_neurons = st.slider(
                    "LSTM neurons per layer:",
                    min_value=10,
                    max_value=500,
                    value=MODEL_CONFIG["default_lstm_neurons"],
                    step=10,
                    help="Number of neurons in each LSTM layer"
                )
            
            with col2:
                linear_layers = st.slider(
                    "Linear layers:",
                    min_value=1,
                    max_value=5,
                    value=MODEL_CONFIG["default_linear_layers"],
                    step=1,
                    help="Number of linear layers after LSTM"
                )
                
                linear_neurons = st.slider(
                    "Linear neurons per layer:",
                    min_value=10,
                    max_value=500,
                    value=MODEL_CONFIG["default_linear_neurons"],
                    step=10,
                    help="Number of neurons in each linear layer"
                )
            
            # Training parameters
            st.subheader("Training Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                epochs = st.number_input(
                    "Number of epochs:",
                    min_value=1,
                    max_value=MODEL_CONFIG["max_epochs"],
                    value=50,
                    step=1,
                    help="Number of complete passes through the training dataset"
                )
                
                batch_size = st.number_input(
                    "Batch size:",
                    min_value=1,
                    max_value=128,
                    value=32,
                    step=1,
                    help="Number of samples per gradient update"
                )
            
            with col2:
                learning_rate = st.number_input(
                    "Learning rate:",
                    min_value=0.0001,
                    max_value=0.1,
                    value=MODEL_CONFIG["default_learning_rate"],
                    format="%.4f",
                    step=0.0001,
                    help="Step size at each update"
                )
                
                # Add model saving/loading options
                save_model_name = st.text_input("Model name (for saving):", value="lstm_model")
            
            # Train model button
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Train Model", type="primary"):
                    with st.spinner("Preparing data..."):
                        # Prepare data for training
                        feature_cols = input_cols + [output_col]
                        data_array = processed_df[feature_cols].values
                        
                        # Split into sequences using optimized function
                        X, y = split_sequences_optimized(data_array, lag_steps, forecast_steps)
                        
                        # Check if sequences were created successfully
                        if len(X) == 0 or len(y) == 0:
                            st.error("Not enough data for the specified lag and forecast steps")
                            return
                        
                        # Split into train and validation sets
                        split_idx = int(len(X) * train_size)
                        X_train, X_val = X[:split_idx], X[split_idx:]
                        y_train, y_val = y[:split_idx], y[split_idx:]
                        
                        # Store model parameters
                        st.session_state.model_params = {
                            "input_size": len(input_cols),  # This is the number of features
                            "lstm_hidden_size": lstm_neurons,
                            "linear_hidden_size": linear_neurons,
                            "lstm_num_layers": lstm_layers,
                            "linear_num_layers": linear_layers,
                            "output_size": forecast_steps
                        }
                        
                        # Store training parameters
                        st.session_state.training_params = {
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "learning_rate": learning_rate
                        }
                    
                    # Train model
                    with st.spinner("Training model..."):
                        try:
                            # Create UI elements for training progress
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            col1, col2 = st.columns(2)
                            train_loss_metric = col1.empty()
                            val_loss_metric = col2.empty()
                            
                            # Train model
                            model, train_losses, val_losses = train_model(
                                X_train, y_train, X_val, y_val,
                                st.session_state.model_params,
                                st.session_state.training_params,
                                progress_bar, status_text, train_loss_metric, val_loss_metric, save_model_name
                            )
                            
                            # Store model and training history
                            st.session_state.model = model
                            st.session_state.train_losses = train_losses
                            st.session_state.val_losses = val_losses
                            
                            st.success("Model trained successfully!")
                            
                            # Generate predictions on validation set
                            with st.spinner("Generating predictions..."):
                                predictions = generate_predictions(
                                    model, X_val, st.session_state.scaler, forecast_steps, feature_cols
                                )
                                
                                # Get actual values
                                actual_values = []
                                for i in range(len(y_val)):
                                    actual_idx = split_idx + i + lag_steps
                                    if actual_idx + forecast_steps <= len(processed_df):
                                        actual_values.append(
                                            processed_df[output_col].iloc[actual_idx:actual_idx + forecast_steps].values
                                        )
                                
                                if actual_values:
                                    actual_values = np.array(actual_values)
                                    
                                    # Store predictions and actual values
                                    st.session_state.predictions = predictions
                                    st.session_state.actual_values = actual_values
                                    
                                    # Calculate residuals
                                    residuals = actual_values.flatten() - predictions.flatten()
                                    st.session_state.residuals = residuals
                            
                            # Clean up memory
                            del model, X_train, y_train, X_val, y_val
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                        except Exception as e:
                            st.error(f"Error training model: {str(e)}")
            
            # Model loading section - SIMPLIFIED VERSION
            st.subheader("Load Saved Model")
            
            # Get list of saved models
            saved_models = get_saved_models()
            
            if saved_models:
                # Create a simple dropdown for model selection
                model_options = [f"{model['display_name']} - {model['mod_time']}" for model in saved_models]
                selected_model_display = st.selectbox("Select a model to load:", model_options)
                
                # Find the selected model
                selected_index = model_options.index(selected_model_display)
                selected_model = saved_models[selected_index]
                
                if st.button("Load Selected Model", key="load_selected_model"):
                    with st.spinner("Loading model..."):
                        try:
                            model, scaler, model_params, training_params = load_model(selected_model["filename"])
                            
                            if model is not None:
                                # Check compatibility with current data
                                is_compatible, message = check_model_compatibility(
                                    model_params, 
                                    st.session_state.input_cols, 
                                    st.session_state.output_col
                                )
                                
                                if is_compatible:
                                    # Store in session state
                                    st.session_state.model = model
                                    st.session_state.scaler = scaler
                                    st.session_state.model_params = model_params
                                    
                                    # For loaded models, initialize empty training history
                                    st.session_state.train_losses = []  # Empty list
                                    st.session_state.val_losses = []    # Empty list
                                    
                                    # Update validation results with the loaded model
                                    success = update_validation_results(
                                        st.session_state.processed_data, 
                                        model, 
                                        scaler, 
                                        st.session_state.lag_steps,
                                        model_params["output_size"],
                                        train_size=0.8
                                    )
                                    
                                    if success:
                                        st.success(f"✅ Model loaded successfully! {message}")
                                        st.rerun()  # Refresh to show updated results
                                    else:
                                        st.error("Failed to update validation results.")
                                else:
                                    st.error(f"❌ Model incompatible: {message}")
                        except Exception as e:
                            st.error(f"❌ Error loading model: {str(e)}")
            else:
                st.info("No saved models found. Train a model first to see saved models here.")
    
    # Results section with tabs
    if st.session_state.model is not None:
        st.header("Model Results")
        
        # Create tabs
        tab1, tab2 = st.tabs(["Training Progress", "Validation Results"])
        
        # Tab 1: Training Progress
        with tab1:
            st.subheader("Training Progress")
            
            # Check if we have training history (only for newly trained models)
            if (st.session_state.train_losses and 
                len(st.session_state.train_losses) > 0):
                
                # Display final training metrics
                col1, col2 = st.columns(2)
                col1.metric("Final Training Loss", f"{st.session_state.train_losses[-1]:.4f}")
                col2.metric("Final Validation Loss", f"{st.session_state.val_losses[-1]:.4f}")
                
                # Display loss chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(st.session_state.train_losses) + 1)),
                    y=st.session_state.train_losses,
                    mode='lines',
                    name='Training Loss',
                    line=dict(color='#1f77b4')
                ))
                fig.add_trace(go.Scatter(
                    x=list(range(1, len(st.session_state.val_losses) + 1)),
                    y=st.session_state.val_losses,
                    mode='lines',
                    name='Validation Loss',
                    line=dict(color='#ff7f0e')
                ))
                fig.update_layout(
                    title="Training and Validation Loss",
                    xaxis_title="Epoch",
                    yaxis_title="Loss",
                    hovermode='x unified',
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True, key="final_loss_chart")
            else:
                st.info("📊 No training history available for loaded models. Training history is only available for models trained in this session.")
                
                # Show model information instead
                if st.session_state.model_params:
                    st.subheader("Model Architecture")
                    col1, col2 = st.columns(2)
                    col1.metric("Input Features", st.session_state.model_params.get("input_size", "N/A"))
                    col1.metric("LSTM Layers", st.session_state.model_params.get("lstm_num_layers", "N/A"))
                    col2.metric("LSTM Neurons", st.session_state.model_params.get("lstm_hidden_size", "N/A"))
                    col2.metric("Output Steps", st.session_state.model_params.get("output_size", "N/A"))
        
        # Tab 2: Validation Results
        with tab2:
            st.subheader("Validation Results")
            
            # Validation results section
            if st.session_state.predictions is not None and st.session_state.actual_values is not None:
                predictions = st.session_state.predictions
                actual_values = st.session_state.actual_values
                residuals = st.session_state.residuals
                forecast_steps = st.session_state.model_params["output_size"]
                
                # Calculate metrics
                mse = mean_squared_error(actual_values.flatten(), predictions.flatten())
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(actual_values.flatten(), predictions.flatten())
                mape = mean_absolute_percentage_error(actual_values.flatten(), predictions.flatten()) * 100
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MSE", f"{mse:.4f}")
                col2.metric("RMSE", f"{rmse:.4f}")
                col3.metric("MAE", f"{mae:.4f}")
                col4.metric("MAPE", f"{mape:.2f}%")
                
                # Plot predictions vs actual values
                fig_pred = go.Figure()
                
                # Add actual values
                for i in range(min(5, len(actual_values))):  # Limit to 5 series for readability
                    idx = len(st.session_state.processed_data) - len(actual_values) * forecast_steps + i * forecast_steps
                    dates = st.session_state.processed_data[st.session_state.date_col].iloc[idx:idx + forecast_steps]
                    
                    # Inverse transform actual values
                    actual_values_transformed = st.session_state.scaler.inverse_transform(
                        st.session_state.processed_data[st.session_state.input_cols + [st.session_state.output_col]].iloc[idx:idx + forecast_steps]
                    )[:, -1]
                    
                    fig_pred.add_trace(go.Scatter(
                        x=dates,
                        y=actual_values_transformed,
                        mode='lines+markers',
                        name=f'Actual {i+1}',
                        marker=dict(size=6)
                    ))
                
                # Add predicted values
                for i in range(min(5, len(predictions))):  # Limit to 5 series for readability
                    idx = len(st.session_state.processed_data) - len(predictions) * forecast_steps + i * forecast_steps
                    dates = st.session_state.processed_data[st.session_state.date_col].iloc[idx:idx + forecast_steps]
                    
                    fig_pred.add_trace(go.Scatter(
                        x=dates,
                        y=predictions[i],
                        mode='lines+markers',
                        name=f'Predicted {i+1}',
                        line=dict(dash='dash'),
                        marker=dict(size=6)
                    ))
                
                fig_pred.update_layout(
                    title="Predicted vs Actual Values",
                    xaxis_title="Date",
                    yaxis_title=st.session_state.output_col,
                    hovermode='x unified',
                    template="plotly_white"
                )
                st.plotly_chart(fig_pred, use_container_width=True, key="actual_vs_predicted")
                
                # Plot residuals
                fig_resid = px.histogram(
                    residuals,
                    nbins=30,
                    title="Distribution of Residuals (Prediction Errors)",
                    template="plotly_white"
                )
                fig_resid.update_layout(
                    xaxis_title="Residual Value",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig_resid, use_container_width=True, key="residuals_histogram")
                
                # Download options
                st.subheader("Download Results")
                
                # Create a dataframe with predictions and actual values
                results_df = pd.DataFrame({
                    'Actual': actual_values.flatten(),
                    'Predicted': predictions.flatten(),
                    'Residuals': residuals
                })
                
                # Download predictions as CSV
                csv = results_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download Predictions CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # Download plot as PNG
                img_bytes = fig_pred.to_image(format="png", scale=2, engine="kaleido")
                b64_img = base64.b64encode(img_bytes).decode()
                href_img = f'<a href="data:image/png;base64,{b64_img}" download="forecast_plot.png">Download Forecast Plot</a>'
                st.markdown(href_img, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #808080; padding: 10px;">
            Made with Streamlit ❤️ by Neelu
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
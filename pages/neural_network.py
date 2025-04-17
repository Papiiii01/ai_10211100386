import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

class CustomDataset(Dataset):
    def __init__(self, X, y):
        # Convert pandas Series/DataFrame to numpy arrays if needed
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
            
        self.X = torch.FloatTensor(X)
        if isinstance(y[0], (str, np.str_)) or len(np.unique(y)) < 10:
            self.y = torch.LongTensor(y)
        else:
            self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, task_type='classification'):
        super(FeedforwardNN, self).__init__()
        self.task_type = task_type
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
        if task_type == 'classification':
            self.activation = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.network(x)
        if self.task_type == 'classification':
            x = self.activation(x)
        return x

def detect_feature_types(df):
    """Detect numerical and categorical columns in the dataframe."""
    numeric_features = []
    categorical_features = []
    
    for column in df.columns:
        if df[column].dtype == 'object' or (df[column].dtype == 'int64' and len(df[column].unique()) < 10):
            categorical_features.append(column)
        else:
            numeric_features.append(column)
            
    return numeric_features, categorical_features

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, task_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.write(f"Training on: {device}")
    model = model.to(device)
    
    # Initialize plots
    plot_placeholder = st.empty()
    metrics_placeholder = st.empty()
    
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if task_type == 'classification':
                loss = criterion(outputs, targets)
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == targets).sum().item()
            else:
                targets = targets.view(-1, 1)
                loss = criterion(outputs, targets)
            
            train_total += targets.size(0)
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                if task_type == 'classification':
                    loss = criterion(outputs, targets)
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == targets).sum().item()
                else:
                    targets = targets.view(-1, 1)
                    loss = criterion(outputs, targets)
                
                val_total += targets.size(0)
                val_loss += loss.item()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if task_type == 'classification':
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            train_metrics.append(train_acc)
            val_metrics.append(val_acc)
            metric_name = 'Accuracy'
        else:
            train_metrics.append(train_loss)
            val_metrics.append(val_loss)
            metric_name = 'MSE'
        
        # Update plots
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=train_losses, name='Train Loss', line=dict(color='blue')))
        fig.add_trace(go.Scatter(y=val_losses, name='Val Loss', line=dict(color='red')))
        fig.update_layout(title='Training Progress', xaxis_title='Epoch', yaxis_title='Loss')
        plot_placeholder.plotly_chart(fig)
        
        # Update metrics
        cols = metrics_placeholder.columns(2)
        cols[0].metric(f"Training {metric_name}", f"{train_metrics[-1]:.2f}")
        cols[1].metric(f"Validation {metric_name}", f"{val_metrics[-1]:.2f}")
        
    return model

def show_neural_network():
    st.title("🧠 Neural Network Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Display dataset info
        st.subheader("Dataset Preview")
        st.write(df.head())
        st.write("Shape:", df.shape)
        
        # Detect feature types
        numeric_features, categorical_features = detect_feature_types(df)
        
        # Feature selection
        st.subheader("Feature Selection")
        selected_features = st.multiselect(
            "Select features:",
            df.columns.tolist(),
            default=[col for col in df.columns if col != df.columns[-1]]
        )
        target = st.selectbox("Select target variable:", df.columns.tolist())
        
        if len(selected_features) > 0 and target and target not in selected_features:
            # Split features by type
            selected_numeric = [f for f in selected_features if f in numeric_features]
            selected_categorical = [f for f in selected_features if f in categorical_features]
            
            # Create preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), selected_numeric),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), selected_categorical)
                ],
                remainder='drop'
            )
            
            # Prepare the data
            X = df[selected_features]
            y = df[target]
            
            # Handle target variable
            if target in categorical_features:
                le = LabelEncoder()
                y = le.fit_transform(y)
                is_classification = True
            else:
                is_classification = len(np.unique(y)) < 10
            
            # Transform features
            X = preprocessor.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create datasets
            train_dataset = CustomDataset(X_train, y_train)
            test_dataset = CustomDataset(X_test, y_test)
            
            # Hyperparameters
            st.subheader("Model Configuration")
            col1, col2, col3 = st.columns(3)
            with col1:
                hidden_size = st.slider("Hidden Layer Size", 32, 256, 128, 32)
            with col2:
                learning_rate = st.select_slider(
                    "Learning Rate",
                    options=[0.0001, 0.001, 0.01, 0.1],
                    value=0.001
                )
            with col3:
                batch_size = st.select_slider(
                    "Batch Size",
                    options=[16, 32, 64, 128],
                    value=32
                )
            
            epochs = st.slider("Number of Epochs", 5, 50, 20)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # Initialize model
            input_size = X.shape[1]  # After preprocessing
            output_size = len(np.unique(y)) if is_classification else 1
            task_type = 'classification' if is_classification else 'regression'
            model = FeedforwardNN(input_size, hidden_size, output_size, task_type)
            
            # Loss and optimizer
            if is_classification:
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.MSELoss()
            
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            # Training
            if st.button("Train Model"):
                st.subheader("Training Progress")
                model = train_model(model, train_loader, test_loader, criterion, optimizer, epochs, task_type)
                
                # Save model
                torch.save(model.state_dict(), 'model.pth')
                st.success("Model trained and saved successfully!")
                
                # Prediction interface
                st.subheader("Make Predictions")
                prediction_input = st.text_area(
                    "Enter new data for prediction (comma-separated values):",
                    help=f"Enter values for features in this order: {', '.join(selected_features)}"
                )
                
                if prediction_input:
                    try:
                        # Process input
                        input_data = pd.DataFrame([prediction_input.split(',')], columns=selected_features)
                        input_data = preprocessor.transform(input_data)
                        input_tensor = torch.FloatTensor(input_data)
                        
                        # Make prediction
                        model.eval()
                        with torch.no_grad():
                            prediction = model(input_tensor)
                            if is_classification:
                                _, predicted_class = torch.max(prediction, 1)
                                if target in categorical_features:
                                    result = le.inverse_transform(predicted_class.numpy())[0]
                                else:
                                    result = predicted_class.item()
                            else:
                                result = prediction.item()
                            
                        st.write(f"Prediction: {result}")
                    except Exception as e:
                        st.error(f"Error in prediction: {str(e)}")
        
        else:
            st.warning("Please select features and a target variable")
    else:
        st.info("Please upload a CSV file to begin. The file should contain your training data with features and target variables.")
        
if __name__ == "__main__":
    show_neural_network() 
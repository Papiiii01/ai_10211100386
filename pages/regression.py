import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go

def show_regression():
    st.header("📊 Regression Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        # Load and preview data
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Data info
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Shape:", df.shape)
        with col2:
            st.write("Missing values:", df.isnull().sum().sum())
        
        # Feature selection with type indication
        st.subheader("Select Features")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        st.write("**Available Features:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Numerical Features:")
            feature_cols = st.multiselect("Select numerical features", 
                                        numeric_cols,
                                        help="Select the numerical features to use for prediction")
        with col2:
            st.write("Categorical Features:")
            categorical_feature_cols = st.multiselect("Select categorical features",
                                                    categorical_cols,
                                                    help="Select the categorical features to use for prediction")
        
        # Combine selected features
        all_features = feature_cols + categorical_feature_cols
        
        # Target variable selection (only numerical)
        st.write("**Target Variable:**")
        target_col = st.selectbox("Select target variable", 
                                 numeric_cols,
                                 help="Select the numerical variable you want to predict")
        
        if len(all_features) > 0 and target_col and target_col not in all_features:
            # Preprocessing options
            st.subheader("Data Preprocessing")
            col1, col2 = st.columns(2)
            with col1:
                scaler_type = st.selectbox(
                    "Select numerical feature scaling",
                    ["StandardScaler", "MinMaxScaler", "RobustScaler"],
                    help="StandardScaler: standardize features by removing the mean and scaling to unit variance\n"
                         "MinMaxScaler: scale features to a fixed range (0, 1)\n"
                         "RobustScaler: scale features using statistics that are robust to outliers"
                )
            with col2:
                test_size = st.slider("Test set size (%)", 10, 40, 20, 
                                    help="Percentage of data to use for testing")
            
            # Create preprocessing pipeline
            if scaler_type == "StandardScaler":
                scaler = StandardScaler()
            elif scaler_type == "MinMaxScaler":
                scaler = MinMaxScaler()
            else:
                scaler = RobustScaler()
                
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', scaler, feature_cols),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_feature_cols)
                ])
            
            # Model selection
            st.subheader("Model Selection")
            col1, col2 = st.columns(2)
            with col1:
                model_type = st.selectbox(
                    "Select regression model",
                    ["Linear Regression", "Ridge Regression", "Lasso Regression"],
                    help="Linear Regression: simple linear model\n"
                         "Ridge Regression: linear regression with L2 regularization\n"
                         "Lasso Regression: linear regression with L1 regularization"
                )
            
            with col2:
                if model_type in ["Ridge Regression", "Lasso Regression"]:
                    alpha = st.slider("Regularization strength (alpha)", 
                                    0.0, 10.0, 1.0,
                                    help="Higher values mean stronger regularization")
                    model = Ridge(alpha=alpha) if model_type == "Ridge Regression" else Lasso(alpha=alpha)
                else:
                    model = LinearRegression()
            
            # Create full pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
            
            # Split data
            X = df[all_features]
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Model evaluation
            st.subheader("Model Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Squared Error", f"{mean_squared_error(y_test, y_pred):.2f}")
            with col2:
                st.metric("Mean Absolute Error", f"{mean_absolute_error(y_test, y_pred):.2f}")
            with col3:
                st.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
            
            # Visualization
            st.subheader("Prediction Visualization")
            
            # Scatter plot
            fig = px.scatter(x=y_test, y=y_pred, 
                           labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                           title='Actual vs Predicted Values')
            
            # Add perfect prediction line
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], 
                                   y=[y_test.min(), y_test.max()],
                                   mode='lines', name='Perfect Prediction',
                                   line=dict(dash='dash')))
            
            # Add confidence interval
            residuals = y_test - y_pred
            std_residuals = np.std(residuals)
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                   y=[y_test.min() - 2*std_residuals, y_test.max() - 2*std_residuals],
                                   mode='lines', name='95% Confidence Interval',
                                   line=dict(dash='dot', color='red')))
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                   y=[y_test.min() + 2*std_residuals, y_test.max() + 2*std_residuals],
                                   mode='lines', showlegend=False,
                                   line=dict(dash='dot', color='red')))
            
            st.plotly_chart(fig)
            
            # Residual plot
            fig = px.scatter(x=y_pred, y=residuals,
                           labels={'x': 'Predicted Values', 'y': 'Residuals'},
                           title='Residual Plot')
            fig.add_hline(y=0, line_dash="dash")
            st.plotly_chart(fig)
            
            # Feature importance for linear models
            if hasattr(pipeline.named_steps['regressor'], 'coef_'):
                st.subheader("Feature Importance")
                
                # Get feature names after preprocessing
                feature_names = (feature_cols + 
                               [f"{col}_{val}" for col, vals in 
                                zip(categorical_feature_cols, 
                                    pipeline.named_steps['preprocessor']
                                    .named_transformers_['cat']
                                    .categories_) 
                                for val in vals[1:]])
                
                # Get coefficients
                coefficients = pipeline.named_steps['regressor'].coef_
                
                # Create feature importance plot
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': np.abs(coefficients)
                }).sort_values('Importance', ascending=True)
                
                fig = px.bar(importance_df, x='Importance', y='Feature', 
                           orientation='h',
                           title='Feature Importance (Absolute Coefficient Values)')
                st.plotly_chart(fig)
            
            # Custom prediction
            st.subheader("Make Predictions")
            st.write("Enter values for your features:")
            
            # Create input fields for features
            input_data = {}
            
            # Numerical features
            if feature_cols:
                st.write("**Numerical Features:**")
                cols = st.columns(min(3, len(feature_cols)))
                for i, col in enumerate(feature_cols):
                    with cols[i % 3]:
                        input_data[col] = st.number_input(f"{col}", 
                                                        value=float(df[col].mean()),
                                                        help=f"Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}")
            
            # Categorical features
            if categorical_feature_cols:
                st.write("**Categorical Features:**")
                cols = st.columns(min(3, len(categorical_feature_cols)))
                for i, col in enumerate(categorical_feature_cols):
                    with cols[i % 3]:
                        input_data[col] = st.selectbox(f"{col}", 
                                                     df[col].unique(),
                                                     help=f"Most common: {df[col].mode().iloc[0]}")
            
            if st.button("Predict", help="Click to get prediction based on input values"):
                # Create input DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Make prediction
                prediction = pipeline.predict(input_df)[0]
                st.success(f"Predicted {target_col}: {prediction:.2f}")
                
                # Show prediction range
                lower_bound = prediction - 2*std_residuals
                upper_bound = prediction + 2*std_residuals
                st.info(f"95% Confidence Interval: [{lower_bound:.2f}, {upper_bound:.2f}]")
            
        else:
            st.warning("Please select at least one feature and a target variable (target should not be in features).")
            
    else:
        st.info("Please upload a CSV file to begin regression analysis.")
        
        # Example dataset format
        st.subheader("Example Dataset Format")
        example_data = {
            'size': [1500, 2000, 1200, 1800, 2200],
            'bedrooms': [3, 4, 2, 3, 4],
            'location': ['urban', 'suburban', 'urban', 'rural', 'suburban'],
            'price': [250000, 350000, 200000, 280000, 400000]
        }
        st.dataframe(pd.DataFrame(example_data))
        st.write("Your CSV file should have:")
        st.write("- Numerical features (e.g., size, bedrooms)")
        st.write("- Optional categorical features (e.g., location)")
        st.write("- A numerical target variable (e.g., price)") 
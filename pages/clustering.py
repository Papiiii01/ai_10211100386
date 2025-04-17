import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

def show_clustering():
    st.header("🔍 Clustering Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        # Load and preview data
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Feature selection
        st.subheader("Select Features for Clustering")
        feature_cols = st.multiselect("Select features for clustering", df.columns.tolist())
        
        if len(feature_cols) >= 2:
            # Data preprocessing
            X = df[feature_cols]
            
            # Identify numerical and categorical columns
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            
            # Create preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
                ])
            
            # Create full pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('clusterer', KMeans(random_state=42))
            ])
            
            # Number of clusters
            n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
            pipeline.named_steps['clusterer'].n_clusters = n_clusters
            
            # Perform clustering
            df['Cluster'] = pipeline.fit_predict(X)
            
            # Get preprocessed data and cluster centers
            X_processed = pipeline.named_steps['preprocessor'].transform(X)
            cluster_centers = pipeline.named_steps['clusterer'].cluster_centers_
            
            # Visualization
            st.subheader("Cluster Visualization")
            
            if len(feature_cols) == 2 and len(numerical_cols) == 2:
                # 2D visualization for numerical features
                fig = go.Figure()
                
                # Add scatter plot for data points
                fig.add_trace(go.Scatter(
                    x=df[numerical_cols[0]], 
                    y=df[numerical_cols[1]],
                    mode='markers',
                    marker=dict(
                        color=df['Cluster'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name='Data Points'
                ))
                
                # Add scatter plot for centroids
                inverse_transform = pipeline.named_steps['preprocessor'].named_transformers_['num'].inverse_transform
                centers_original_scale = inverse_transform(cluster_centers[:, :len(numerical_cols)])
                
                fig.add_trace(go.Scatter(
                    x=centers_original_scale[:, 0],
                    y=centers_original_scale[:, 1],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=15,
                        symbol='x'
                    ),
                    name='Cluster Centroids'
                ))
                
                fig.update_layout(title='2D Cluster Visualization with Centroids',
                                xaxis_title=numerical_cols[0],
                                yaxis_title=numerical_cols[1])
                st.plotly_chart(fig)
                
            elif len(feature_cols) == 3 and len(numerical_cols) == 3:
                # 3D visualization for numerical features
                fig = go.Figure()
                
                # Add scatter plot for data points
                fig.add_trace(go.Scatter3d(
                    x=df[numerical_cols[0]], 
                    y=df[numerical_cols[1]], 
                    z=df[numerical_cols[2]],
                    mode='markers',
                    marker=dict(
                        color=df['Cluster'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name='Data Points'
                ))
                
                # Add scatter plot for centroids
                inverse_transform = pipeline.named_steps['preprocessor'].named_transformers_['num'].inverse_transform
                centers_original_scale = inverse_transform(cluster_centers[:, :len(numerical_cols)])
                
                fig.add_trace(go.Scatter3d(
                    x=centers_original_scale[:, 0],
                    y=centers_original_scale[:, 1],
                    z=centers_original_scale[:, 2],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=8,
                        symbol='x'
                    ),
                    name='Cluster Centroids'
                ))
                
                fig.update_layout(title='3D Cluster Visualization with Centroids',
                                scene=dict(
                                    xaxis_title=numerical_cols[0],
                                    yaxis_title=numerical_cols[1],
                                    zaxis_title=numerical_cols[2]
                                ))
                st.plotly_chart(fig)
                
            else:
                # PCA for dimensionality reduction
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_processed)
                centers_pca = pca.transform(cluster_centers)
                
                fig = go.Figure()
                
                # Add scatter plot for data points
                fig.add_trace(go.Scatter(
                    x=X_pca[:, 0], 
                    y=X_pca[:, 1],
                    mode='markers',
                    marker=dict(
                        color=df['Cluster'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name='Data Points'
                ))
                
                # Add scatter plot for centroids
                fig.add_trace(go.Scatter(
                    x=centers_pca[:, 0],
                    y=centers_pca[:, 1],
                    mode='markers',
                    marker=dict(
                        color='red',
                        size=15,
                        symbol='x'
                    ),
                    name='Cluster Centroids'
                ))
                
                fig.update_layout(title='Cluster Visualization with Centroids (PCA)',
                                xaxis_title='First Principal Component',
                                yaxis_title='Second Principal Component')
                st.plotly_chart(fig)
            
            # Cluster statistics
            st.subheader("Cluster Statistics")
            
            # Calculate statistics separately for numerical and categorical columns
            stats_dict = {}
            
            # For numerical columns
            if len(numerical_cols) > 0:
                num_stats = df.groupby('Cluster')[numerical_cols].agg(['mean', 'count'])
                stats_dict['Numerical Features'] = num_stats
            
            # For categorical columns
            if len(categorical_cols) > 0:
                # Calculate mode for each categorical column
                mode_stats = df.groupby('Cluster')[categorical_cols].agg(lambda x: x.mode().iloc[0] if not x.empty else None)
                mode_stats.columns = [f"{col}_most_common" for col in categorical_cols]
                
                # Calculate count
                count_stats = df.groupby('Cluster').size().to_frame('count')
                
                # Combine statistics
                cat_stats = pd.concat([count_stats, mode_stats], axis=1)
                stats_dict['Categorical Features'] = cat_stats
            
            # Display statistics
            for stat_type, stats in stats_dict.items():
                st.write(f"**{stat_type}**")
                st.dataframe(stats)
            
            # Download clustered data
            st.download_button(
                label="Download Clustered Data",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='clustered_data.csv',
                mime='text/csv'
            )
            
        else:
            st.warning("Please select at least 2 features for clustering.")
            
    else:
        st.info("Please upload a CSV file to begin clustering analysis.") 
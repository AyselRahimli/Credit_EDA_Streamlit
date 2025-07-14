import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Credit Approval EDA Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Generate sample credit approval data similar to UCI dataset"""
    np.random.seed(42)
    n_samples = 690
    
    # Generate synthetic data similar to UCI Credit Approval dataset
    data = {
        'A1': np.random.choice(['a', 'b'], n_samples, p=[0.6, 0.4]),  # Categorical
        'A2': np.random.normal(31.5, 11.9, n_samples),  # Age (continuous)
        'A3': np.random.normal(4.76, 4.96, n_samples),  # Debt (continuous)
        'A4': np.random.choice(['u', 'y', 'l', 't'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),  # Married
        'A5': np.random.choice(['g', 'p', 'gg'], n_samples, p=[0.6, 0.3, 0.1]),  # Bank customer
        'A6': np.random.choice(['c', 'd', 'cc', 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', 'aa', 'ff'], n_samples),
        'A7': np.random.choice(['v', 'h', 'bb', 'j', 'n', 'z', 'dd', 'ff', 'o'], n_samples),
        'A8': np.random.normal(2.22, 3.35, n_samples),  # Continuous
        'A9': np.random.choice(['t', 'f'], n_samples, p=[0.7, 0.3]),  # Boolean
        'A10': np.random.choice(['t', 'f'], n_samples, p=[0.5, 0.5]),  # Boolean
        'A11': np.random.randint(0, 20, n_samples),  # Integer
        'A12': np.random.choice(['t', 'f'], n_samples, p=[0.4, 0.6]),  # Boolean
        'A13': np.random.choice(['g', 'p', 's'], n_samples, p=[0.6, 0.3, 0.1]),  # Categorical
        'A14': np.random.normal(184.0, 173.8, n_samples),  # Continuous
        'A15': np.random.normal(1017.4, 5210.1, n_samples),  # Income (continuous)
        'Class': np.random.choice(['+', '-'], n_samples, p=[0.56, 0.44])  # Target
    }
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
    for col in ['A1', 'A2', 'A6', 'A7']:
        missing_col_indices = np.random.choice(missing_indices, len(missing_indices)//4, replace=False)
        for idx in missing_col_indices:
            data[col][idx] = np.nan
    
    return pd.DataFrame(data)

def preprocess_data(df):
    """Preprocess the data for analysis"""
    df_processed = df.copy()
    
    # Handle missing values
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'unknown')
        else:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    return df_processed

def main():
    st.markdown('<div class="main-header">💳 Credit Approval EDA Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")
    
    # Data loading section
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio("Choose data source:", ["Sample Data", "Upload CSV"])
    
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload your credit data CSV", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error loading file: {e}")
                df = load_sample_data()
        else:
            df = load_sample_data()
    else:
        df = load_sample_data()
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Sidebar filters
    st.sidebar.subheader("🔍 Data Filters")
    
    # Class filter
    if 'Class' in df_processed.columns:
        class_filter = st.sidebar.multiselect(
            "Filter by Class:", 
            options=df_processed['Class'].unique(),
            default=df_processed['Class'].unique()
        )
        df_filtered = df_processed[df_processed['Class'].isin(class_filter)]
    else:
        df_filtered = df_processed
    
    # Numerical columns filter
    numerical_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        selected_num_cols = st.sidebar.multiselect(
            "Select Numerical Columns:",
            options=numerical_cols,
            default=numerical_cols[:5] if len(numerical_cols) > 5 else numerical_cols
        )
    
    # Categorical columns filter
    categorical_cols = df_filtered.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        selected_cat_cols = st.sidebar.multiselect(
            "Select Categorical Columns:",
            options=categorical_cols,
            default=categorical_cols[:5] if len(categorical_cols) > 5 else categorical_cols
        )
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Overview", "📊 Distributions", "🔗 Correlations", "📉 Relationships", 
        "🎯 Target Analysis", "🔍 Dimensionality", "🤖 ML Insights", "📋 Data Quality"
    ])
    
    # TAB 1: OVERVIEW
    with tab1:
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df_filtered))
        with col2:
            st.metric("Features", len(df_filtered.columns))
        with col3:
            st.metric("Missing Values", df_filtered.isnull().sum().sum())
        with col4:
            if 'Class' in df_filtered.columns:
                approval_rate = (df_filtered['Class'] == '+').mean() * 100
                st.metric("Approval Rate", f"{approval_rate:.1f}%")
        
        # Data sample
        st.subheader("Data Sample")
        st.dataframe(df_filtered.head(10), use_container_width=True)
        
        # Data types
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Types")
            dtype_df = pd.DataFrame({
                'Column': df_filtered.columns,
                'Data Type': df_filtered.dtypes.astype(str),
                'Non-Null Count': df_filtered.count(),
                'Null Count': df_filtered.isnull().sum()
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.subheader("Statistical Summary")
            st.dataframe(df_filtered.describe(), use_container_width=True)
    
    # TAB 2: DISTRIBUTIONS
    with tab2:
        st.markdown('<div class="section-header">Feature Distributions</div>', unsafe_allow_html=True)
        
        # Numerical distributions
        if numerical_cols:
            st.subheader("Numerical Feature Distributions")
            
            col1, col2 = st.columns(2)
            with col1:
                plot_type = st.selectbox("Plot Type:", ["Histogram", "Box Plot", "Violin Plot"])
            with col2:
                bins = st.slider("Histogram Bins:", 10, 100, 30)
            
            for col in selected_num_cols:
                fig = go.Figure()
                
                if plot_type == "Histogram":
                    fig.add_trace(go.Histogram(
                        x=df_filtered[col],
                        nbinsx=bins,
                        name=col,
                        opacity=0.7
                    ))
                elif plot_type == "Box Plot":
                    fig.add_trace(go.Box(
                        y=df_filtered[col],
                        name=col,
                        boxpoints='outliers'
                    ))
                else:  # Violin Plot
                    fig.add_trace(go.Violin(
                        y=df_filtered[col],
                        name=col,
                        box_visible=True,
                        meanline_visible=True
                    ))
                
                fig.update_layout(
                    title=f"{plot_type} - {col}",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Categorical distributions
        if categorical_cols:
            st.subheader("Categorical Feature Distributions")
            
            for col in selected_cat_cols:
                value_counts = df_filtered[col].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Bar Chart - {col}",
                        labels={'x': col, 'y': 'Count'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Pie Chart - {col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: CORRELATIONS
    with tab3:
        st.markdown('<div class="section-header">Correlation Analysis</div>', unsafe_allow_html=True)
        
        if len(numerical_cols) > 1:
            # Correlation matrix
            corr_matrix = df_filtered[numerical_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix",
                color_continuous_scale='RdBu'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Strong correlations
            st.subheader("Strong Correlations (|r| > 0.5)")
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        strong_corr.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': corr_val
                        })
            
            if strong_corr:
                st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
            else:
                st.info("No strong correlations found.")
    
    # TAB 4: RELATIONSHIPS
    with tab4:
        st.markdown('<div class="section-header">Feature Relationships</div>', unsafe_allow_html=True)
        
        if len(numerical_cols) >= 2:
            # Scatter plot controls
            col1, col2, col3 = st.columns(3)
            with col1:
                x_var = st.selectbox("X Variable:", numerical_cols)
            with col2:
                y_var = st.selectbox("Y Variable:", numerical_cols, index=1 if len(numerical_cols) > 1 else 0)
            with col3:
                color_var = st.selectbox("Color By:", ['None'] + categorical_cols)
            
            # Create scatter plot
            if color_var != 'None':
                fig = px.scatter(
                    df_filtered,
                    x=x_var,
                    y=y_var,
                    color=color_var,
                    title=f"{x_var} vs {y_var}",
                    opacity=0.7,
                    hover_data=numerical_cols[:3]
                )
            else:
                fig = px.scatter(
                    df_filtered,
                    x=x_var,
                    y=y_var,
                    title=f"{x_var} vs {y_var}",
                    opacity=0.7
                )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Pair plot for selected features
            if len(selected_num_cols) >= 2:
                st.subheader("Pair Plot")
                pair_features = st.multiselect(
                    "Select features for pair plot:",
                    selected_num_cols,
                    default=selected_num_cols[:4]
                )
                
                if len(pair_features) >= 2:
                    fig = px.scatter_matrix(
                        df_filtered[pair_features],
                        title="Pair Plot",
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # TAB 5: TARGET ANALYSIS
    with tab5:
        st.markdown('<div class="section-header">Target Variable Analysis</div>', unsafe_allow_html=True)
        
        if 'Class' in df_filtered.columns:
            # Target distribution
            target_dist = df_filtered['Class'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    x=target_dist.index,
                    y=target_dist.values,
                    title="Target Distribution",
                    labels={'x': 'Class', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(
                    values=target_dist.values,
                    names=target_dist.index,
                    title="Target Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance by target
            if numerical_cols:
                st.subheader("Feature Distributions by Target")
                
                feature_to_analyze = st.selectbox(
                    "Select feature to analyze by target:",
                    numerical_cols
                )
                
                fig = px.box(
                    df_filtered,
                    x='Class',
                    y=feature_to_analyze,
                    title=f"{feature_to_analyze} by Target Class"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Categorical features by target
            if categorical_cols:
                st.subheader("Categorical Features by Target")
                
                cat_feature = st.selectbox(
                    "Select categorical feature:",
                    [col for col in categorical_cols if col != 'Class']
                )
                
                if cat_feature:
                    crosstab = pd.crosstab(df_filtered[cat_feature], df_filtered['Class'])
                    
                    fig = px.bar(
                        crosstab,
                        title=f"{cat_feature} by Target Class",
                        barmode='group'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # TAB 6: DIMENSIONALITY
    with tab6:
        st.markdown('<div class="section-header">Dimensionality Reduction</div>', unsafe_allow_html=True)
        
        if len(numerical_cols) >= 2:
            # Encode categorical variables for dimensionality reduction
            df_encoded = df_filtered.copy()
            le = LabelEncoder()
            
            for col in categorical_cols:
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            
            # PCA
            st.subheader("Principal Component Analysis (PCA)")
            
            n_components = st.slider("Number of Components:", 2, min(10, len(df_encoded.columns)-1), 2)
            
            pca = PCA(n_components=n_components)
            pca_result = pca.fit_transform(df_encoded.select_dtypes(include=[np.number]))
            
            # Explained variance
            fig = px.bar(
                x=range(1, n_components+1),
                y=pca.explained_variance_ratio_,
                title="Explained Variance Ratio by Component",
                labels={'x': 'Component', 'y': 'Explained Variance Ratio'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # PCA scatter plot
            pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
            if 'Class' in df_filtered.columns:
                pca_df['Class'] = df_filtered['Class'].values
                
                fig = px.scatter(
                    pca_df,
                    x='PC1',
                    y='PC2',
                    color='Class',
                    title="PCA - First Two Components"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # t-SNE (for datasets with reasonable size)
            if len(df_encoded) <= 1000:
                st.subheader("t-SNE Visualization")
                
                tsne = TSNE(n_components=2, random_state=42)
                tsne_result = tsne.fit_transform(df_encoded.select_dtypes(include=[np.number]))
                
                tsne_df = pd.DataFrame(tsne_result, columns=['t-SNE1', 't-SNE2'])
                if 'Class' in df_filtered.columns:
                    tsne_df['Class'] = df_filtered['Class'].values
                    
                    fig = px.scatter(
                        tsne_df,
                        x='t-SNE1',
                        y='t-SNE2',
                        color='Class',
                        title="t-SNE Visualization"
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # TAB 7: ML INSIGHTS
    with tab7:
        st.markdown('<div class="section-header">Machine Learning Insights</div>', unsafe_allow_html=True)
        
        if 'Class' in df_filtered.columns and len(numerical_cols) > 0:
            # Prepare data for ML
            df_ml = df_filtered.copy()
            
            # Encode categorical variables
            le = LabelEncoder()
            for col in categorical_cols:
                if col != 'Class':
                    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            
            # Encode target
            df_ml['Class'] = le.fit_transform(df_ml['Class'])
            
            # Feature importance using Random Forest
            st.subheader("Feature Importance")
            
            X = df_ml.drop('Class', axis=1)
            y = df_ml['Class']
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance (Random Forest)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Model performance
            st.subheader("Model Performance")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            rf.fit(X_train, y_train)
            
            train_score = rf.score(X_train, y_train)
            test_score = rf.score(X_test, y_test)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", f"{train_score:.3f}")
            with col2:
                st.metric("Test Accuracy", f"{test_score:.3f}")
            
            # Confusion matrix
            y_pred = rf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix",
                labels=dict(x="Predicted", y="Actual")
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Clustering analysis
            st.subheader("Clustering Analysis")
            
            n_clusters = st.slider("Number of Clusters:", 2, 10, 3)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X)
            
            cluster_df = pd.DataFrame(X.iloc[:, :2])
            cluster_df['Cluster'] = clusters
            
            fig = px.scatter(
                cluster_df,
                x=cluster_df.columns[0],
                y=cluster_df.columns[1],
                color='Cluster',
                title=f"K-Means Clustering (k={n_clusters})"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 8: DATA QUALITY
with tab8:
    st.markdown('<div class="section-header">Data Quality Assessment</div>', unsafe_allow_html=True)

    # Missing values analysis
    st.subheader("Missing Values Analysis")

    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100

    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': missing_percent.values
    }).sort_values('Missing Count', ascending=False)

    # Create bar chart for missing values
    fig = px.bar(
        missing_df,
        x='Column',
        y='Missing Percentage',
        title="Missing Values by Column"
    )
    
    # Safely update the x-axis tick angle
    fig.update_layout(
        xaxis_tickangle=45
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(missing_df, use_container_width=True)

if __name__ == "__main__":
    main()

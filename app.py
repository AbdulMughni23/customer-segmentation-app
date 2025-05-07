import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import io
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from ucimlrepo import fetch_ucirepo 

# Set page configuration
st.set_page_config(
    page_title="Customer Segmentation Tool",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define app title and description
st.title("🛍️ Customer Segmentation Analysis Tool")
st.markdown('''
This app performs RFM analysis and K-Means clustering on retail transaction data.
Upload your own data or use the sample Online Retail dataset.
''')

# Sidebar for navigation and options
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Upload", "RFM Analysis", "Clustering", "Marketing Strategy"])


  



# Function to load sample data
@st.cache_data
def load_sample_data():
    try:
        # fetch dataset 
        online_retail = fetch_ucirepo(id=352) 
        df = online_retail.data.original
        return df
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        # Return a small sample dataset for demonstration
        data = {
            'InvoiceNo': ['536365', '536366', '536367', '536368', '536369'],
            'StockCode': ['85123A', '71053', '84406B', '84029G', '84029E'],
            'Description': ['WHITE HANGING HEART T-LIGHT HOLDER', 'WHITE METAL LANTERN', 'CREAM CUPID HEARTS COAT HANGER', 'KNITTED UNION FLAG HOT WATER BOTTLE', 'RED WOOLLY HOTTIE WHITE HEART'],
            'Quantity': [6, 6, 8, 6, 6],
            'InvoiceDate': ['2010-12-01 08:26:00', '2010-12-01 08:28:00', '2010-12-01 08:34:00', '2010-12-01 08:34:00', '2010-12-01 08:34:00'],
            'UnitPrice': [2.55, 3.39, 2.75, 3.39, 3.39],
            'CustomerID': [17850, 17850, 17850, 17850, 17850],
            'Country': ['United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom']
        }
        df = pd.DataFrame(data)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        return df

# Function to clean the data
def clean_data(df):
    # Make a copy of the original dataset
    df_clean = df.copy()
    
    # Convert InvoiceDate to datetime if not already
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    
    # Remove rows with missing CustomerID
    df_clean = df_clean.dropna(subset=['CustomerID'])
    
    # Convert CustomerID to integer type
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)
    
    # Filter out canceled transactions (where InvoiceNo starts with 'C')
    df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
    
    # Keep only transactions with positive quantity and unit price
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
    
    # Calculate total amount spent per transaction
    df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']
    
    return df_clean

# Function to calculate RFM features
def calculate_rfm(df, country_filter=None):
    # Filter by country if specified
    if country_filter:
        df = df[df['Country'] == country_filter]
    
    # Set the snapshot date as one day after the last transaction date
    snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)
    
    # Calculate Recency, Frequency, and Monetary value for each customer
    # Recency: days since last purchase
    recency_df = df.groupby('CustomerID')['InvoiceDate'].max().reset_index()
    recency_df.columns = ['CustomerID', 'LastPurchaseDate']
    recency_df['Recency'] = (snapshot_date - recency_df['LastPurchaseDate']).dt.days
    
    # Frequency: number of invoices (transactions)
    frequency_df = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    frequency_df.columns = ['CustomerID', 'Frequency']
    
    # Monetary: total revenue
    monetary_df = df.groupby('CustomerID')['TotalAmount'].sum().reset_index()
    monetary_df.columns = ['CustomerID', 'Monetary']
    
    # Merge all RFM features into a single DataFrame
    rfm_df = recency_df.merge(frequency_df, on='CustomerID').merge(monetary_df, on='CustomerID')
    
    return rfm_df, snapshot_date

# Function to perform K-Means clustering
def perform_clustering(rfm_df, n_clusters=5):
    # Create a copy of the RFM DataFrame for transformation
    rfm_transformed = rfm_df.copy()
    
    # Apply log transformation to Monetary value
    rfm_transformed['Monetary_Log'] = np.log1p(rfm_transformed['Monetary'])
    
    # Apply log transformation to Frequency if highly skewed
    if rfm_transformed['Frequency'].skew() > 1:
        rfm_transformed['Frequency_Log'] = np.log1p(rfm_transformed['Frequency'])
        rfm_features = rfm_transformed[['Recency', 'Frequency_Log', 'Monetary_Log']]
    else:
        rfm_features = rfm_transformed[['Recency', 'Frequency', 'Monetary_Log']]
    
    # Standardize RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_features)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(rfm_scaled)
    
    # Add cluster labels to the original RFM DataFrame
    rfm_df['Cluster'] = cluster_labels
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(rfm_scaled)
    
    return rfm_df, kmeans, scaler, pca, pca_result, rfm_features.columns.tolist()

# Function to name clusters
def name_clusters(rfm_df_with_clusters):
    # Compute the mean of RFM features per cluster
    cluster_profiles = rfm_df_with_clusters.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    
    thresholds = {
    'recency_low': cluster_profiles['Recency'].quantile(0.25),  # Lower = better
    'frequency_high': cluster_profiles['Frequency'].quantile(0.75),
    'monetary_high': cluster_profiles['Monetary'].quantile(0.75)
}

    # Initialize cluster names dictionary
    cluster_names = {}
    for cluster in cluster_profiles.index:
        recency = cluster_profiles.loc[cluster, 'Recency']
        frequency = cluster_profiles.loc[cluster, 'Frequency']
        monetary = cluster_profiles.loc[cluster, 'Monetary']

        # Priority order: Most valuable segments first!
        if (recency <= thresholds['recency_low']) and \
           (frequency >= thresholds['frequency_high']) and \
           (monetary >= thresholds['monetary_high']):
            cluster_names[cluster] = "Elite Customers"
        elif (frequency >= thresholds['frequency_high']) and \
             (monetary >= thresholds['monetary_high']):
            cluster_names[cluster] = "Loyal High-Spenders"
        elif (recency <= thresholds['recency_low']):
            cluster_names[cluster] = "Recently Active"
        elif (recency > cluster_profiles['Recency'].quantile(0.9)):
            cluster_names[cluster] = "Churned Customers"
        else:
            cluster_names[cluster] = f"Segment {cluster} (Neutral)"
    
    return cluster_profiles, cluster_names

# Function to generate marketing strategy
def generate_marketing_strategy(rfm_df_with_clusters, cluster_profiles, cluster_names):
    # Define marketing actions for each segment
    marketing_actions = {
        'High-Value Champions': "Loyalty rewards, exclusive previews, and VIP experiences",
        'Loyal Customers': "Personalized product recommendations and membership benefits",
        'Potential Loyalists': "Targeted discounts on previously purchased categories",
        'At Risk Customers': "Re-engagement campaigns with special offers",
        "Can't Lose Them": "Tailored communication highlighting new products"
    }
    
    # Create a DataFrame for marketing strategy
    marketing_strategy = pd.DataFrame({
        'Cluster': list(cluster_names.keys()),
        'Segment_Name': list(cluster_names.values()),
        'Customer_Count': rfm_df_with_clusters['Cluster'].value_counts().sort_index().values,
        'Avg_Monetary': cluster_profiles['Monetary'].values.round(2),
        'Current_Revenue': rfm_df_with_clusters.groupby('Cluster')['Monetary'].sum().sort_index().values.round(2)
    })
    
    # Map segment names to their marketing actions
    marketing_strategy['Marketing_Action'] = marketing_strategy['Segment_Name'].map(
        lambda x: marketing_actions.get(x, "Standard promotion"))
    
    # Simulate a 5% conversion uplift per target segment
    marketing_strategy['Conv_Uplift'] = 0.05  # 5% uplift
    marketing_strategy['Revenue_Uplift'] = (marketing_strategy['Current_Revenue'] * 
                                         marketing_strategy['Conv_Uplift']).round(2)
    marketing_strategy['Expected_Revenue'] = (marketing_strategy['Current_Revenue'] + 
                                           marketing_strategy['Revenue_Uplift']).round(2)
    
    # Calculate ROI improvement
    total_current_revenue = marketing_strategy['Current_Revenue'].sum()
    total_expected_revenue = marketing_strategy['Expected_Revenue'].sum()
    roi_improvement = (total_expected_revenue - total_current_revenue) / total_current_revenue * 100
    
    return marketing_strategy, roi_improvement

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'clean_data' not in st.session_state:
    st.session_state.clean_data = None
if 'rfm_data' not in st.session_state:
    st.session_state.rfm_data = None
if 'clustered_data' not in st.session_state:
    st.session_state.clustered_data = None

# Home page
if page == "Home":
    st.header("Welcome to the Customer Segmentation Tool!")
    
    col1, col2 = st.columns(2)

    # -------------------------------------------------------------------------------------------------------------------------------------------------------
    
    with col1:
        st.markdown("""
            ### How It Works  
            1. **Upload** your retail transaction data  
            2. **Clean** the data automatically  
            3. **Generate** RFM analysis  
            4. **Create** customer segments  
            5. **Develop** targeted marketing strategies  

            This tool uses K-Means clustering on RFM (Recency, Frequency, Monetary)  
            features to identify meaningful customer segments.
        """)

    with col2:
        st.image(
            "https://www.omniconvert.com/blog/wp-content/uploads/2020/11/RFM-analysis-dashboard-1024x576.png",
            caption="RFM Customer Segmentation Dashboard",
            use_column_width=True
        )


    st.markdown("""
        ### Benefits  
        - **Better understand** your customer base  
        - **Identify** high-value customers  
        - **Discover** at-risk customers  
        - **Create** targeted marketing campaigns  
        - **Improve** customer retention  
        - **Increase** customer lifetime value  
    """)


    # -------------------------------------------------------------------------------------------------------------------------------------------------------
    
    if st.button("Start with Sample Data"):
        with st.spinner("Loading sample data..."):
            st.session_state.data = load_sample_data()
            st.session_state.clean_data = clean_data(st.session_state.data)
            st.success("Sample data loaded successfully! Navigate to 'Data Upload' to see the data.")

# Data Upload page
elif page == "Data Upload":
    st.header("Data Upload & Inspection")
    
    upload_method = st.radio("Select data source:", ["Upload Excel/CSV File", "Use Sample Data"])
    
    if upload_method == "Upload Excel/CSV File":
        uploaded_file = st.file_uploader("Upload your retail transaction data (Excel or CSV)", 
                                         type=["xlsx", "csv"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.data = df
                st.success(f"File uploaded successfully! {df.shape[0]} rows and {df.shape[1]} columns.")
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
    else:
        if st.button("Load Sample Data"):
            with st.spinner("Loading sample data..."):
                st.session_state.data = load_sample_data()
                st.success("Sample data loaded successfully!")
    
    if st.session_state.data is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data.head())
        
        st.subheader("Data Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Number of Rows:** {st.session_state.data.shape[0]}")
            st.write(f"**Number of Columns:** {st.session_state.data.shape[1]}")
            st.write(f"**Date Range:** {st.session_state.data['InvoiceDate'].min():%Y-%m-%d} to {st.session_state.data['InvoiceDate'].max():%Y-%m-%d}")
        
        with col2:
            st.write(f"**Countries:** {len(st.session_state.data['Country'].unique())}")
            st.write(f"**Customers:** {st.session_state.data['CustomerID'].nunique()}")
            st.write(f"**Transactions:** {st.session_state.data['InvoiceNo'].nunique()}")
        
        if st.button("Clean Data"):
            with st.spinner("Cleaning data..."):
                st.session_state.clean_data = clean_data(st.session_state.data)
                st.success(f"Data cleaned successfully! {st.session_state.clean_data.shape[0]} rows remaining after cleaning.")
        
        if st.session_state.clean_data is not None:
            st.subheader("Cleaned Data Preview")
            st.dataframe(st.session_state.clean_data.head())
            
            country_options = ["All"] + list(st.session_state.clean_data['Country'].unique())
            selected_country = st.selectbox("Filter by Country:", country_options)
            
            if selected_country != "All":
                filtered_data = st.session_state.clean_data[st.session_state.clean_data['Country'] == selected_country]
            else:
                filtered_data = st.session_state.clean_data
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Plot number of transactions over time
                transactions_by_date = filtered_data.groupby(filtered_data['InvoiceDate'].dt.date).size().reset_index()
                transactions_by_date.columns = ['Date', 'Number of Transactions']
                
                fig = px.line(transactions_by_date, x='Date', y='Number of Transactions', 
                             title='Number of Transactions Over Time')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Plot revenue over time
                revenue_by_date = filtered_data.groupby(filtered_data['InvoiceDate'].dt.date)['TotalAmount'].sum().reset_index()
                revenue_by_date.columns = ['Date', 'Revenue']
                
                fig = px.line(revenue_by_date, x='Date', y='Revenue', 
                             title='Revenue Over Time')
                st.plotly_chart(fig, use_container_width=True)

# RFM Analysis page
elif page == "RFM Analysis":
    st.header("RFM Analysis")
    
    if st.session_state.clean_data is None:
        st.warning("Please upload and clean your data first!")
        st.stop()
    
    country_options = ["All"] + list(st.session_state.clean_data['Country'].unique())
    selected_country = st.selectbox("Filter by Country:", country_options)
    
    if st.button("Generate RFM Analysis"):
        with st.spinner("Calculating RFM metrics..."):
            if selected_country != "All":
                rfm_df, snapshot_date = calculate_rfm(st.session_state.clean_data, selected_country)
            else:
                rfm_df, snapshot_date = calculate_rfm(st.session_state.clean_data)
            
            st.session_state.rfm_data = rfm_df
            st.session_state.snapshot_date = snapshot_date
            st.success("RFM analysis completed successfully!")
    
    if st.session_state.rfm_data is not None:
        st.subheader("RFM Data Preview")
        st.dataframe(st.session_state.rfm_data.head())
        
        st.write(f"**Snapshot Date for RFM Calculation:** {st.session_state.snapshot_date:%Y-%m-%d}")
        st.write(f"**Number of Customers in RFM Analysis:** {len(st.session_state.rfm_data)}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Recency distribution
            fig = px.histogram(st.session_state.rfm_data, x='Recency', 
                              title='Recency Distribution', 
                              labels={'Recency': 'Days Since Last Purchase'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Frequency distribution
            fig = px.histogram(st.session_state.rfm_data, x='Frequency', 
                              title='Frequency Distribution',
                              labels={'Frequency': 'Number of Purchases'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Monetary distribution
            fig = px.histogram(st.session_state.rfm_data, x='Monetary', 
                              title='Monetary Distribution',
                              labels={'Monetary': 'Total Spending'})
            st.plotly_chart(fig, use_container_width=True)
        
        # RFM correlation heatmap
        st.subheader("RFM Correlation Heatmap")
        corr = st.session_state.rfm_data[['Recency', 'Frequency', 'Monetary']].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis',
                       title='RFM Feature Correlation')
        st.plotly_chart(fig, use_container_width=True)

# Clustering page
elif page == "Clustering":
    st.header("Customer Segmentation with K-Means Clustering")
    
    if st.session_state.rfm_data is None:
        st.warning("Please complete the RFM analysis first!")
        st.stop()
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_clusters = st.slider("Number of Clusters (k)", min_value=2, max_value=10, value=5)
    
    with col2:
        if st.button("Perform Clustering"):
            with st.spinner("Performing K-Means clustering..."):
                rfm_df_with_clusters, kmeans_model, scaler, pca, pca_result, feature_names = perform_clustering(
                    st.session_state.rfm_data, n_clusters)
                cluster_profiles, cluster_names = name_clusters(rfm_df_with_clusters)
                
                st.session_state.clustered_data = rfm_df_with_clusters
                st.session_state.kmeans_model = kmeans_model
                st.session_state.cluster_profiles = cluster_profiles
                st.session_state.cluster_names = cluster_names
                st.session_state.pca_result = pca_result
                st.session_state.feature_names = feature_names
                
                st.success("Clustering completed successfully!")
    
    if st.session_state.clustered_data is not None:
        st.subheader("Customer Segments")
        
        # Create a DataFrame for PCA visualization
        pca_df = pd.DataFrame(data=st.session_state.pca_result, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = st.session_state.clustered_data['Cluster']
        
        # Map cluster numbers to cluster names
        pca_df['Segment'] = pca_df['Cluster'].map(st.session_state.cluster_names)
        
        # Create a 2D scatter plot with plotly
        fig = px.scatter(
            pca_df, x='PC1', y='PC2', color='Segment',
            title='Customer Segments Visualization (PCA)',
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        
        # Add cluster centroids
        centroids_pca = PCA(n_components=2).fit_transform(st.session_state.kmeans_model.cluster_centers_)
        for i, centroid in enumerate(centroids_pca):
            fig.add_trace(go.Scatter(
                x=[centroid[0]], 
                y=[centroid[1]],
                mode='markers',
                marker=dict(symbol='x', size=15, color='black', line=dict(width=2)),
                name=f'Centroid {i}'
            ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display cluster profiles
        st.subheader("Cluster Profiles (Average Values)")
        
        # Create a copy of cluster profiles with segment names
        profile_df = st.session_state.cluster_profiles.copy()
        profile_df['Segment Name'] = [st.session_state.cluster_names[i] for i in profile_df.index]
        profile_df = profile_df[['Segment Name', 'Recency', 'Frequency', 'Monetary']]
        
        st.dataframe(profile_df.round(2))
        
        # Radar chart for cluster profiles
        st.subheader("Cluster Profile Comparison")
        
        # Scale the cluster profiles for better visualization (0-100 range)
        profile_scaled = st.session_state.cluster_profiles.copy()
        min_max_scaler = lambda x: (x - x.min()) / (x.max() - x.min()) * 100
        profile_scaled = profile_scaled.transform(min_max_scaler)
        
        # Create radar chart with plotly
        categories = ['Recency', 'Frequency', 'Monetary']
        fig = go.Figure()
        
        for i in profile_scaled.index:
            values = profile_scaled.loc[i].tolist()
            values.append(values[0])  # Close the loop
            
            segment_name = st.session_state.cluster_names[i]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],  # Close the loop
                fill='toself',
                name=segment_name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Cluster Profile Comparison (Radar Chart)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display customer distribution across clusters
        st.subheader("Customer Distribution Across Segments")
        
        cluster_counts = st.session_state.clustered_data['Cluster'].value_counts().sort_index()
        cluster_names_list = [st.session_state.cluster_names[i] for i in cluster_counts.index]
        
        fig = px.pie(
            values=cluster_counts.values,
            names=cluster_names_list,
            title='Customer Distribution by Segment'
        )
        st.plotly_chart(fig, use_container_width=True)

# Marketing Strategy page
elif page == "Marketing Strategy":
    st.header("Targeted Marketing Strategy")
    
    if st.session_state.clustered_data is None or st.session_state.cluster_profiles is None:
        st.warning("Please perform clustering first!")
        st.stop()
    
    # Generate marketing strategy
    marketing_strategy, roi_improvement = generate_marketing_strategy(
        st.session_state.clustered_data,
        st.session_state.cluster_profiles,
        st.session_state.cluster_names
    )
    
    st.subheader("Customer Segment Profiles and Recommended Actions")
    
    # Display marketing strategy table
    st.dataframe(marketing_strategy[['Segment_Name', 'Customer_Count', 'Avg_Monetary', 'Marketing_Action']])
    
    # Revenue contribution by segment
    st.subheader("Revenue Contribution by Segment")
    
    fig = px.pie(
        marketing_strategy, 
        values='Current_Revenue', 
        names='Segment_Name',
        title='Revenue Contribution by Customer Segment'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Revenue uplift visualization
    st.subheader("Simulated Revenue Uplift by Segment")
    
    fig = px.bar(
        marketing_strategy,
        x='Segment_Name',
        y='Revenue_Uplift',
        color='Segment_Name',
        title='Projected Revenue Uplift by Segment (5% Conversion Improvement)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ROI improvement
    st.metric(
        label="Projected ROI Uplift",
        value=f"{roi_improvement:.2f}%",
        delta="Positive Impact"
    )
    
    # Detailed marketing recommendations
    st.subheader("Detailed Marketing Recommendations")
    
    for _, row in marketing_strategy.iterrows():
        with st.expander(f"{row['Segment_Name']} ({row['Customer_Count']} customers)"):
            st.write(f"**Average Spend:** £{row['Avg_Monetary']:.2f}")
            st.write(f"**Current Revenue:** £{row['Current_Revenue']:.2f}")
            st.write(f"**Marketing Action:** {row['Marketing_Action']}")
            st.write(f"**Projected Revenue Uplift:** £{row['Revenue_Uplift']:.2f} (with 5% conversion improvement)")
            
            # Add segment-specific marketing tactics
            st.write("**Specific Tactics:**")
            
            if "High-Value" in row['Segment_Name']:
                st.write("- Implement a VIP loyalty program with exclusive benefits")
                st.write("- Offer early access to new products")
                st.write("- Provide personal shopping assistance")
                st.write("- Send birthday/anniversary special offers")
            
            elif "Loyal" in row['Segment_Name']:
                st.write("- Create a tiered loyalty program")
                st.write("- Send personalized product recommendations based on purchase history")
                st.write("- Implement a referral program with incentives")
                st.write("- Offer bundle discounts on frequently purchased items")
            
            elif "Potential" in row['Segment_Name']:
                st.write("- Send targeted promotions for products similar to past purchases")
                st.write("- Offer first-time category discounts")
                st.write("- Create educational content about product benefits")
                st.write("- Implement a 'second purchase' discount strategy")
            
            elif "At Risk" in row['Segment_Name']:
                st.write("- Send 'We miss you' emails with special offers")
                st.write("- Implement a win-back campaign with significant discounts")
                st.write("- Request feedback to address potential issues")
                st.write("- Create limited-time offers to create urgency")
            
            elif "Can't Lose" in row['Segment_Name']:
                st.write("- Highlight new products in categories they've purchased from")
                st.write("- Create 'back in stock' notifications for previously viewed items")
                st.write("- Offer loyalty points boosts for purchases in the next 30 days")
                st.write("- Send personalized recommendations based on browsing history")
            
            else:
                st.write("- Implement standard marketing promotions")
                st.write("- Focus on building brand awareness")
                st.write("- Offer general discounts on popular products")

    # Call to action
    st.subheader("Implementation Roadmap")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Month 1:** Set up customer segments in marketing systems")
    
    with col2:
        st.info("**Month 2:** Launch targeted campaigns for top 2 segments")
    
    with col3:
        st.info("**Month 3:** Scale to all segments and measure performance")
    
    st.success(f"Implementing these targeted marketing strategies could result in a {roi_improvement:.2f}% ROI improvement!")

# Footer
st.markdown("---")
st.markdown("Customer Segmentation Tool | Built with Streamlit")
st.markdown("© 2025 Mohammad Abdul Mughni")
st.markdown("This app is for educational purposes only. Please ensure compliance with data privacy regulations when using real customer data.")


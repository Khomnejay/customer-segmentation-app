import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns

# Config
st.set_page_config(page_title="Customer Segmentation App", layout="wide")

# Load and preprocess data
@st.cache_data
def load_and_process_data():
    df = pd.read_csv("online_retail.csv", encoding='ISO-8859-1')
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].apply(lambda x: np.log1p(x))
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(rfm_scaled)
    rfm['PCA1'] = pca_components[:, 0]
    rfm['PCA2'] = pca_components[:, 1]
    
    return rfm

# Load data
rfm = load_and_process_data()

# Sidebar filter
st.sidebar.header("🔍 Filter Clusters")
selected_clusters = st.sidebar.multiselect("Select Clusters to View", sorted(rfm['Cluster'].unique()), default=sorted(rfm['Cluster'].unique()))
filtered_data = rfm[rfm['Cluster'].isin(selected_clusters)]

# Main title
st.title("📊 Market Segmentation using Clustering")
st.write("This app segments customers based on RFM (Recency, Frequency, Monetary) analysis using KMeans clustering.")

# PCA Plot
st.subheader("🌀 Cluster Visualization with PCA")
fig1, ax1 = plt.subplots()
scatter = ax1.scatter(filtered_data['PCA1'], filtered_data['PCA2'], c=filtered_data['Cluster'], cmap='tab10')
ax1.set_xlabel("PCA 1")
ax1.set_ylabel("PCA 2")
ax1.set_title("Customer Clusters in PCA Space")
st.pyplot(fig1)

# RFM Mean by Cluster
st.subheader("📈 RFM Metrics by Cluster")

fig2, axes = plt.subplots(1, 3, figsize=(18, 4))

for i, metric in enumerate(['Recency', 'Frequency', 'Monetary']):
    sns.barplot(data=rfm, x='Cluster', y=metric, ax=axes[i], palette='Set2')
    axes[i].set_title(f'Average {metric} per Cluster')
    axes[i].set_xlabel("Cluster")
    axes[i].set_ylabel(metric)

st.pyplot(fig2)

# Display Data
st.subheader("📋 Clustered Customer Data")
st.dataframe(filtered_data.head(20))

# Download button
st.download_button(
    label="📥 Download Clustered Data as CSV",
    data=filtered_data.to_csv(index=False).encode('utf-8'),
    file_name='clustered_customers.csv',
    mime='text/csv'
)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Title and description
st.set_page_config(page_title="Mall Customer Clustering", layout="wide")
st.title("🛍️ Mall Customer Clustering with PCA & Streamlit")
st.markdown("Cluster mall customers using **KMeans** or **DBSCAN** and visualize with PCA!")

# Upload CSV or use default
uploaded_file = st.file_uploader("📤 Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("mall_customers.csv")

# Preprocess
df = df.drop(columns=["CustomerID"], errors='ignore')
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
df = df.drop_duplicates()

# Feature scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# PCA for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Sidebar - choose algorithm
st.sidebar.header("🧠 Clustering Settings")
algorithm = st.sidebar.selectbox("Choose algorithm", ["KMeans", "DBSCAN"])

if algorithm == "KMeans":
    k = st.sidebar.slider("Number of clusters (k)", 2, 10, 5)
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(scaled_data)
elif algorithm == "DBSCAN":
    eps = st.sidebar.slider("eps", 0.1, 5.0, 0.5)
    min_samples = st.sidebar.slider("min_samples", 2, 10, 5)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(scaled_data)

# Add cluster labels to DataFrame
df['Cluster'] = labels
pca_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])
pca_df["Cluster"] = labels

# Evaluation scores
if len(set(labels)) > 1 and -1 not in labels:
    silhouette = silhouette_score(scaled_data, labels)
    db_index = davies_bouldin_score(scaled_data, labels)
    st.success(f"✅ **Silhouette Score:** {silhouette:.2f}")
    st.success(f"✅ **Davies-Bouldin Index:** {db_index:.2f}")
else:
    st.warning("⚠️ Not enough clusters to calculate scores (try adjusting parameters)")

# Plot PCA Clusters
st.subheader("📊 PCA Cluster Visualization")
fig, ax = plt.subplots()
palette = sns.color_palette("Set2", len(np.unique(labels)))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Cluster", palette=palette, ax=ax)
ax.set_title("PCA Reduced Clusters")
st.pyplot(fig)

# Show cluster summary
st.subheader("📋 Cluster Summary")
if len(set(labels)) > 1:
    summary = df.groupby("Cluster").mean(numeric_only=True)
    st.dataframe(summary)
else:
    st.write("Not enough clusters to show summary.")

# Show raw data
with st.expander("🔍 Show Raw Data"):
    st.dataframe(df)

# Footer
st.markdown("---")
st.markdown("📌 Built by Fatima Azfar using Streamlit")

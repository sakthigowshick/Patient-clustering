import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA

# ---------------------------
# Load & Clean Dataset
# ---------------------------
data = pd.read_csv("patient_dataset.csv")

# Fill missing values
numeric_cols = ["plasma_glucose", "skin_thickness", "insulin"]
for col in numeric_cols:
    data[col].fillna(data[col].median(), inplace=True)

categorical_cols = ["gender", "residence_type"]
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Select numeric features only
X = data.select_dtypes(include=['float64', 'int64']).values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# Helper for DBSCAN
# ---------------------------
def get_dbscan_model(eps=0.5, min_samples=5):
    return DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)

# ---------------------------
# Prediction + Visualization
# ---------------------------
def predict_cluster(algorithm, k, eps, min_samples, *features):
    features_scaled = scaler.transform([features])

    if algorithm == "KMeans":
        model = KMeans(n_clusters=int(k), random_state=42).fit(X_scaled)
        cluster = model.predict(features_scaled)[0]
        labels = model.labels_
        title = f"KMeans (k={k})"
  
    elif algorithm == "Hierarchical":
        new_data = np.vstack([X_scaled, features_scaled])
        labels = AgglomerativeClustering(n_clusters=int(k)).fit_predict(new_data)
        cluster = labels[-1]
        title = f"Hierarchical (k={k})"

        # PCA on new_data
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(new_data)

        plt.figure(figsize=(6, 5))
        plt.scatter(X_pca[:-1, 0], X_pca[:-1, 1], c=labels[:-1], cmap="tab10", alpha=0.6)
        plt.scatter(X_pca[-1, 0], X_pca[-1, 1], c="red", marker="*", s=200, edgecolors="black", label="New Sample")
        plt.title(f"{title} — New sample → Cluster {cluster}")
        plt.legend()
    
        return f"✅ Belongs to Cluster {cluster} ({title})", plt

    elif algorithm == "DBSCAN":
        model = get_dbscan_model(eps=eps, min_samples=int(min_samples))
        if len(model.core_sample_indices_) == 0:
            return "⚠ DBSCAN found no clusters. Try changing eps/min_samples.", None
        
        dists = euclidean_distances(features_scaled, model.components_)
        nearest_idx = np.argmin(dists)
        nearest_dist = dists[0, nearest_idx]
        cluster = model.labels_[model.core_sample_indices_][nearest_idx]
        if nearest_dist > model.eps:
            return f"🚨 OUTLIER (noise, dist={nearest_dist:.2f})", None
        labels = model.labels_
        title = f"DBSCAN (eps={model.eps}, min_samples={min_samples})"

    # ---- Visualization (PCA 2D) ----
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    new_point = pca.transform(features_scaled)

    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", alpha=0.6)
    plt.scatter(new_point[:, 0], new_point[:, 1], c="red", marker="*", s=200, edgecolors="black", label="New Sample")
    plt.title(f"{title} — New sample → Cluster {cluster}")
    plt.legend()
    
    return f"✅ Belongs to Cluster {cluster} ({title})", plt

# ---------------------------
# Login Validation
# ---------------------------
def login(username, password):
    if username == "admin" and password == "1234":
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(value="❌ Invalid login. Try again."), gr.update(visible=False)

# ---------------------------
# Gradio UI
# ---------------------------
with gr.Blocks() as demo:
    # Page 1: Login
    with gr.Row(visible=True) as login_page:
        with gr.Column():
            gr.Markdown("## 🔑 Login to Access Patient Clustering App")
            username = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")
            login_msg = gr.Textbox(label="Login Status")

    # Page 2: Clustering App
    with gr.Row(visible=False) as app_page:
        with gr.Column():
            gr.Markdown("## 🏥 Patient Clustering App")

            algorithm = gr.Dropdown(["KMeans", "Hierarchical", "DBSCAN"], label="Select Algorithm")
            k_value = gr.Number(label="Number of Clusters (k for KMeans/Hierarchical)", value=3)
            eps_value = gr.Slider(label="DBSCAN eps", minimum=0.1, maximum=2.0, step=0.1, value=0.5)
            min_samples_value = gr.Number(label="DBSCAN min_samples", value=5)

            inputs = []
            with gr.Accordion("Enter Patient Feature Values", open=False):
                for col in data.select_dtypes(include=['float64', 'int64']).columns:
                    inputs.append(gr.Number(label=col, value=float(data[col].median())))

            btn = gr.Button("Find Cluster")
            output_text = gr.Textbox(label="Result")
            output_plot = gr.Plot()

    # Button Actions
    login_btn.click(fn=login, inputs=[username, password], outputs=[login_msg, app_page])
    btn.click(fn=predict_cluster, inputs=[algorithm, k_value, eps_value, min_samples_value] + inputs, outputs=[output_text, output_plot])

# ---------------------------
# Launch
# ---------------------------
if __name__ == "__main__":
    demo.launch()

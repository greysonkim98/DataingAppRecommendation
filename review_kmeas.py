import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import random
import os
import matplotlib.pyplot as plt

# -----------------------------
# Visualization Functions
# -----------------------------
def visualize_clusters(X, labels, k, app_name='App'):
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for i in range(k):
        cluster_points = X_2d[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i+1}', alpha=0.6)
    plt.title(f'{app_name} Review Clusters (PCA 2D)')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# -----------------------------
# Loss Functions
# -----------------------------
def compute_sse(X, labels, centroids):
    sse = 0
    for i, center in enumerate(centroids):
        cluster_points = X[labels == i]
        sse += np.sum((cluster_points - center) ** 2)
    return sse

# -----------------------------
# Extract Representitive Functions
# -----------------------------
def extract_cluster_centroids(df, k, text_col='content',
                               model_name='all-MiniLM-L6-v2',
                               max_iter=1000, tol=1e-10,
                               random_state=42,
                               app_name='App',
                               embedding_cache=None):

    random.seed(random_state)
    np.random.seed(random_state)
    df = df.copy()
    texts = df[text_col].astype(str).tolist()

    # Load embeddings is there is
    if embedding_cache and os.path.exists(embedding_cache):
        print(f"Loading embeddings from {embedding_cache} ...")
        X = np.load(embedding_cache)
    # Generate embeddings
    else:
        print(f"⚙️  Encoding sentences using {model_name} ...")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True)
        X = np.array(embeddings)
        if embedding_cache:
            np.save(embedding_cache, X)
            print(f"Saved embeddings to {embedding_cache}")

    # Perform Kmeans algorithm
    index = np.random.choice(len(X), k, replace=False)
    centroid = X[index]
    sse_history = []
    iteration_count = 0

    for _ in range(max_iter):
        distances = np.linalg.norm(X[:, None, :] - centroid[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroid[i]
            for i in range(k)
        ])
        shift = np.linalg.norm(centroid - new_centroids)
        centroid = new_centroids
        sse = compute_sse(X, labels, centroid)
        sse_history.append(sse)

        iteration_count += 1
        if shift < tol:
            break
    
    # Display kmeans algorithm performance
    print(f" Final Iteration: {iteration_count} | Final SSE: {sse:.2f}")
    print(" Cluster sizes:", np.bincount(labels))
    print(f"Shift after last iteration: {shift:.10f}")

    # SSE Plot
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(sse_history) + 1), sse_history, marker='o')
    plt.title("SSE over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("SSE (Sum of Squared Errors)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    visualize_clusters(X, labels, k, app_name)  
    
    

    # Extract representative sentences - centroids
    representative_texts = []
    for i in range(k):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) == 0:
            representative_texts.append("[Empty Cluster]")
            continue
        cluster_vectors = X[cluster_indices]
        center = centroid[i]
        distances = np.linalg.norm(cluster_vectors - center, axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]
        representative_texts.append(texts[closest_idx])

    return representative_texts

# -----------------------------
# Data Modification
# -----------------------------
def balance_reviews(df: pd.DataFrame, score_col='score', text_col='content',
                    min_length=200, random_state=42):
    # match the number of 1 star and 5 star reviews
    df = df.copy()
    df['review_length'] = df[text_col].astype(str).str.len()
    df_filtered = df[df['review_length'] >= min_length].copy()
    df_1 = df_filtered[df_filtered[score_col] == 1]
    df_5 = df_filtered[df_filtered[score_col] == 5]
    df_mid = df_filtered[df_filtered[score_col].isin([2, 3, 4])]
    n = min(len(df_1), len(df_5))
    df_1_sampled = df_1.sample(n=n, random_state=random_state)
    df_5_sampled = df_5.sample(n=n, random_state=random_state)
    df_balanced = pd.concat([df_1_sampled, df_5_sampled, df_mid], ignore_index=True)
    return df_balanced[[text_col, score_col, 'appVersion']].copy()

# -----------------------------
# Main Functions
# -----------------------------
def main():
    app_name = "bumble"
    file_path = "bumble_google_play_reviews.csv"
    embedding_path = f"{app_name}_embeddings.npy"
    k = 10

    # Load dataset
    df = pd.read_csv(file_path)
    balanced_df = balance_reviews(df)
    balanced_df.to_csv(f"{app_name}_balanced_reviews.csv", index=False)

    clusters = extract_cluster_centroids(
        balanced_df,
        k=k,
        app_name=app_name.capitalize(),
        embedding_cache=embedding_path
    )

    print(f"\n=== {app_name.upper()} CLUSTERS ===\n")
    for i, sent in enumerate(clusters):
        print(f"[Cluster {i+1}]\n{sent}\n{'-'*60}")

if __name__ == "__main__":
    main()

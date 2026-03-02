from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# Elbow Method
def elbow_method(skill_matrix, max_k=10, save_path="outputs/figures/elbow_method.png"):
    inertias = []

    for k in range(1, max_k + 1):
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        model.fit(skill_matrix)
        inertias.append(model.inertia_)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_k + 1), inertias, marker="o")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal K")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Elbow plot saved to: {save_path}")

# Clustering
def cluster_jobs(skill_matrix, k=4):
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(skill_matrix)

    return labels, model
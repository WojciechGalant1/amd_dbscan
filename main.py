"""
Skrypt porównawczy dla DBSCAN vs AMD-DBSCAN
Testuje oba algorytmy na zbiorach danych o pojedynczej i wielokrotnej gęstości
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
import time

from dbscan import DBSCAN
from amd_dbscan import AMD_DBSCAN


def generate_multi_density_data():
    """Generuje zbiór danych o wielokrotnej gęstości ze zmienną gęstością klastrów"""
    # Gęsty klaster
    X1, y1 = make_blobs(n_samples=300, centers=[(0, 0)], cluster_std=0.2, random_state=42)
    y1 = y1 * 0  # etykieta 0
    
    # Klaster o średniej gęstości
    X2, y2 = make_blobs(n_samples=300, centers=[(5, 5)], cluster_std=1.0, random_state=43)
    y2 = y2 * 0 + 1  # etykieta 1
    
    # Rzadki klaster
    X3, y3 = make_blobs(n_samples=200, centers=[(8, -2)], cluster_std=1.8, random_state=44)
    y3 = y3 * 0 + 2  # etykieta 2
    
    X = np.vstack([X1, X2, X3])
    y_true = np.hstack([y1, y2, y3])
    return X, y_true


def generate_single_density_data():
    """Generuje zbiór danych o pojedynczej gęstości"""
    X, y_true = make_blobs(n_samples=600, centers=3, cluster_std=0.6, random_state=42)
    return X, y_true


def evaluate_clustering(y_true, y_pred, labels_true_name="True", labels_pred_name="Predicted"):
    """Oblicza metryki ewaluacyjne"""
    # Usuwanie punktów szumu (-1) dla niektórych metryk
    mask = y_pred != -1
    if np.sum(mask) == 0:
        return {"nmi": 0.0, "ari": 0.0, "noise_ratio": 1.0, "n_clusters": 0}
    
    nmi = normalized_mutual_info_score(y_true[mask], y_pred[mask])
    ari = adjusted_rand_score(y_true[mask], y_pred[mask])
    noise_ratio = np.sum(y_pred == -1) / len(y_pred)
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    
    return {
        "nmi": nmi,
        "ari": ari,
        "noise_ratio": noise_ratio,
        "n_clusters": n_clusters
    }


def plot_comparison(X, y_true, dbscan_labels, amd_labels, title_prefix=""):
    """Rysuje porównanie wyników klasteryzacji"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Prawdziwe etykiety
    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="tab10", s=20, alpha=0.7)
    axes[0].set_title(f"{title_prefix}True Labels")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].grid(True, alpha=0.3)
    
    # Wyniki DBSCAN
    dbscan_eval = evaluate_clustering(y_true, dbscan_labels)
    axes[1].scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap="tab10", s=20, alpha=0.7)
    axes[1].set_title(f"{title_prefix}DBSCAN\n"
                      f"NMI: {dbscan_eval['nmi']:.3f}, "
                      f"Clusters: {dbscan_eval['n_clusters']}, "
                      f"Noise: {dbscan_eval['noise_ratio']:.1%}")
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")
    axes[1].grid(True, alpha=0.3)
    
    # Wyniki AMD-DBSCAN
    amd_eval = evaluate_clustering(y_true, amd_labels)
    axes[2].scatter(X[:, 0], X[:, 1], c=amd_labels, cmap="tab10", s=20, alpha=0.7)
    axes[2].set_title(f"{title_prefix}AMD-DBSCAN\n"
                      f"NMI: {amd_eval['nmi']:.3f}, "
                      f"Clusters: {amd_eval['n_clusters']}, "
                      f"Noise: {amd_eval['noise_ratio']:.1%}")
    axes[2].set_xlabel("Feature 1")
    axes[2].set_ylabel("Feature 2")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def compare_on_dataset(X, y_true, dataset_name="Dataset"):
    """Porównuje DBSCAN i AMD-DBSCAN na danym zbiorze danych"""
    print(f"\n{'='*60}")
    print(f"Testing on {dataset_name}")
    print(f"{'='*60}")
    print(f"Dataset shape: {X.shape}")
    print(f"True number of clusters: {len(set(y_true))}")
    
    # Test DBSCAN z ręcznie dostrojonymi parametrami
    # Dla sprawiedliwego porównania, wypróbowanie kilku kombinacji parametrów
    print("\n--- DBSCAN ---")
    best_dbscan_nmi = -1
    best_dbscan_labels = None
    best_dbscan_params = None
    
    # Wypróbowanie różnych wartości eps
    eps_candidates = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    min_pts = 5
    
    for eps in eps_candidates:
        model = DBSCAN(eps=eps, min_pts=min_pts)
        labels = model.fit_predict(X)
        eval_result = evaluate_clustering(y_true, labels)
        if eval_result["nmi"] > best_dbscan_nmi:
            best_dbscan_nmi = eval_result["nmi"]
            best_dbscan_labels = labels
            best_dbscan_params = (eps, min_pts)
    
    print(f"Best DBSCAN parameters: eps={best_dbscan_params[0]}, min_pts={best_dbscan_params[1]}")
    dbscan_eval = evaluate_clustering(y_true, best_dbscan_labels)
    print(f"  NMI: {dbscan_eval['nmi']:.4f}")
    print(f"  ARI: {dbscan_eval['ari']:.4f}")
    print(f"  Number of clusters: {dbscan_eval['n_clusters']}")
    print(f"  Noise ratio: {dbscan_eval['noise_ratio']:.2%}")
    
    # Test AMD-DBSCAN
    print("\n--- AMD-DBSCAN ---")
    start_time = time.time()
    model_amd = AMD_DBSCAN(verbose=False, max_k_search=50)
    amd_labels, amd_eval_dict = model_amd.fit_predict(X, y_true=y_true)
    amd_time = time.time() - start_time
    
    amd_eval = evaluate_clustering(y_true, amd_labels)
    print(f"  Execution time: {amd_time:.2f}s")
    print(f"  NMI: {amd_eval['nmi']:.4f}")
    print(f"  ARI: {amd_eval['ari']:.4f}")
    print(f"  Number of clusters: {amd_eval['n_clusters']}")
    print(f"  Noise ratio: {amd_eval['noise_ratio']:.2%}")
    print(f"  VNN (Variance of Neighbors): {model_amd.vnn_:.2f}")
    print(f"  Adaptive k: {model_amd.adaptive_k_}")
    print(f"  Candidate Eps: {[f'{e:.3f}' for e in model_amd.candidate_eps_]}")
    
    # Porównanie
    print("\n--- Comparison ---")
    nmi_improvement = amd_eval['nmi'] - dbscan_eval['nmi']
    print(f"NMI improvement: {nmi_improvement:+.4f} ({nmi_improvement/dbscan_eval['nmi']*100:+.1f}%)")
    
    return best_dbscan_labels, amd_labels, dbscan_eval, amd_eval


def main():
    """Główna funkcja porównawcza"""
    print("="*60)
    print("DBSCAN vs AMD-DBSCAN Comparison")
    print("="*60)
    
    # Test 1: Zbiór danych o wielokrotnej gęstości
    X_multi, y_multi = generate_multi_density_data()
    dbscan_labels_multi, amd_labels_multi, dbscan_eval_multi, amd_eval_multi = \
        compare_on_dataset(X_multi, y_multi, "Multi-Density Dataset")
    
    fig1 = plot_comparison(X_multi, y_multi, dbscan_labels_multi, amd_labels_multi, 
                          "Multi-Density: ")
    plt.savefig("comparison_multi_density.png", dpi=150, bbox_inches='tight')
    print("\nSaved plot: comparison_multi_density.png")
    
    # Test 2: Zbiór danych o pojedynczej gęstości
    X_single, y_single = generate_single_density_data()
    dbscan_labels_single, amd_labels_single, dbscan_eval_single, amd_eval_single = \
        compare_on_dataset(X_single, y_single, "Single-Density Dataset")
    
    fig2 = plot_comparison(X_single, y_single, dbscan_labels_single, amd_labels_single,
                          "Single-Density: ")
    plt.savefig("comparison_single_density.png", dpi=150, bbox_inches='tight')
    print("\nSaved plot: comparison_single_density.png")
    
    # Podsumowanie
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nMulti-Density Dataset:")
    print(f"  DBSCAN NMI: {dbscan_eval_multi['nmi']:.4f}")
    print(f"  AMD-DBSCAN NMI: {amd_eval_multi['nmi']:.4f}")
    print(f"  Improvement: {(amd_eval_multi['nmi'] - dbscan_eval_multi['nmi'])/dbscan_eval_multi['nmi']*100:+.1f}%")
    
    print("\nSingle-Density Dataset:")
    print(f"  DBSCAN NMI: {dbscan_eval_single['nmi']:.4f}")
    print(f"  AMD-DBSCAN NMI: {amd_eval_single['nmi']:.4f}")
    print(f"  Improvement: {(amd_eval_single['nmi'] - dbscan_eval_single['nmi'])/dbscan_eval_single['nmi']*100:+.1f}%")
    
    plt.show()


if __name__ == "__main__":
    main()

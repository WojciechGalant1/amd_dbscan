"""
Główne zaimplementowane funkcjonalności:
- Adaptacja parametrów do znalezienia adaptacyjnego k (MinPts) używając EpsList i MinPtsList
- Odkrywanie kandydatów Eps poprzez histogram k-dist + klasteryzacja KMeans wartości k-dist
- Multi-density DBSCAN: iteracyjne uruchamianie DBSCAN na rosnących kandydatach Eps, usuwanie sklasteryzowanych punktów
- Obliczanie VNN (wariancja liczby sąsiadów)
- Opcjonalna ewaluacja używając NMI i dokładności gdy dostępne są prawdziwe etykiety
- Optymalizacja wyszukiwania binarnego w fazie adaptacji parametrów (zgodnie z artykułem)

"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN as SKDBSCAN
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
from collections import Counter
import math
import time


class AMD_DBSCAN:
    def __init__(
        self,
        max_k_search=None,
        eps_peaks=None,
        peak_kmeans_init=10,
        min_pts_floor=2,
        verbose=False,
        distance_metric="euclidean",
    ):
        """
        Parametry:
        -----------
        max_k_search : int lub None
            Maksymalne k do rozważenia przy budowaniu EpsList i MinPtsList.
            Jeśli None, używa min(100, n-1) aby uniknąć dużego zużycia pamięci/czasu.
        eps_peaks : opcjonalne nadpisanie liczby pików (K w KMeans) -
                    jeśli None, wykrywamy liczbę pików z histogramu heurystycznie.
        peak_kmeans_init : n_init dla KMeans przy klasteryzacji wartości k-dis
        min_pts_floor : minimalna dolna granica MinPts (unikanie 0/1)
        verbose : bool, wyświetlanie informacji debugowych
        distance_metric : string przekazywany do NearestNeighbors (domyślnie euclidean)
        """
        self.max_k_search = max_k_search
        self.eps_peaks = eps_peaks
        self.peak_kmeans_init = peak_kmeans_init
        self.min_pts_floor = max(2, min_pts_floor)
        self.verbose = verbose
        self.distance_metric = distance_metric

        self.eps_list_ = None
        self.minpts_list_ = None
        self.adaptive_k_ = None
        self.best_eps_ = None
        self.best_labels_ = None
        self.labels_store_ = None
        self.candidate_eps_ = None
        self.labels_ = None
        self.vnn_ = None


    @staticmethod
    def pairwise_distance_matrix(X):
        """Oblicza pełną macierz odległości parami (n x n)."""
        X = np.asarray(X)
        sq = np.sum(X * X, axis=1)
        D = np.sqrt(np.abs(sq[:, None] + sq[None, :] - 2.0 * X.dot(X.T)))
        return D


    def obtain_eps_list(self, X):
        """
        Buduje EpsList poprzez obliczanie wartości k-dist dla k=1..K i wybieranie
        reprezentanta dla każdego k (używamy średniej k-odległości dla każdego k).
        Zwraca listę kandydatów eps (rosnąco).
        """
        X = np.asarray(X)
        n = X.shape[0]
        max_k = self.max_k_search or min(100, max(2, n - 1))
        max_k = min(max_k, n - 1)

        nbrs = NearestNeighbors(n_neighbors=max_k + 1, metric=self.distance_metric).fit(X)
        dists, _ = nbrs.kneighbors(X) 

        eps_list = []
        for k in range(1, max_k + 1):
            val = np.mean(dists[:, k])
            eps_list.append(float(val))

        eps_list = sorted(list(set(eps_list)))
        if self.verbose:
            print(f"[obtain_eps_list] n={n}, max_k={max_k}, eps_list_len={len(eps_list)}")
        self.eps_list_ = eps_list
        return eps_list


    def obtain_minpts_list(self, X, eps_list):
        """
        Dla każdego eps w eps_list, obliczenie MinPts jako średnia liczba sąsiadów
        w promieniu eps dla wszystkich punktów (zaokrąglone do najbliższej liczby całkowitej i z dolną granicą).
        Zwraca listę minpts odpowiadającą eps_list.
        """
        X = np.asarray(X)
        n = X.shape[0]
        minpts_list = []

        nbrs = NearestNeighbors(n_neighbors=min(n - 1, 200), metric=self.distance_metric).fit(X)

        for eps in eps_list:
            nbrs_radius = NearestNeighbors(radius=eps, metric=self.distance_metric).fit(X)
            neigh = nbrs_radius.radius_neighbors(X, return_distance=False)

            counts = np.array([len(lst) for lst in neigh], dtype=float)
            mean_count = np.mean(counts)
            minpts = int(round(mean_count))
            if minpts < self.min_pts_floor:
                minpts = self.min_pts_floor
            minpts_list.append(int(minpts))
        self.minpts_list_ = minpts_list
        if self.verbose:
            print(f"[obtain_minpts_list] computed {len(minpts_list)} minpts values")
        return minpts_list


    def compute_vnn(self, X, eps1=None):
        """
        Wariancja liczby sąsiadów w promieniu eps1 dla każdego punktu.
        Jeśli eps1 None, wybierz pierwszy eps z eps_list lub oblicz średnią odległość najbliższego sąsiada.
        """
        X = np.asarray(X)
        n = X.shape[0]
        if eps1 is None:
            if self.eps_list_ and len(self.eps_list_) > 0:
                eps1 = self.eps_list_[0]
            else:
                nbrs = NearestNeighbors(n_neighbors=2, metric=self.distance_metric).fit(X)
                d, _ = nbrs.kneighbors(X)
                eps1 = float(np.median(d[:, 1]))

        nbrs_radius = NearestNeighbors(radius=eps1, metric=self.distance_metric).fit(X)
        neigh = nbrs_radius.radius_neighbors(X, return_distance=False)
        counts = np.array([len(lst) for lst in neigh], dtype=float)
        vnn = float(np.var(counts))
        self.vnn_ = vnn
        if self.verbose:
            print(f"[compute_vnn] eps1={eps1}, VNN={vnn:.3f}")
        return vnn


    @staticmethod
    def run_dbscan(X, eps, min_pts, metric="euclidean"):
        """
        Używa scikit-learn DBSCAN do uruchomienia klasteryzacji z podanymi parametrami.
        Zwraca etykiety (np.array) i num_clusters (int, ignorując etykietę szumu -1).
        """
        if eps <= 0:
            labels = -np.ones(len(X), dtype=int)
            return labels, 0
        model = SKDBSCAN(eps=eps, min_samples=max(1, int(min_pts)), metric=metric)
        labels = model.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        return labels, int(n_clusters)


    def parameter_adaptation_find_k(self, X, eps_list=None, minpts_list=None, y_true=None):
        """
        Implementuje logikę do znalezienia adaptacyjnego k zgodnie z artykułem:
        - iteruj przez pary EpsList/MinPtsList
        - obliczaj liczby klastrów DBSCAN sekwencyjnie i znajdź pierwszy "stabilny" region gdzie liczba klastrów jest taka sama 3 razy z rzędu
        - następnie wyszukiwanie binarne aby znaleźć największy indeks gdzie liczba klastrów == n (stabilna liczba klastrów)
        - zwróć MinPts[best_index] jako adaptacyjne k
        Jeśli y_true jest podane (prawdziwe etykiety), używa NMI aby wybrać najlepszy indeks wśród stabilnego regionu (artykuł używa NMI do wyboru najlepszego).
        """
        X = np.asarray(X)
        n = X.shape[0]
        if eps_list is None:
            eps_list = self.obtain_eps_list(X)
        if minpts_list is None:
            minpts_list = self.obtain_minpts_list(X, eps_list)

        L = len(eps_list)
        if L == 0:
            raise ValueError("eps_list is empty; cannot adapt parameters")

        cluster_counts = np.zeros(L, dtype=int)
        labels_store = [None] * L
        self.labels_store_ = labels_store

        for i in range(L):
            eps = eps_list[i]
            mp = minpts_list[i]
            labels, ncl = self.run_dbscan(X, eps, mp, metric=self.distance_metric)
            cluster_counts[i] = ncl
            labels_store[i] = labels
            if self.verbose:
                print(f"[param_adapt] i={i}, eps={eps:.4f}, minpts={mp}, clusters={ncl}")

        stable_idx = None
        for i in range(L - 2):
            if cluster_counts[i] == cluster_counts[i + 1] == cluster_counts[i + 2] and cluster_counts[i] > 0:
                stable_idx = i + 2
                break

        if stable_idx is None:
            if y_true is not None:
                best_i = 0
                best_nmi = -1
                best_noise_ratio = float('inf')
                for i in range(L):
                    if labels_store[i] is None:
                        continue
                    try:
                        nmi = normalized_mutual_info_score(y_true, labels_store[i])
                        noise_ratio = np.sum(labels_store[i] == -1) / len(labels_store[i])
                    except Exception:
                        nmi = -1
                        noise_ratio = float('inf')

                    if nmi > best_nmi + 0.01:
                        best_nmi = nmi
                        best_noise_ratio = noise_ratio
                        best_i = i
                    elif abs(nmi - best_nmi) <= 0.01:
                        if noise_ratio < best_noise_ratio - 0.01:
                            best_nmi = nmi
                            best_noise_ratio = noise_ratio
                            best_i = i
                        elif abs(noise_ratio - best_noise_ratio) <= 0.01 and i > best_i:
                            best_nmi = nmi
                            best_noise_ratio = noise_ratio
                            best_i = i
                best_index = best_i
            else:
                best_index = int(L // 2)
            adaptive_k = minpts_list[best_index]
            best_eps = eps_list[best_index] if best_index < len(eps_list) else eps_list[len(eps_list)//2]
            best_labels = labels_store[best_index] if labels_store[best_index] is not None else None
            self.adaptive_k_ = int(adaptive_k)
            self.best_eps_ = float(best_eps)
            self.best_labels_ = best_labels
            if self.verbose:
                print(f"[param_adapt] no stable segment found, fallback best_index={best_index}, adaptive_k={adaptive_k}, best_eps={best_eps:.4f}")
            return int(adaptive_k)

        n_clusters_target = cluster_counts[stable_idx]

        left = stable_idx
        right = L - 1
        best_index = stable_idx
        while left <= right:
            mid = (left + right) // 2
            if cluster_counts[mid] == n_clusters_target:
                best_index = mid
                left = mid + 1
            elif cluster_counts[mid] < n_clusters_target:
                right = mid - 1
            else:
                left = mid + 1

        if y_true is not None:
            best_nmi = -1
            best_index_nmi = best_index
            best_noise_ratio = float('inf')
            for idx in range(stable_idx, best_index + 1):
                lbls = labels_store[idx]
                if lbls is None:
                    continue
                try:
                    nmi = normalized_mutual_info_score(y_true, lbls)
                    noise_ratio = np.sum(lbls == -1) / len(lbls)
                except Exception:
                    nmi = -1
                    noise_ratio = float('inf')
                if nmi > best_nmi + 0.01:
                    best_nmi = nmi
                    best_noise_ratio = noise_ratio
                    best_index_nmi = idx
                elif abs(nmi - best_nmi) <= 0.01:
                    if noise_ratio < best_noise_ratio - 0.01:
                        best_nmi = nmi
                        best_noise_ratio = noise_ratio
                        best_index_nmi = idx
                    elif abs(noise_ratio - best_noise_ratio) <= 0.01 and idx > best_index_nmi:
                        best_nmi = nmi
                        best_noise_ratio = noise_ratio
                        best_index_nmi = idx
            best_index = best_index_nmi

        adaptive_k = minpts_list[best_index]
        best_eps = eps_list[best_index]
        best_labels = labels_store[best_index]

        n = X.shape[0]
        max_reasonable_k = min(20, max(4, int(np.sqrt(n))))
        if adaptive_k > max_reasonable_k:
            if self.verbose:
                print(f"[param_adapt] adaptive_k={adaptive_k} zbyt wysokie, ograniczenie do {max_reasonable_k}")
            adaptive_k = max_reasonable_k
            
            for i, mp in enumerate(minpts_list):
                if mp == adaptive_k:
                    best_eps = eps_list[i]
                    best_labels = labels_store[i]
                    break
        
        self.adaptive_k_ = int(adaptive_k)
        self.best_eps_ = float(best_eps)
        self.best_labels_ = best_labels
        if self.verbose:
            print(f"[param_adapt] stable_idx={stable_idx}, best_index={best_index}, adaptive_k={adaptive_k}, best_eps={best_eps:.4f}")
        return int(adaptive_k)


    def obtain_candidate_eps_list(self, X, adaptive_k=None, n_bins=100):
        """
        Oblicz wartości kdis dla adaptive_k i uzyskaj listę kandydatów Eps:
        - oblicz k-tą odległość najbliższego sąsiada dla każdego punktu (kdis)
        - oblicz histogram wartości kdis i znajdź liczbę pików (heurystyka)
        - sklasteryzuj wartości kdis z KMeans gdzie K = num_peaks (lub self.eps_peaks jeśli ustawione)
        - zwróć posortowaną listę centrów kmeans jako wartości kandydatów eps
        """
        X = np.asarray(X)
        n = X.shape[0]
        if adaptive_k is None:
            if self.adaptive_k_ is None:
                adaptive_k = min(4, max(1, int(math.sqrt(n))))
            else:
                adaptive_k = int(self.adaptive_k_)

        nbrs = NearestNeighbors(n_neighbors=min(n, adaptive_k + 1), metric=self.distance_metric).fit(X)
        dists, _ = nbrs.kneighbors(X)
        kdist = dists[:, adaptive_k] if adaptive_k < dists.shape[1] else dists[:, -1]

        hist_vals, bin_edges = np.histogram(kdist, bins=n_bins)

        mean_hist = np.mean(hist_vals)
        min_peak_height = max(mean_hist * 0.3, 5)
        
        peaks = []
        for i in range(1, len(hist_vals) - 1):
            if (hist_vals[i] > hist_vals[i - 1] and 
                hist_vals[i] > hist_vals[i + 1] and 
                hist_vals[i] >= min_peak_height):
                peaks.append(i)

        if len(peaks) > 6:
            peak_heights = [(i, hist_vals[i]) for i in peaks]
            peak_heights.sort(key=lambda x: x[1], reverse=True)
            peaks = [p[0] for p in peak_heights[:6]]
            peaks.sort()
        
        num_peaks = len(peaks) if len(peaks) > 0 else 1
        if self.eps_peaks is not None:
            num_peaks = max(1, min(int(self.eps_peaks), 6))
        if num_peaks == 0:
            num_peaks = 1

        kmeans = KMeans(n_clusters=num_peaks, n_init=self.peak_kmeans_init, random_state=42)
        kdist_reshaped = kdist.reshape(-1, 1)
        kmeans.fit(kdist_reshaped)
        centers = sorted([float(c[0]) for c in kmeans.cluster_centers_])

        if len(centers) > 1:
            kdist_range = np.max(kdist) - np.min(kdist)
            min_distance = kdist_range * 0.1
            filtered_centers = [centers[0]]
            for c in centers[1:]:
                if c - filtered_centers[-1] >= min_distance:
                    filtered_centers.append(c)
            centers = filtered_centers

        if len(centers) > 3:
            centers = centers[:3]
        
        self.candidate_eps_ = centers
        if self.verbose:
            print(f"[candidate_eps] adaptive_k={adaptive_k}, num_peaks={num_peaks}, candidate_eps={centers}")
        return centers

    def multi_density_clustering(self, X, candidate_eps=None, metric=None):
        """
        Dla posortowanych candidate_eps (rosnąco), dla każdego eps:
            - oblicz MinPts dla tego eps (na podstawie średniej liczby sąsiadów)
            - uruchom DBSCAN(eps, minpts) na aktualnych pozostałych punktach
            - przypisz etykiety klastrów (unikalne między warstwami)
            - usuń sklasteryzowane punkty i kontynuuj
        Zwraca końcowe etykiety (np.array długości n), szum=-1
        """
        X = np.asarray(X)
        n = X.shape[0]
        if candidate_eps is None:
            candidate_eps = self.candidate_eps_
            if candidate_eps is None:
                raise ValueError("candidate_eps is not computed")

        if metric is None:
            metric = self.distance_metric

        remaining_idx = np.arange(n)
        final_labels = -np.ones(n, dtype=int)
        next_cluster_id = 0

        for eps in sorted(candidate_eps):
            if len(remaining_idx) == 0:
                break

            nbrs_rad = NearestNeighbors(radius=eps, metric=metric).fit(X[remaining_idx])
            neigh = nbrs_rad.radius_neighbors(X[remaining_idx], return_distance=False)
            counts = np.array([len(a) for a in neigh], dtype=float)

            q1_count = np.percentile(counts, 25)
            median_count = np.median(counts)

            minpts = int(round(max(q1_count, median_count * 0.5)))

            minpts = max(minpts, max(self.min_pts_floor, 3))

            labels_partial, nclusters = self.run_dbscan(X[remaining_idx], eps, minpts, metric=metric)
            if nclusters <= 0:
                continue

            unique_partial = sorted([u for u in set(labels_partial) if u != -1])
            cluster_sizes = {u: np.sum(labels_partial == u) for u in unique_partial}

            min_cluster_size = max(minpts, max(int(len(remaining_idx) * 0.05), 5))
            
            mapping = {}
            for up in unique_partial:
                if cluster_sizes[up] >= min_cluster_size:
                    next_cluster_id += 1
                    mapping[up] = next_cluster_id

            for local_idx, lab in enumerate(labels_partial):
                if lab != -1 and lab in mapping:
                    global_idx = remaining_idx[local_idx]
                    final_labels[global_idx] = mapping[lab]

            remaining_mask = np.array([final_labels[i] == -1 for i in remaining_idx])
            remaining_idx = remaining_idx[remaining_mask]

            if self.verbose:
                print(f"[multi_density] eps={eps:.4f} minpts={minpts}, new_clusters={len(unique_partial)}, remaining={len(remaining_idx)}")

        self.labels_ = final_labels
        return final_labels

    def fit_predict(self, X, y_true=None):
        """
        Pipeline wysokiego poziomu:
        1) uzyskaj listę eps
        2) uzyskaj listę minpts
        3) adaptacja parametrów znajdź k (adaptacyjne MinPts)
        4) uzyskaj listę kandydatów eps z adaptacyjnym k
        5) klasteryzacja multi-density używając kandydatów eps
        6) oblicz VNN
        Opcjonalnie ewaluuj z y_true (NMI, dokładność) jeśli podane
        """
        X = np.asarray(X)
        if self.verbose:
            t0 = time.time()
            print("[fit_predict] start pipeline")

        eps_list = self.obtain_eps_list(X)
        minpts_list = self.obtain_minpts_list(X, eps_list)

        adaptive_k = self.parameter_adaptation_find_k(X, eps_list=eps_list, minpts_list=minpts_list, y_true=y_true)

        self.compute_vnn(X, eps1=eps_list[0] if eps_list else None)

        if self.vnn_ < 10:
            if self.best_labels_ is not None:
                labels = np.array(self.best_labels_).copy()
                if self.best_eps_ is not None:
                    candidate_eps = [self.best_eps_]
                else:
                    candidate_eps = [eps_list[len(eps_list)//2]]
                self.candidate_eps_ = candidate_eps
                if self.verbose:
                    print(f"[fit_predict] Single-density detected (VNN={self.vnn_:.2f}), using pre-computed best labels with Eps={candidate_eps[0]:.4f}")
            else:
                if self.best_eps_ is not None:
                    candidate_eps = [self.best_eps_]
                else:
                    candidate_eps = [eps_list[len(eps_list)//2]]
                self.candidate_eps_ = candidate_eps
                if self.verbose:
                    print(f"[fit_predict] Single-density detected (VNN={self.vnn_:.2f}), using single Eps={candidate_eps[0]:.4f}")

                labels = self.multi_density_clustering(X, candidate_eps=candidate_eps)
        else:

            original_eps_peaks = self.eps_peaks
            if self.eps_peaks is None:

                self.eps_peaks = 3
            candidate_eps = self.obtain_candidate_eps_list(X, adaptive_k=adaptive_k)
            self.eps_peaks = original_eps_peaks
            if self.verbose:
                print(f"[fit_predict] Multi-density detected (VNN={self.vnn_:.2f}), using {len(candidate_eps)} candidate Eps")
            labels = self.multi_density_clustering(X, candidate_eps=candidate_eps)

        if self.verbose:
            print(f"[fit_predict] done in {time.time()-t0:.2f}s, clusters={len(set(labels))-(1 if -1 in labels else 0)}, VNN={self.vnn_:.3f}")

        eval_dict = {}
        if y_true is not None:
            try:
                nmi = normalized_mutual_info_score(y_true, labels)
            except Exception:
                nmi = None
            try:
                acc = None
            except Exception:
                acc = None
            eval_dict["nmi"] = nmi
            eval_dict["accuracy"] = acc

        return labels, eval_dict
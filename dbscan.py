import numpy as np
from collections import deque

class DBSCAN:
    def __init__(self, eps=0.5, min_pts=5):
        """
        eps      – promień sąsiedztwa
        min_pts  – minimalna liczba punktów w sąsiedztwie (włącznie z punktem centralnym)
        """
        self.eps = eps
        self.min_pts = min_pts

    def _region_query(self, X, point_idx):
        """
        Zwraca listę indeksów punktów w odległości <= eps
        """
        neighbors = []
        for i, p in enumerate(X):
            if np.linalg.norm(p - X[point_idx]) <= self.eps:
                neighbors.append(i)
        return neighbors

    def fit_predict(self, X):
        """
        Główna implementacja DBSCAN.
        Zwraca etykiety klastrów (lub -1 dla szumu).
        """
        X = np.asarray(X)
        n = len(X)

        labels = [-1] * n   # -1 oznacza szum/nieprzypisane
        cluster_id = 0
        visited = set()

        for i in range(n):
            if i in visited:
                continue

            visited.add(i)
            neighbors = self._region_query(X, i)

            # Jeśli punkt ma za mało sąsiadów, to szum
            if len(neighbors) < self.min_pts:
                labels[i] = -1
                continue

            # nowy klaster
            cluster_id += 1
            labels[i] = cluster_id

            # rozszerzanie klastera BFS-em
            queue = deque(neighbors)

            while queue:
                idx = queue.popleft()

                if idx not in visited:
                    visited.add(idx)
                    new_neighbors = self._region_query(X, idx)

                    # Jeśli punkt ma dużo sąsiadów -> punkt rdzeniowy, rozszerzanie
                    if len(new_neighbors) >= self.min_pts:
                        queue.extend(new_neighbors)

                # Jeśli punkt nie jest jeszcze przypisany, przypissnie do klastra
                if labels[idx] == -1:
                    labels[idx] = cluster_id

        return np.array(labels)

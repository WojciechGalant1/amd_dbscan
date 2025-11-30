import numpy as np
from collections import deque

class DBSCAN:
    def __init__(self, eps=0.5, min_pts=5):
        """
        eps      – promień sąsiedztwa (maksymalna odległość, aby punkt był uznany za sąsiada)
        min_pts  – minimalna liczba punktów w sąsiedztwie, aby punkt był rdzeniowy
                    (liczone łącznie z punktem centralnym)
        """
        self.eps = eps
        self.min_pts = min_pts

    def _region_query(self, X, point_idx):
        """
        Zwraca listę indeksów punktów znajdujących się w odległości <= eps od punktu point_idx.
        Ta funkcja implementuje przeszukiwanie sąsiedztwa dla DBSCAN.
        """
        neighbors = []
        for i, p in enumerate(X):
            # Sprawdzamy odległość euklidesową od punktu centralnego
            if np.linalg.norm(p - X[point_idx]) <= self.eps:
                neighbors.append(i)
        return neighbors

    def fit_predict(self, X):
        """
        Główna implementacja algorytmu DBSCAN.
        
        Kroki:
        1. Iterujemy po punktach, pomijając już odwiedzone.
        2. Dla każdego punktu sprawdzamy jego sąsiedztwo.
        3. Jeśli liczba sąsiadów < min_pts → oznaczamy jako szum (label = -1).
        4. W przeciwnym razie tworzymy nowy klaster.
        5. Rozszerzamy klaster BFS-em (kolejką), dołączając punkty rdzeniowe
           oraz oznaczając punkty graniczne.
        
        Zwraca:
        labels – wektor etykiet klastrów (liczby całkowite od 1 wzwyż, -1 = szum)
        """
        X = np.asarray(X)
        n = len(X)

        labels = [-1] * n       # Na początku każdy punkt jest oznaczony jako szum
        cluster_id = 0          # Licznik aktualnego klastra
        visited = set()         # Zbiór odwiedzonych punktów (aby nie odwiedzać 2x)

        for i in range(n):
            # Punkt już sprawdzony → pomiń
            if i in visited:
                continue

            visited.add(i)

            # Pobieramy sąsiadów punktu i
            neighbors = self._region_query(X, i)

            # Jeśli punkt nie jest rdzeniowy → pozostaje szumem
            if len(neighbors) < self.min_pts:
                labels[i] = -1
                continue

            # W przeciwnym razie startujemy nowy klaster
            cluster_id += 1
            labels[i] = cluster_id

            # Kolejka do rozszerzania klastra (BFS)
            queue = deque(neighbors)

            # Rozszerzanie klastra
            while queue:
                idx = queue.popleft()

                # Jeśli punkt jeszcze nie był odwiedzony → odwiedzamy go
                if idx not in visited:
                    visited.add(idx)

                    # Sprawdzamy jego sąsiedztwo
                    new_neighbors = self._region_query(X, idx)

                    # Punkt rdzeniowy – dokładamy jego sąsiadów do kolejki (rozszerzamy klaster)
                    if len(new_neighbors) >= self.min_pts:
                        queue.extend(new_neighbors)

                # Jeśli punkt był wcześniej oznaczony jako szum (-1), a teraz trafia do klastra,
                # oznacz go odpowiednim numerem klastra
                if labels[idx] == -1:
                    labels[idx] = cluster_id

        return np.array(labels)

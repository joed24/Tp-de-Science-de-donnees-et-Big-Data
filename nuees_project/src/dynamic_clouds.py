"""Implémentation from scratch des Nuées dynamiques.

Le moteur alterne deux opérations :
    1. affectation des individus aux représentations les plus proches ;
    2. reconstruction des représentations à partir de la partition.

Aucune fonction de clustering de scikit-learn (ou d'une autre bibliothèque
spécialisée) n'est utilisée.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

EPS = 1e-12


def squared_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Distance euclidienne au carré."""
    d = np.asarray(a) - np.asarray(b)
    return float(np.dot(d, d))


def pairwise_squared_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Matrice des distances euclidiennes au carré."""
    return np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)


def initialize_labels_random(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Initialise K classes à partir de K individus distincts.

    Ce choix suit l'esprit de l'article de Diday : en l'absence d'information
    a priori, les étalons de départ peuvent être tirés automatiquement.
    """
    n = len(X)
    if k < 1 or k > n:
        raise ValueError("k doit être compris entre 1 et le nombre d'observations.")
    seed_indices = rng.choice(n, size=k, replace=False)
    centers = X[seed_indices]
    return np.argmin(pairwise_squared_distances(X, centers), axis=1)


def choose_representatives(X: np.ndarray, q: int) -> np.ndarray:
    """Sélection gloutonne de q étalons parmi les individus de la classe.

    Le premier étalon est le médoïde. Les suivants minimisent progressivement
    la somme des distances de couverture des observations vers l'ensemble des
    étalons. L'objectif est d'obtenir un petit squelette représentatif de la
    classe, sans bibliothèque de clustering.
    """
    X = np.asarray(X, dtype=float)
    if len(X) == 0:
        raise ValueError("Une classe vide ne peut pas fournir d'étalons.")
    q = max(1, min(int(q), len(X)))
    D = pairwise_squared_distances(X, X)

    selected = [int(np.argmin(np.sum(D, axis=1)))]
    remaining = set(range(len(X)))
    remaining.remove(selected[0])
    min_d = D[:, selected[0]].copy()

    while len(selected) < q and remaining:
        best_idx = min(
            remaining,
            key=lambda idx: float(np.sum(np.minimum(min_d, D[:, idx]))),
        )
        selected.append(int(best_idx))
        remaining.remove(best_idx)
        min_d = np.minimum(min_d, D[:, best_idx])

    return X[np.asarray(selected)].copy()


def covariance_matrix(X: np.ndarray, regularization: float = 1e-6) -> np.ndarray:
    """Covariance robuste, régularisée pour permettre l'inversion."""
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    if len(X) <= 1:
        return np.eye(d) * regularization
    cov = np.cov(X, rowvar=False, bias=False)
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    if cov.shape != (d, d):
        cov = np.eye(d) * max(float(np.mean(np.diag(cov))), regularization)
    return cov + regularization * np.eye(d)


@dataclass
class PointPrototype:
    center: np.ndarray

    def distance(self, x: np.ndarray) -> float:
        return squared_euclidean(x, self.center)


@dataclass
class PointsPrototype:
    representatives: np.ndarray

    def distance(self, x: np.ndarray) -> float:
        return float(np.min(np.sum((self.representatives - x) ** 2, axis=1)))


@dataclass
class AxesPrototype:
    center: np.ndarray
    axes: np.ndarray
    eigenvalues: np.ndarray

    def distance(self, x: np.ndarray) -> float:
        if self.axes.size == 0:
            return squared_euclidean(x, self.center)
        z = x - self.center
        projection = self.axes @ (self.axes.T @ z)
        residual = z - projection
        return float(np.dot(residual, residual))


@dataclass
class DistributionPrototype:
    mean: np.ndarray
    covariance: np.ndarray
    inv_covariance: np.ndarray

    def distance(self, x: np.ndarray) -> float:
        z = x - self.mean
        return float(z @ self.inv_covariance @ z)


@dataclass
class StructurePrototype:
    center: np.ndarray
    covariance: np.ndarray
    inv_covariance: np.ndarray
    representatives: np.ndarray

    def components(self, x: np.ndarray) -> tuple[float, float, float]:
        d_center = squared_euclidean(x, self.center)
        z = x - self.center
        d_maha = float(z @ self.inv_covariance @ z)
        d_rep = float(np.min(np.sum((self.representatives - x) ** 2, axis=1)))
        return d_center, d_maha, d_rep

    def distance(self, x: np.ndarray, scales: tuple[float, float, float]) -> float:
        a, b, c = self.components(x)
        sa, sb, sc = scales
        return (a / sa + b / sb + c / sc) / 3.0


class DynamicClouds:
    """Moteur générique des Nuées dynamiques."""

    MODES = {
        "point": "Point (K-Means)",
        "points": "Points représentatifs",
        "axes": "Axes factoriels",
        "distribution": "Distribution probabiliste",
        "structure": "Structure représentative",
    }

    def __init__(
        self,
        k: int,
        mode: str,
        q: int = 3,
        max_iter: int = 50,
        tol: float = 1e-5,
        seed: int = 42,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Mode inconnu : {mode}")
        if k < 1:
            raise ValueError("k doit être positif.")
        self.k = int(k)
        self.mode = mode
        self.q = max(1, int(q))
        self.max_iter = max(1, int(max_iter))
        self.tol = float(tol)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.prototypes: list[Any] = []
        self.labels_: np.ndarray | None = None
        self.history_: list[float] = []

    def _repair_empty_class(self, X: np.ndarray, labels: np.ndarray, empty_class: int) -> np.ndarray:
        """Réinjecte dans une classe vide l'individu actuellement le plus mal représenté."""
        counts = np.bincount(labels, minlength=self.k)
        donors = np.where(counts[labels] > 1)[0]
        if len(donors) == 0:
            donors = np.arange(len(X))
        idx = int(self.rng.choice(donors))
        labels = labels.copy()
        labels[idx] = empty_class
        return labels

    def _build_prototypes(self, X: np.ndarray, labels: np.ndarray) -> list[Any]:
        prototypes: list[Any] = []
        d = X.shape[1]

        # Une partition sans classe vide est indispensable pour construire les prototypes.
        labels = labels.copy()
        for j in range(self.k):
            if not np.any(labels == j):
                labels = self._repair_empty_class(X, labels, j)

        for j in range(self.k):
            members = X[labels == j]
            mean = np.mean(members, axis=0)

            if self.mode == "point":
                prototypes.append(PointPrototype(mean.copy()))

            elif self.mode == "points":
                reps = choose_representatives(members, self.q)
                prototypes.append(PointsPrototype(reps))

            elif self.mode == "axes":
                cov = covariance_matrix(members)
                values, vectors = np.linalg.eigh(cov)
                order = np.argsort(values)[::-1]
                values = values[order]
                vectors = vectors[:, order]
                # En dimension 2, un axe est suffisant pour conserver une distance
                # à un sous-espace non trivial.
                q_axes = min(self.q, max(1, d - 1), d)
                prototypes.append(AxesPrototype(mean.copy(), vectors[:, :q_axes], values[:q_axes]))

            elif self.mode == "distribution":
                cov = covariance_matrix(members)
                prototypes.append(
                    DistributionPrototype(mean.copy(), cov, np.linalg.pinv(cov))
                )

            elif self.mode == "structure":
                cov = covariance_matrix(members)
                reps = choose_representatives(members, self.q)
                prototypes.append(
                    StructurePrototype(mean.copy(), cov, np.linalg.pinv(cov), reps)
                )

        return prototypes

    def _assign(self, X: np.ndarray, prototypes: list[Any]) -> tuple[np.ndarray, np.ndarray]:
        distances = np.zeros((len(X), self.k), dtype=float)

        if self.mode != "structure":
            for j, prototype in enumerate(prototypes):
                distances[:, j] = [prototype.distance(x) for x in X]
        else:
            component_values = np.asarray(
                [prototype.components(x) for prototype in prototypes for x in X],
                dtype=float,
            )
            scales = tuple(
                max(float(np.median(component_values[:, i])), EPS) for i in range(3)
            )
            for j, prototype in enumerate(prototypes):
                distances[:, j] = [prototype.distance(x, scales) for x in X]

        labels = np.argmin(distances, axis=1)
        return labels, distances

    def fit(self, X: np.ndarray) -> "DynamicClouds":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or len(X) < self.k:
            raise ValueError("X doit être une matrice 2D contenant au moins k observations.")
        if not np.isfinite(X).all():
            raise ValueError("X contient des valeurs non finies.")

        labels = initialize_labels_random(X, self.k, self.rng)
        self.history_ = []
        previous_criterion = np.inf

        for iteration in range(1, self.max_iter + 1):
            prototypes = self._build_prototypes(X, labels)
            new_labels, distances = self._assign(X, prototypes)
            criterion = float(np.sum(np.min(distances, axis=1)))
            self.history_.append(criterion)

            changed = not np.array_equal(labels, new_labels)
            relative_change = (
                abs(previous_criterion - criterion) / max(abs(previous_criterion), 1.0)
            )

            self.prototypes = prototypes
            labels = new_labels

            if not changed or relative_change < self.tol:
                break
            previous_criterion = criterion

        # Reconstruction finale correspondant à la dernière partition.
        self.prototypes = self._build_prototypes(X, labels)
        final_labels, final_distances = self._assign(X, self.prototypes)
        final_criterion = float(np.sum(np.min(final_distances, axis=1)))
        self.labels_ = final_labels
        if not self.history_ or abs(self.history_[-1] - final_criterion) > EPS:
            self.history_.append(final_criterion)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.prototypes:
            raise RuntimeError("Le modèle doit être entraîné avant predict().")
        X = np.asarray(X, dtype=float)
        labels, _ = self._assign(X, self.prototypes)
        return labels


def load_numeric_csv(path: str):
    """Charge un CSV et standardise ses colonnes numériques.

    Les colonnes nommées label/target/class sont considérées comme des
    informations de référence et ne sont pas utilisées pour l'apprentissage.
    """
    import pandas as pd

    df = pd.read_csv(path)
    ignored: list[str] = []
    candidate = df.copy()
    for col in list(candidate.columns):
        if col.lower() in {"label", "target", "class"}:
            ignored.append(col)
            candidate = candidate.drop(columns=[col])

    numeric = candidate.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        raise ValueError("Le CSV doit contenir au moins deux colonnes numériques.")

    X = numeric.to_numpy(dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < EPS] = 1.0
    X = (X - mean) / std
    return X, df, ignored


def make_demo_dataset(seed: int = 42, n_per_cluster: int = 100):
    """Génère un jeu 2D reproductible présentant des formes différentes."""
    rng = np.random.default_rng(seed)
    centers = np.array([[-4.0, -2.5], [-1.0, 3.2], [3.4, 2.0], [3.0, -3.2]])
    covariances = [
        np.array([[1.3, 0.55], [0.55, 0.55]]),
        np.array([[0.55, -0.25], [-0.25, 1.25]]),
        np.array([[1.1, 0.0], [0.0, 0.65]]),
        np.array([[0.45, 0.25], [0.25, 1.5]]),
    ]
    blocks = [
        rng.multivariate_normal(center, cov, n_per_cluster)
        for center, cov in zip(centers, covariances)
    ]
    X = np.vstack(blocks)
    labels = np.concatenate(
        [np.full(n_per_cluster, i, dtype=int) for i in range(len(centers))]
    )
    return X, labels

"""Implémentation du Perceptron de Rosenblatt (1958) en POO avec NumPy.

Le perceptron est un classifieur binaire linéaire :

    z = w·x + b          (entrée nette)
    ŷ = 1 si z >= 0 sinon 0   (fonction d'activation de Heaviside)

Règle d'apprentissage de Rosenblatt (mise à jour à chaque exemple) :

    Δw = η (y - ŷ) x
    Δb = η (y - ŷ)
"""
from __future__ import annotations

import numpy as np


class Perceptron:
    """Classifieur Perceptron binaire (étiquettes 0 / 1).

    Parameters
    ----------
    eta : float, default=0.01
        Taux d'apprentissage (learning rate), strictement positif.
    n_iter : int, default=50
        Nombre maximal de passes (epochs) sur le jeu d'entraînement.
    shuffle : bool, default=True
        Mélange les exemples à chaque epoch (évite les cycles dus à l'ordre).
    random_state : int | None, default=None
        Graine pour l'initialisation des poids et le mélange (reproductibilité).
    early_stopping : bool, default=True
        Arrête l'entraînement dès qu'une epoch ne produit aucune erreur
        (convergence, garantie si les données sont linéairement séparables).

    Attributes
    ----------
    w_ : np.ndarray, shape (n_features,)
        Poids appris.
    b_ : float
        Biais appris.
    errors_ : list[int]
        Nombre d'erreurs de classification (mises à jour) à chaque epoch.
    classes_ : np.ndarray
        Étiquettes des deux classes (classes_[0] -> 0, classes_[1] -> 1).
    n_iter_ : int
        Nombre d'epochs réellement effectuées.
    """

    def __init__(
        self,
        eta: float = 0.01,
        n_iter: int = 50,
        shuffle: bool = True,
        random_state: int | None = None,
        early_stopping: bool = True,
    ) -> None:
        if eta <= 0:
            raise ValueError("eta doit être strictement positif.")
        if n_iter < 1:
            raise ValueError("n_iter doit être >= 1.")
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.early_stopping = early_stopping

    # ------------------------------------------------------------------ #
    # Entraînement
    # ------------------------------------------------------------------ #
    def fit(self, X, y) -> "Perceptron":
        """Entraîne le perceptron.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,) — exactement deux classes.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X doit être un tableau 2D (n_samples, n_features).")
        if len(X) != len(y):
            raise ValueError("X et y doivent avoir le même nombre de lignes.")

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Le perceptron est un classifieur binaire (2 classes).")
        y_bin = (y == self.classes_[1]).astype(int)

        rng = np.random.default_rng(self.random_state)
        self.w_ = rng.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0.0
        self.errors_: list[int] = []

        for epoch in range(self.n_iter):
            idx = rng.permutation(len(X)) if self.shuffle else np.arange(len(X))
            errors = 0
            for i in idx:
                update = self.eta * (y_bin[i] - self._activation(self.net_input(X[i])))
                if update != 0.0:
                    self.w_ += update * X[i]
                    self.b_ += update
                    errors += 1
            self.errors_.append(errors)
            self.n_iter_ = epoch + 1
            if self.early_stopping and errors == 0:
                break
        return self

    # ------------------------------------------------------------------ #
    # Inférence
    # ------------------------------------------------------------------ #
    def net_input(self, X) -> np.ndarray:
        """Entrée nette z = w·x + b."""
        return np.dot(X, self.w_) + self.b_

    @staticmethod
    def _activation(z) -> np.ndarray:
        """Fonction de Heaviside : 1 si z >= 0, sinon 0."""
        return np.where(z >= 0.0, 1, 0)

    def decision_function(self, X) -> np.ndarray:
        """Distance signée (non normalisée) à l'hyperplan de décision."""
        self._check_fitted()
        return self.net_input(np.asarray(X, dtype=float))

    def predict(self, X) -> np.ndarray:
        """Renvoie les étiquettes prédites (dans les classes d'origine)."""
        return self.classes_[self._activation(self.decision_function(X))]

    def score(self, X, y) -> float:
        """Exactitude (accuracy) moyenne sur (X, y)."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if not hasattr(self, "w_"):
            raise RuntimeError("Appelez fit() avant d'utiliser le modèle.")

    def __repr__(self) -> str:
        return (f"Perceptron(eta={self.eta}, n_iter={self.n_iter}, "
                f"shuffle={self.shuffle}, random_state={self.random_state})")

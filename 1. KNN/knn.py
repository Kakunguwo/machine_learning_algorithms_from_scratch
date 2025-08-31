import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    """
    A simple k-Nearest Neighbors classifier (from scratch).

    Parameters
    ----------
    k : int (default=3)
        Number of nearest neighbors to vote.

    Attributes
    ----------
    X_train : np.ndarray
        Training features, shape (n_samples, n_features).
    y_train : np.ndarray
        Training labels, shape (n_samples,).
    """

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        """Store the training data (no training happens for KNN)."""
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        X = np.asarray(X)
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        """
        Predict the label for a single sample x by majority vote among k nearest.
        """
        # distances to all training points
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # neighbor labels
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
        label, _count = Counter(k_nearest_labels).most_common(1)[0]
        return label

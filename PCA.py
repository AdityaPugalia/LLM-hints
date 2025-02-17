import numpy as np
import joblib
from sklearn.decomposition import PCA

class PCAEmbeddingReducer:
    """
    A PCA-based embedding reducer that preserves a given percentage of variance.
    Allows saving and loading trained PCA models.

    Attributes:
        variance_threshold (float): The percentage of variance to preserve.
        pca (PCA): Scikit-learn PCA model.
        optimal_components (int): Number of components preserving the variance.
    """

    def __init__(self, variance_threshold=0.90):
        """
        Initializes the PCAEmbeddingReducer with a specified variance threshold.

        :param variance_threshold: Percentage of variance to preserve (default=90%).
        """
        self.variance_threshold = variance_threshold
        self.pca = None
        self.optimal_components = None

    def fit(self, embeddings, transform : bool = False):
        """
        Fits PCA on the given embeddings and determines the optimal number of components.

        :param embeddings: NumPy array of shape (n_samples, original_dim)
        """
        # Step 1: Fit PCA on the data
        pca_full = PCA()
        pca_full.fit(embeddings)

        # Step 2: Compute cumulative variance
        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

        # Step 3: Find the minimum components to preserve the required variance
        # Corrected logic
        self.optimal_components = (
            np.argmax(cumulative_variance >= self.variance_threshold) + 1
            if np.any(cumulative_variance >= self.variance_threshold)
            else len(cumulative_variance)
        )
        print(f"Optimal components for {cumulative_variance[self.optimal_components - 1]*100}% variance: {self.optimal_components}")

        # Step 4: Fit PCA with the optimal number of components
        self.pca = PCA(n_components=self.optimal_components)
        if transform:
            return np.ascontiguousarray(self.pca.fit_transform(embeddings))
        else:
            self.pca.fit(embeddings)
            return None

    def transform(self, embeddings):
        """
        Reduces the dimensionality of new embeddings using the trained PCA model.

        :param embeddings: NumPy array of shape (n_samples, original_dim)
        :return: NumPy array of shape (n_samples, optimal_dim)
        """
        if self.pca is None:
            raise ValueError("PCA model is not trained. Call `fit()` first.")
        return np.ascontiguousarray(self.pca.transform(embeddings))

    def save(self, filepath):
        """
        Saves the trained PCA model to a file.

        :param filepath: Path to save the PCA model.
        """
        if self.pca is None:
            raise ValueError("PCA model is not trained. Call `fit()` first.")
        joblib.dump({"pca": self.pca, "optimal_components": self.optimal_components}, filepath)
        print(f"PCA model saved to {filepath}")

    def load(self, filepath):
        """
        Loads a pre-trained PCA model from a file.

        :param filepath: Path to load the PCA model from.
        """
        model_data = joblib.load(filepath)
        self.pca = model_data["pca"]
        self.optimal_components = model_data["optimal_components"]
        print(f"PCA model loaded from {filepath}")
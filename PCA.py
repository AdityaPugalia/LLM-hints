import numpy as np
import joblib
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse


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
        self.reducer = None
        self.optimal_components = None

    def fit(self, embeddings, transform: bool = False):
        """
        Fits SVD (or PCA) on the given sparse embeddings and determines the optimal number of components.
        
        :param embeddings: NumPy array or sparse matrix of shape (n_samples, original_dim)
        :param transform: If True, returns the transformed embeddings.
        :return: Transformed embeddings if transform=True, otherwise None.
        """
        # Step 1: Check if data is sparse
        is_sparse = issparse(embeddings)

        # Step 2: Scale the data if it's not sparse (recommended for PCA)
        if not is_sparse:
            scaler = StandardScaler()
            embeddings = scaler.fit_transform(embeddings)
            reducer_full = PCA()
            pca = True
        else:
            # Step 3: Use Truncated SVD (PCA alternative for sparse data)
            reducer_full = TruncatedSVD(n_components=min(embeddings.shape) - 1)
            pca = False
        reducer_full.fit(embeddings)

        # Step 4: Compute cumulative variance
        cumulative_variance = np.cumsum(reducer_full.explained_variance_ratio_)

        # Step 5: Find the minimum components to preserve the required variance
        self.optimal_components = (
            np.argmax(cumulative_variance >= self.variance_threshold) + 1
            if np.any(cumulative_variance >= self.variance_threshold)
            else len(cumulative_variance)
        )

        print(f"Optimal components for {cumulative_variance[self.optimal_components - 1]*100:.4f}% variance: {self.optimal_components}")

        # Step 6: Fit SVD with the optimal number of components
        if pca:
            self.reducer = PCA(n_components= self.optimal_components)
        else:
            self.reducer = TruncatedSVD(n_components=self.optimal_components)
        
        if transform:
            return np.ascontiguousarray(self.reducer.fit_transform(embeddings))
        else:
            self.reducer.fit(embeddings)
            return None

    def transform(self, embeddings):
        """
        Reduces the dimensionality of new embeddings using the trained PCA model.

        :param embeddings: NumPy array of shape (n_samples, original_dim)
        :return: NumPy array of shape (n_samples, optimal_dim)
        """
        if self.reducer is None:
            raise ValueError("PCA model is not trained. Call `fit()` first.")
        return np.ascontiguousarray(self.reducer.transform(embeddings))

    def save(self, filepath):
        """
        Saves the trained PCA model to a file.

        :param filepath: Path to save the PCA model.
        """
        if self.reducer is None:
            raise ValueError("PCA model is not trained. Call `fit()` first.")
        joblib.dump({"reducer": self.reducer, "optimal_components": self.optimal_components}, filepath)
        print(f"PCA model saved to {filepath}")

    def load(self, filepath):
        """
        Loads a pre-trained PCA model from a file.

        :param filepath: Path to load the PCA model from.
        """
        model_data = joblib.load(filepath)
        self.reducer = model_data["reducer"]
        self.optimal_components = model_data["optimal_components"]
        print(f"PCA model loaded from {filepath}")
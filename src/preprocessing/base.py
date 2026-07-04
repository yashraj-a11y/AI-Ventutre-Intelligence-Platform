from abc import ABC, abstractmethod
import joblib


class BasePreprocessor(ABC):
    """
    Abstract base class for all preprocessing components.
    """

    @abstractmethod
    def fit(self, X, y=None):
        """Learn preprocessing parameters from data."""
        pass

    @abstractmethod
    def transform(self, X):
        """Apply learned preprocessing to data."""
        pass

    def fit_transform(self, X, y=None):
        """
        Fit the preprocessor and immediately transform the data.
        """
        self.fit(X, y)
        return self.transform(X)

    def save(self, filepath):
        """
        Save the fitted preprocessor to disk.
        """
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        """
        Load a saved preprocessor.
        """
        return joblib.load(filepath)
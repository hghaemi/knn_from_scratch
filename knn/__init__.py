"""
KNN Package - K-Nearest Neighbors implementation from scratch
"""

from .models import KNNClassifier, KNNRegressor
from .utils import generate_classification_data, generate_regression_data, euclidean_distance

__version__ = "0.1.0"
__author__ = "M. Hossein Ghaemi"
__email__ = "h.ghaemi.2003@gmail.com"

__all__ = [
    'KNNClassifier',
    'KNNRegressor', 
    'generate_classification_data',
    'generate_regression_data',
    'euclidean_distance'
]
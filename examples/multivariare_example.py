import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knn import KNNClassifier, KNNRegressor, generate_classification_data, generate_regression_data
from knn.utils import train_test_split, accuracy_score, mean_squared_error, manhattan_distance


def compare_distance_metrics():
    print("=== Comparing Distance Metrics ===")
    
    X, y = generate_classification_data(n_samples=300, n_features=5, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    metrics = [
        ('Euclidean', 'euclidean'),
        ('Manhattan', manhattan_distance)
    ]
    
    for name, metric in metrics:
        knn = KNNClassifier(k=7, distance_metric=metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Distance - Accuracy: {accuracy:.3f}")


def compare_k_values():
    print("\n=== Comparing K Values ===")
    
    X, y = generate_classification_data(n_samples=300, n_features=3, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    k_values = [1, 3, 5, 7, 11, 15]
    
    for k in k_values:
        knn = KNNClassifier(k=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"k={k:2d} - Accuracy: {accuracy:.3f}")


def weighted_regression():
    print("\n=== Weighted vs Uniform Regression ===")
    
    X, y = generate_regression_data(n_samples=200, n_features=2, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    weights = ['uniform', 'distance']
    
    for weight in weights:
        knn = KNNRegressor(k=5, weights=weight)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"{weight.capitalize()} weights - MSE: {mse:.3f}")


if __name__ == "__main__":
    compare_distance_metrics()
    compare_k_values()
    weighted_regression()
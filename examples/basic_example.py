import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knn import KNNClassifier, KNNRegressor, generate_classification_data, generate_regression_data
from knn.utils import train_test_split, accuracy_score, mean_squared_error


def classification_example():
    print("=== KNN Classification Example ===")
    
    X, y = generate_classification_data(n_samples=200, n_features=2, n_classes=3, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    knn = KNNClassifier(k=5)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    y_proba = knn.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Sample predictions: {y_pred[:5]}")
    print(f"Sample probabilities: {y_proba[:3]}")


def regression_example():
    print("\n=== KNN Regression Example ===")
    
    X, y = generate_regression_data(n_samples=200, n_features=1, noise=0.2, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    knn = KNNRegressor(k=5, weights='distance')
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Mean Squared Error: {mse:.3f}")
    print(f"Sample predictions: {y_pred[:5]}")
    print(f"Sample actual: {y_test[:5]}")


if __name__ == "__main__":
    classification_example()
    regression_example()
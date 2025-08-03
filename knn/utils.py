import numpy as np


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))


def generate_classification_data(n_samples=100, n_features=2, n_classes=2, 
                                n_clusters_per_class=1, random_state=None):

    if random_state is not None:
        np.random.seed(random_state)
    
    X = []
    y = []
    
    samples_per_class = n_samples // n_classes
    samples_per_cluster = samples_per_class // n_clusters_per_class
    
    for class_idx in range(n_classes):
        for cluster_idx in range(n_clusters_per_class):

            center = np.random.randn(n_features) * 3
            

            cluster_samples = np.random.randn(samples_per_cluster, n_features) + center
            X.extend(cluster_samples)
            y.extend([class_idx] * samples_per_cluster)
    

    remaining = n_samples - len(X)
    if remaining > 0:
        class_idx = np.random.randint(0, n_classes)
        center = np.random.randn(n_features) * 3
        remaining_samples = np.random.randn(remaining, n_features) + center
        X.extend(remaining_samples)
        y.extend([class_idx] * remaining)
    
    X = np.array(X)
    y = np.array(y)
    

    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def generate_regression_data(n_samples=100, n_features=1, noise=0.1, random_state=None):

    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    
    if n_features == 1:
        y = np.sin(X[:, 0] * 2) + 0.5 * X[:, 0] + noise * np.random.randn(n_samples)
    else:
        y = np.sum(X ** 2, axis=1) + noise * np.random.randn(n_samples)
    
    return X, y


def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
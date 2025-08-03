import numpy as np
from collections import Counter
from .utils import euclidean_distance


class KNNClassifier:
    def __init__(self, k=5, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _get_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return euclidean_distance(x1, x2)
        elif callable(self.distance_metric):
            return self.distance_metric(x1, x2)
        else:
            raise ValueError("Unsupported distance metric")
    
    def _get_neighbors(self, x):
        distances = []

        for i, train_point in enumerate(self.X_train):
            dist = self._get_distance(x, train_point)
            distances.append((dist, i))
        
        distances.sort(key=lambda x: x[0])
        neighbor_indices = [idx for _, idx in distances[:self.k]]
        return neighbor_indices
    
    def predict(self, X):
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction")
            
        X = np.array(X)
        predictions = []
        
        for x in X:
            neighbor_indices = self._get_neighbors(x)
            neighbor_labels = [self.y_train[i] for i in neighbor_indices]
            
            most_common = Counter(neighbor_labels).most_common(1)
            predictions.append(most_common[0][0])
            
        return np.array(predictions)
    
    def predict_proba(self, X):

        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction")
            
        X = np.array(X)
        classes = np.unique(self.y_train)
        probabilities = []
        
        for x in X:
            neighbor_indices = self._get_neighbors(x)
            neighbor_labels = [self.y_train[i] for i in neighbor_indices]
            
            class_counts = Counter(neighbor_labels)
            probs = []
            for cls in classes:
                probs.append(class_counts.get(cls, 0) / self.k)
            probabilities.append(probs)
            
        return np.array(probabilities)


class KNNRegressor:
    
    def __init__(self, k=5, distance_metric='euclidean', weights='uniform'):
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self
    
    def _get_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return euclidean_distance(x1, x2)
        elif callable(self.distance_metric):
            return self.distance_metric(x1, x2)
        else:
            raise ValueError("Unsupported distance metric")
    
    def _get_neighbors(self, x):

        distances = []

        for i, train_point in enumerate(self.X_train):
            dist = self._get_distance(x, train_point)
            distances.append((dist, i))
        
        distances.sort(key=lambda x: x[0])
        return distances[:self.k]
    
    def predict(self, X):

        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction")
            
        X = np.array(X)
        predictions = []
        
        for x in X:
            neighbors = self._get_neighbors(x)
            
            if self.weights == 'uniform':
                neighbor_values = [self.y_train[idx] for _, idx in neighbors]
                prediction = np.mean(neighbor_values)
            elif self.weights == 'distance':
                # Distance-weighted average
                total_weight = 0
                weighted_sum = 0
                
                for dist, idx in neighbors:

                    if dist == 0:
                        prediction = self.y_train[idx]
                        break

                    weight = 1 / dist
                    weighted_sum += weight * self.y_train[idx]
                    total_weight += weight
                    
                else:
                    prediction = weighted_sum / total_weight
            
            predictions.append(prediction)
            
        return np.array(predictions)
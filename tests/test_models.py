import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knn import KNNClassifier, KNNRegressor, generate_classification_data, generate_regression_data
from knn.utils import train_test_split, accuracy_score, mean_squared_error


class TestKNNClassifier(unittest.TestCase):
    
    def setUp(self):
        self.X, self.y = generate_classification_data(n_samples=100, n_features=2, 
                                                    n_classes=2, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42)
    
    def test_fit_predict(self):
        knn = KNNClassifier(k=3)
        knn.fit(self.X_train, self.y_train)
        predictions = knn.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))


    def test_predict_proba(self):
        knn = KNNClassifier(k=5)
        knn.fit(self.X_train, self.y_train)
        probabilities = knn.predict_proba(self.X_test)
        
        self.assertEqual(probabilities.shape, (len(self.X_test), 2))

        for prob_row in probabilities:
            self.assertAlmostEqual(sum(prob_row), 1.0, places=5)
    
    def test_different_k_values(self):

        for k in [1, 3, 5, 7]:
            knn = KNNClassifier(k=k)
            knn.fit(self.X_train, self.y_train)
            predictions = knn.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, predictions)
            self.assertGreater(accuracy, 0.5)  # Should be better than random
    
    def test_unfitted_model(self):

        knn = KNNClassifier(k=3)
        with self.assertRaises(ValueError):
            knn.predict(self.X_test)


class TestKNNRegressor(unittest.TestCase):
    
    def setUp(self):

        self.X, self.y = generate_regression_data(n_samples=100, n_features=1, 
                                                noise=0.1, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42)
    
    def test_fit_predict(self):
        knn = KNNRegressor(k=3)
        knn.fit(self.X_train, self.y_train)
        predictions = knn.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(isinstance(pred, (int, float, np.number)) for pred in predictions))
    

    def test_uniform_vs_distance_weights(self):
        knn_uniform = KNNRegressor(k=5, weights='uniform')
        knn_distance = KNNRegressor(k=5, weights='distance')
        
        knn_uniform.fit(self.X_train, self.y_train)
        knn_distance.fit(self.X_train, self.y_train)
        
        pred_uniform = knn_uniform.predict(self.X_test)
        pred_distance = knn_distance.predict(self.X_test)
        
        mse_uniform = mean_squared_error(self.y_test, pred_uniform)
        mse_distance = mean_squared_error(self.y_test, pred_distance)
        
        self.assertLess(mse_uniform, 10)
        self.assertLess(mse_distance, 10)
    
    def test_unfitted_model(self):

        knn = KNNRegressor(k=3)
        with self.assertRaises(ValueError):
            knn.predict(self.X_test)


class TestUtilityFunctions(unittest.TestCase):
    
    def test_data_generation(self):
        X_cls, y_cls = generate_classification_data(n_samples=50, n_features=3, 
                                                  n_classes=2, random_state=42)
        self.assertEqual(X_cls.shape, (50, 3))
        self.assertEqual(len(y_cls), 50)
        self.assertTrue(all(label in [0, 1] for label in y_cls))
        
        X_reg, y_reg = generate_regression_data(n_samples=50, n_features=2, 
                                              noise=0.1, random_state=42)
        self.assertEqual(X_reg.shape, (50, 2))
        self.assertEqual(len(y_reg), 50)
    
    def test_train_test_split(self):

        X = np.random.randn(100, 2)
        y = np.random.randint(0, 2, 100)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        self.assertEqual(len(X_train), 70)
        self.assertEqual(len(X_test), 30)
        self.assertEqual(len(y_train), 70)
        self.assertEqual(len(y_test), 30)


if __name__ == '__main__':
    unittest.main()
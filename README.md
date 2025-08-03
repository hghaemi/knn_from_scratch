# K-Nearest Neighbors from Scratch

A complete, educational implementation of K-Nearest Neighbors (KNN) algorithm for both classification and regression tasks, built from scratch using only NumPy and Matplotlib.

## 🎯 Features

- **Pure Python Implementation**: Built from scratch using only NumPy for computations
- **Dual Functionality**: Both classification and regression implementations
- **Multiple Distance Metrics**: Euclidean, Manhattan, and custom distance functions support
- **Flexible Weighting**: Uniform and distance-based weighting schemes for regression
- **Comprehensive Metrics**: Built-in evaluation metrics including accuracy, MSE, R², precision, recall, and F1-score
- **Visualization Tools**: Decision boundary plotting and performance analysis
- **Data Utilities**: Synthetic data generation and preprocessing utilities
- **Robust Design**: Handles edge cases, multiple classes, and automatic parameter validation
- **Educational Focus**: Clean, well-documented code perfect for learning

## 📋 Requirements

- Python 3.7+
- NumPy 2.3+
- Matplotlib 3.3+ (for visualizations)

## 🚀 Installation

### From source:
```bash
git clone https://github.com/hghaemi/knn_from_scratch.git
cd knn_from_scratch
pip install -e .
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### For development:
```bash
pip install -r requirements-dev.txt
```

## 💡 Quick Start

### Classification Example

```python
from knn import KNNClassifier, generate_classification_data
from knn.utils import train_test_split, accuracy_score

# Generate sample data
X, y = generate_classification_data(n_samples=1000, n_features=2, n_classes=3, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = KNNClassifier(k=5)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")
```

### Regression Example

```python
from knn import KNNRegressor, generate_regression_data
from knn.utils import train_test_split, mean_squared_error, r2_score

# Generate sample data
X, y = generate_regression_data(n_samples=1000, n_features=2, noise=0.1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = KNNRegressor(k=5, weights='distance')
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f"MSE: {mse:.4f}, R²: {r2:.4f}")
```

### Custom Distance Metrics

```python
from knn.utils import manhattan_distance

# Using built-in Manhattan distance
model_manhattan = KNNClassifier(k=5, distance_metric=manhattan_distance)
model_manhattan.fit(X_train, y_train)

# Using custom distance metric
def custom_distance(x1, x2):
    return np.sum(np.abs(x1 - x2) ** 1.5)

model_custom = KNNClassifier(k=5, distance_metric=custom_distance)
model_custom.fit(X_train, y_train)
```

## 📊 Advanced Usage

### Distance Metrics Comparison

```python
from knn.utils import manhattan_distance

# Compare different distance metrics
metrics = {
    'Euclidean': 'euclidean',
    'Manhattan': manhattan_distance
}

for name, metric in metrics.items():
    model = KNNClassifier(k=5, distance_metric=metric)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"{name} Distance - Accuracy: {accuracy:.4f}")
```

### Optimal K Selection

```python
# Find optimal k value
k_values = range(1, 21)
accuracies = []

for k in k_values:
    model = KNNClassifier(k=k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, predictions))

optimal_k = k_values[np.argmax(accuracies)]
print(f"Optimal k: {optimal_k}")
```

### Weighted vs Uniform Regression

```python
# Compare weighting schemes
weights = ['uniform', 'distance']

for weight in weights:
    model = KNNRegressor(k=5, weights=weight)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"{weight.capitalize()} weights - MSE: {mse:.4f}")
```

### Data Preprocessing

```python
# For high-dimensional data, consider feature scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNNClassifier(k=5)
model.fit(X_train_scaled, y_train)
```

## 🏗️ API Reference

### KNNClassifier

**Parameters:**
- `k` (int, default=5): Number of neighbors to consider
- `distance_metric` (str or callable, default='euclidean'): Distance metric to use

**Methods:**
- `fit(X, y)`: Train the classifier
- `predict(X)`: Make class predictions
- `predict_proba(X)`: Get prediction probabilities

### KNNRegressor

**Parameters:**
- `k` (int, default=5): Number of neighbors to consider
- `distance_metric` (str or callable, default='euclidean'): Distance metric to use
- `weights` (str, default='uniform'): Weight function ('uniform' or 'distance')

**Methods:**
- `fit(X, y)`: Train the regressor
- `predict(X)`: Make continuous predictions

### Utility Functions

**Data Generation:**
- `generate_classification_data()`: Generate synthetic classification datasets
- `generate_regression_data()`: Generate synthetic regression datasets

**Distance Metrics:**
- `euclidean_distance(x1, x2)`: Calculate Euclidean distance
- `manhattan_distance(x1, x2)`: Calculate Manhattan distance

**Evaluation Metrics:**
- `accuracy_score(y_true, y_pred)`: Calculate accuracy
- `mean_squared_error(y_true, y_pred)`: Calculate MSE
- `r2_score(y_true, y_pred)`: Calculate R² score

**Data Utilities:**
- `train_test_split(X, y, test_size=0.2)`: Split data into train/test sets

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=knn

# Run specific test file
python -m pytest tests/test_models.py -v
```

## 📁 Project Structure

```
knn-from-scratch/
├── knn/
│   ├── __init__.py
│   ├── models.py          # KNN classifier and regressor classes
│   └── utils.py           # Data generation and utility functions
├── examples/
│   ├── basic_example.py   # Simple usage examples
│   ├── multivariate_example.py  # Advanced usage examples
│   └── visualization_demo.ipynb # Jupyter notebook demos
├── tests/
│   ├── __init__.py
│   └── test_models.py     # Comprehensive test suite
├── setup.py
├── requirements.txt       # Production dependencies
├── requirements-dev.txt   # Development dependencies
├── README.md
├── LICENSE
└── .gitignore
```

## 🤝 Examples

Check out the `examples/` directory for:
- **basic_example.py**: Simple classification and regression examples
- **multivariate_example.py**: Advanced usage with different distance metrics
- **visualization_demo.ipynb**: Interactive Jupyter notebook with visualizations

## 🔬 Mathematical Background

This implementation uses:
- **Distance Metrics**: 
  - Euclidean: d(x,y) = √Σ(x_i - y_i)²
  - Manhattan: d(x,y) = Σ|x_i - y_i|
- **Classification**: Majority vote among k nearest neighbors
- **Regression**: 
  - Uniform: ŷ = (1/k) * Σy_i
  - Distance-weighted: ŷ = Σ(w_i * y_i) / Σw_i, where w_i = 1/d_i

## 🚀 Performance Characteristics

- **Time Complexity**: 
  - Training: O(1) - lazy learning
  - Prediction: O(n * d) per query, where n is training samples, d is dimensions
- **Space Complexity**: O(n * d) for storing training data
- **Best for**: Small to medium datasets (< 50k samples)
- **Considerations**: Performance degrades with high dimensions (curse of dimensionality)

## 🎯 When to Use KNN

**Good for:**
- Simple baseline models
- Non-linear decision boundaries
- Multi-class classification
- Recommendation systems
- Pattern recognition

**Consider alternatives when:**
- Working with high-dimensional data
- Need fast prediction times
- Dataset is very large (> 100k samples)
- Features have very different scales

## 🐛 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `python -m pytest`
5. Commit your changes: `git commit -am 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- Email: h.ghaemi.2003@gmail.com
- GitHub: [@hghaemi](https://github.com/hghaemi/knn_from_scratch.git)

## 🙏 Acknowledgments

- Built for educational purposes to understand KNN from first principles
- Inspired by scikit-learn's API design
- Mathematical foundations from pattern recognition literature

## 🔄 Version History

- **v0.1.0**: Initial release with core functionality
  - KNN Classification and Regression implementations
  - Multiple distance metrics support
  - Comprehensive utilities and visualization tools
  - Full test coverage

---

*Happy Learning! 🚀*
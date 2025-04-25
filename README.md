# ML-Lib

[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A lightweight Python machine learning library implementing core algorithms from scratch.

## Features

- **Clustering Algorithms**
  - K-Means
- **Nearest Neighbors**
  - K-Nearest Neighbors Classifier
  - K-Nearest Neighbors Regressor
- **Linear Models**
  - Linear Regression
  - Logistic Regression

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/yourusername/ml-lib.git
Or clone and install locally:

bash
git clone https://github.com/yourusername/ml-lib.git
cd ml-lib
pip install -e .
Quick Start
K-Nearest Neighbors Regression
python
from ml_lib.neighbors import KNNRegressor
import numpy as np

# Sample data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([3, 7, 11])

# Create and fit model
model = KNNRegressor(k=2)
model.fit(X, y)

# Predict
predictions = model.predict(np.array([[2, 3]]))
print(predictions)  # Output: [5.0]
K-Means Clustering
python
from ml_lib.clustering import KMeansClustering
import numpy as np

# Sample data
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

# Create and fit model
kmeans = KMeansClustering(n_clusters=2)
labels = kmeans.fit(X)

print(labels)  # Cluster assignments
print(kmeans.centroids)  # Cluster centers
Documentation
KNNRegressor
python
KNNRegressor(
    k=5,                     # Number of neighbors
    weights='uniform',       # 'uniform' or 'distance'
    metric='euclidean'       # 'euclidean' or 'manhattan'
)
KMeansClustering
python
KMeansClustering(
    n_clusters=2,           # Number of clusters
    max_iter=100            # Maximum iterations
)
Contributing
Fork the repository

Create a new branch (git checkout -b feature-branch)

Commit your changes (git commit -am 'Add new feature')

Push to the branch (git push origin feature-branch)

Open a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
Nitesh - n9106822@gmail.com
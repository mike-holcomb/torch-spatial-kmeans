# Torch Spatial K-Means

Implementation of a Kernelized K-means clustering algorithm to perform GPU-accelerated clustering using a combination of non-spatial and spatial features.

## Install

1. Clone this repo
2. Install with `pip`

```bash
git clone <this repo>
cd <this repo>
python -m pip install .
```

## Usage

```python
import torch_spatial_kmeans as tsk

data = torch.randn(100, 4)  # Example data with 100 samples, 2 features, and 2 spatial coordinates
feature_weights = torch.ones(2)  # Example feature weights
spatial_weight = 1.0  # Example spatial weight
k = 3  # Number of clusters

centroids, cluster_assignments = tsk.spatial_kmeans(data, k, feature_weights, spatial_weight)
```

# Torch Spatial K-Means

Implementation of a Kernelized K-means clustering algorithm to perform GPU-accelerated clustering using a combination of non-spatial and spatial features.

Assumes input data is of the shape [N, F + S], where
* `N` = number of examples
* `F` = number of non-spatial features
* `S` = number of spatial features

Always assumes that the the spatial features are the last `num_spatial_dims` columns in the provided data.  `F` and `S` must both be greater than 0.

The provided number of clusters (`k` parameter) must be between 2 and `N` (number of provided samples).

## Install

1. Clone this repo
2. Install with `pip`

```bash
git clone https://github.com/mike-holcomb/torch-spatial-kmeans.git
cd torch-spatial-kmeans
python -m pip install .
```

## Usage

```python
>>> import torch_spatial_kmeans as tsk

>>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
>>> centroids, cluster_assignments = tsk.spatial_kmeans(
    data, k=2, spatial_weight=0.5, num_spatial_dims=2)
>>> centroids
tensor([[1, 2, 3],
        [4, 5, 6]])
>>> cluster_assignments
tensor([0, 1])
```

## Docs

```bash
python -m pydoc torch_spatial_kmeans
```

## Tests
```bash
python tests/test_kmeans_functions.py
```

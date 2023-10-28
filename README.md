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
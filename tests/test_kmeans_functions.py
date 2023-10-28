import unittest
import torch
from torch_spatial_kmeans import (
    spatial_kmeans, 
    kmeans_plus_plus_initialization, 
    custom_kernel, 
    assign_clusters, 
    update_centroids, 
    has_converged
)

class TestKMeansFunctions(unittest.TestCase):
    
    def setUp(self):
        # Example data for testing
        self.data = torch.randn(100, 5)  # 100 samples, 3 features, 2 spatial coordinates
        self.k = 3
        self.spatial_weight = 1.0
        self.num_spatial_dims = 2

    def test_kmeans_plus_plus_initialization(self):
        centroids = kmeans_plus_plus_initialization(self.data, self.k)
        self.assertEqual(centroids.size(), (self.k, self.data.size(1)))

    def test_custom_kernel(self):
        centroids = kmeans_plus_plus_initialization(self.data, self.k)
        distances = custom_kernel(self.data, centroids, self.spatial_weight, self.num_spatial_dims)
        self.assertEqual(distances.size(), (self.data.size(0), self.k))

    def test_assign_clusters(self):
        centroids = kmeans_plus_plus_initialization(self.data, self.k)
        cluster_assignments = assign_clusters(self.data, centroids, self.spatial_weight, self.num_spatial_dims)
        self.assertEqual(cluster_assignments.size(), (self.data.size(0),))

    def test_update_centroids(self):
        centroids = kmeans_plus_plus_initialization(self.data, self.k)
        cluster_assignments = assign_clusters(self.data, centroids, self.spatial_weight, self.num_spatial_dims)
        new_centroids = update_centroids(self.data, cluster_assignments, self.k)
        self.assertEqual(new_centroids.size(), (self.k, self.data.size(1)))

    def test_has_converged(self):
        centroids1 = kmeans_plus_plus_initialization(self.data, self.k)
        centroids2 = centroids1.clone()
        self.assertTrue(has_converged(centroids1, centroids2))
        centroids2[0] += 1.0  # Modify one centroid
        self.assertFalse(has_converged(centroids1, centroids2))

    def test_kmeans(self):
        centroids, cluster_assignments = spatial_kmeans(self.data, self.k, self.spatial_weight, self.num_spatial_dims)
        self.assertEqual(centroids.size(), (self.k, self.data.size(1)))
        self.assertEqual(cluster_assignments.size(), (self.data.size(0),))

if __name__ == '__main__':
    unittest.main()

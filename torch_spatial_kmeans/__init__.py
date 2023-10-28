"""__init__.py

torch_spatial_kmeans package

Provides a PyTorch implementation of spatial K-means clustering.

Examples usage of `spatial_kmeans`:

    >>> import torch
    >>> from torch_spatial_kmeans import spatial_kmeans
    >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
    >>> centroids, cluster_assignments = spatial_kmeans(data, k=2, spatial_weight=0.5, num_spatial_dims=2)
    >>> centroids
    tensor([[1, 2, 3],
            [4, 5, 6]])
    >>> cluster_assignments
    tensor([0, 1])
"""
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to compute the combined distance between two sets of points
def custom_kernel(X, Y, spatial_weight, num_spatial_dims):
    """Compute the combined distance between two sets of points.
    
    Args:
        X (torch.Tensor): The first set of points. Shape (N, D).
        Y (torch.Tensor): The second set of points. Shape (M, D).
        spatial_weight (float): The weight to apply to the spatial distance.
        num_spatial_dims (int): The number of spatial dimensions.
        
    Returns:
        torch.Tensor: The combined distance between the two sets of points.

    Examples:

        >>> X = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> Y = torch.tensor([[7, 8, 9], [10, 11, 12]])
        >>> custom_kernel(X, Y, spatial_weight=0.5, num_spatial_dims=2)
        tensor([[ 54.0000,  99.0000],
                [ 99.0000, 144.0000]])
    """
    # Split the data into features and spatial components
    features_X, spatial_X = X[:, :-num_spatial_dims], X[:, -num_spatial_dims:]
    features_Y, spatial_Y = Y[:, :-num_spatial_dims], Y[:, -num_spatial_dims:]
    
    # Compute the squared distances between the features and spatial components
    feature_distances_sq = torch.cdist(features_X, features_Y, p=2).pow(2)
    spatial_distances_sq = torch.cdist(spatial_X, spatial_Y, p=2).pow(2)
    
    # Apply the spatial weight to the spatial distances
    weighted_spatial_distances = spatial_distances_sq * spatial_weight
    
    # Combine the feature and spatial distances
    combined_distances = feature_distances_sq + weighted_spatial_distances
    
    return combined_distances

# Function to assign clusters based on the current centroids
def assign_clusters(data, centroids, spatial_weight, num_spatial_dims):
    """Assign clusters based on the current centroids.
    
    Args:
        data (torch.Tensor): The data to cluster. Shape (N, D).
        centroids (torch.Tensor): The current centroids. Shape (K, D).
        spatial_weight (float): The weight to apply to the spatial distance.
        num_spatial_dims (int): The number of spatial dimensions.
        
    Returns:
        torch.Tensor: The cluster assignments for each data point. Shape (N,).
        
    Examples:
    
        >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> centroids = torch.tensor([[7, 8, 9], [10, 11, 12]])
        >>> assign_clusters(data, centroids, spatial_weight=0.5, num_spatial_dims=2)
        tensor([0, 1])
    """

    # Compute the combined distances between the data and centroids
    combined_distances = custom_kernel(data, centroids, spatial_weight, num_spatial_dims)

    # Assign each data point to the closest centroid using the combined distances
    cluster_assignments = torch.argmin(combined_distances, dim=1)

    return cluster_assignments

# Function to initialize centroids using the k-means++ algorithm
def kmeans_plus_plus_initialization(data, k):
    """Initialize centroids using the k-means++ algorithm.
    
    Args:
        data (torch.Tensor): The data to cluster. Shape (N, D).
        k (int): The number of clusters.
        
    Returns:
        torch.Tensor: The initial centroids. Shape (K, D).
        
    Examples:
    
        >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> kmeans_plus_plus_initialization(data, k=2)
        tensor([[1, 2, 3],
                [4, 5, 6]])
    """

    # Initialize the first centroid randomly
    first_centroid_index = torch.randint(0, data.size(0), (1,)).item()
    centroids = data[first_centroid_index].unsqueeze(0)

    # Initialize the remaining centroids using the k-means++ algorithm
    for _ in range(k - 1):
        distances = torch.min(torch.cdist(data, centroids, p=2), dim=1)[0]

        # Compute the probability of each data point being selected as a centroid
        probabilities = distances.pow(2) / distances.pow(2).sum()

        # Select the next centroid
        centroid_index = torch.multinomial(probabilities, 1).item()

        # Add the next centroid to the list of centroids
        centroids = torch.cat([centroids, data[centroid_index].unsqueeze(0)], dim=0)
    
    return centroids

# Function to update centroids based on the current cluster assignments
def update_centroids(data, cluster_assignments, k):
    """Update centroids based on the current cluster assignments.
    
    Args:
        data (torch.Tensor): The data to cluster. Shape (N, D).
        cluster_assignments (torch.Tensor): The current cluster assignments. Shape (N,).
        k (int): The number of clusters.
        
    Returns:
        torch.Tensor: The updated centroids. Shape (K, D).
        
    Examples:
            
        >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> cluster_assignments = torch.tensor([0, 1])
        >>> update_centroids(data, cluster_assignments, k=2)
        tensor([[1, 2, 3],
                [4, 5, 6]])
    """
    new_centroids = []

    # Compute the new centroid for each cluster
    for i in range(k):
        # Get the data points assigned to the current cluster
        cluster_data = data[cluster_assignments == i]

        # Compute the mean of the data points assigned to the current cluster
        cluster_mean = cluster_data.mean(dim=0)\
            if len(cluster_data) > 0 else torch.tensor(float('nan'))
        
        # If there are no data points assigned to the current cluster, re-initialize the centroid
        if torch.isnan(cluster_mean).any():
            cluster_mean = data[torch.randint(0, data.size(0), (1,)).item()]

        new_centroids.append(cluster_mean)
    return torch.stack(new_centroids)

# Function to check for convergence
def has_converged(old_centroids, new_centroids, tol=1e-4):
    """Check for convergence.
    
    Args:
        old_centroids (torch.Tensor): The previous centroids. Shape (K, D).
        new_centroids (torch.Tensor): The updated centroids. Shape (K, D).
        tol (float): The tolerance for convergence.
        
    Returns:
        bool: Whether the centroids have converged.
        
    Examples:
    
        >>> old_centroids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> new_centroids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        >>> has_converged(old_centroids, new_centroids, tol=1e-4)
        True"""
    return torch.norm(old_centroids - new_centroids) < tol


def spatial_kmeans(data, k, spatial_weight, num_spatial_dims, max_iters=100):
    """Perform spatial K-means clustering.

    Args:
        data (torch.Tensor): The data to cluster. Shape (N, D).
        k (int): The number of clusters.
        spatial_weight (float): The weight to apply to the spatial distance.
        num_spatial_dims (int): The number of spatial dimensions.
        max_iters (int): The maximum number of iterations.

    Returns:
        torch.Tensor: The centroids. Shape (K, D).
        torch.Tensor: The cluster assignments for each data point. Shape (N,).

    Examples:
    
            >>> data = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> centroids, cluster_assignments = spatial_kmeans(data, k=2, spatial_weight=0.5, num_spatial_dims=2)
            >>> centroids
            tensor([[1, 2, 3],
                    [4, 5, 6]])
            >>> cluster_assignments
            tensor([0, 1])
    """
    # Move the data to the device
    data = data.float().to(device)

    # Initialize the centroids
    centroids = kmeans_plus_plus_initialization(data, k)

    iters = 0
    converged = False

    # Iterate until convergence or the maximum number of iterations is reached
    while not converged and iters < max_iters:
        old_centroids = centroids

        # Assign clusters based on the current centroids
        cluster_assignments = assign_clusters(data, centroids, spatial_weight, num_spatial_dims)
        
        # Update the centroids based on the current cluster assignments
        centroids = update_centroids(data, cluster_assignments, k)

        # Check for convergence
        converged = has_converged(old_centroids, centroids)
        iters += 1
    return centroids.cpu(), cluster_assignments.cpu()


if __name__ == "__main__":
    pass
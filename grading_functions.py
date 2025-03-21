import numpy as np
from sklearn.metrics import pairwise_distances

# Unsupervised Methods

def cosine_similarity(x, c):
    # Calculates the cosine similarity between two vectors
    x = x.flatten()  # Ensure x is a 1D array
    c = c.flatten()  # Ensure c is a 1D array
    dot_product = np.dot(x, c)
    norm_x = np.linalg.norm(x)
    norm_c = np.linalg.norm(c)
    similarity = dot_product / (norm_x * norm_c)
    return similarity

def knn_cluster_grade(observation, cluster_data, k=2):
    # Calculates the average distance to the k-nearest neighbors in the cluster
    distances = pairwise_distances(cluster_data, observation.reshape(1, -1), metric='euclidean').flatten()
    sorted_distances = np.sort(distances)
    k_nearest_distances = sorted_distances[:k]
    grade = np.mean(k_nearest_distances)
    return grade

def cluster_membership_strength(observation, cluster_observations):
    # Calculates the inverse of the average distance to the cluster
    distances = pairwise_distances(cluster_observations, observation.reshape(1, -1), metric='euclidean').flatten()
    avg_distance = np.mean(distances)
    membership_degree = 1 / (avg_distance + np.finfo(float).eps)
    return membership_degree

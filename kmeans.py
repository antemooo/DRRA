# X: data matrix of size (n_samples,n_features)
# n_clusters: number of clusters
# output 1: labels of X with size (n_samples,)
# output 2: centroids of clusters
import numpy as np
from tqdm import tqdm
from euclidean_distance import euclidean_vectorized
from euclidean_distance import euclidean_vectorized2


def kmeans(X, n_clusters):
    # initialize labels and prev_labels. prev_labels will be compared with labels to check if the stopping condition
    # have been reached.
    prev_labels = np.zeros(X.shape[0])
    labels = np.zeros(X.shape[0])

    # init random indices
    indices = np.random.permutation(X.shape[0])[:n_clusters]
    # indices = np.random.choice(X.shape[0], n_clusters, replace=False)  # np.random.permutation(X.shape[0])[:n_clusters]

    # assign centroids using the indices
    centroids = X[indices]

    # the interative algorithm goes here
    while (True):
        # calculate the distances to the centroids
        distances = euclidean_vectorized(X, centroids)
        # assign labels
        labels = np.argmin(distances, axis=1)
        # print("labels shape is", labels.shape)
        # reshape labels for the further use
        labels_new = np.squeeze(np.asarray(labels))
        # print("labels_new shape is", labels_new.shape)
        # stopping condition
        # YOUR CODE GOES HERE
        if np.array_equal(labels, prev_labels):
            # if np.sum(labels != prev_labels) == 0:
            break

        # calculate new centroids
        for cluster_indx in range(centroids.shape[0]):
            members = X[labels_new == cluster_indx]
            centroids[cluster_indx, :] = np.mean(members, axis=0)

        # keep the labels for next round's usage
        prev_labels = np.argmin(distances, axis=1)

    return labels, centroids

def kmeans2(X, n_clusters):
    # initialize labels and prev_labels. prev_labels will be compared with labels to check if the stopping condition
    # have been reached.
    prev_labels = np.zeros(X.shape[0])
    labels = np.zeros(X.shape[0])

    # init random indices
    indices = np.random.permutation(X.shape[0])[:n_clusters]
    # indices = np.random.choice(X.shape[0], n_clusters, replace=False)  # np.random.permutation(X.shape[0])[:n_clusters]

    # assign centroids using the indices
    centroids = X[indices]

    # the interative algorithm goes here
    while (True):
        # calculate the distances to the centroids
        distances = euclidean_vectorized2(X, centroids)
        # assign labels
        labels = np.argmin(distances, axis=1)
        # print("labels shape is", labels.shape)
        # reshape labels for the further use
        labels_new = np.squeeze(np.asarray(labels))
        # print("labels_new shape is", labels_new.shape)
        # stopping condition
        if np.array_equal(labels, prev_labels):
            # if np.sum(labels != prev_labels) == 0:
            break

        # calculate new centroids
        for cluster_indx in range(centroids.shape[0]):
            members = X[labels_new == cluster_indx]
            centroids[cluster_indx, :] = np.mean(members, axis=0)

        # keep the labels for next round's usage
        prev_labels = np.argmin(distances, axis=1)

    return labels, centroids

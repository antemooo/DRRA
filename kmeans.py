# X: data matrix of size (n_samples,n_features)
# n_clusters: number of clusters
# output 1: labels of X with size (n_samples,)
# output 2: centroids of clusters
import numpy as np

from euclidean_distance import euclidean_vectorized


def kmeans(X, n_clusters):
    # initialize labels and prev_labels. prev_labels will be compared with labels to check if the stopping condition
    # have been reached.
    prev_labels = np.zeros(X.shape[0])
    labels = np.zeros(X.shape[0])

    # init random indices
    # YOUR CODE GOES HERE
    indices = np.random.choice(X.shape[0], n_clusters, replace=False)  # np.random.permutation(X.shape[0])[:n_clusters]

    # assign centroids using the indices
    # YOUR CODE GOES HERE
    centroids = X[indices]

    # the interative algorithm goes here
    while (True):
        # calculate the distances to the centroids
        # YOUR CODE GOES HERE
        distances = euclidean_vectorized(X, centroids)
        # assign labels
        # YOUR CODE GOES HERE
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
        # YOUR CODE GOES HERE
        for cluster_indx in range(centroids.shape[0]):
            members = X[labels_new == cluster_indx]
            centroids[cluster_indx, :] = np.mean(members, axis=0)

        # keep the labels for next round's usage
        # YOUR CODE GOES HERE
        prev_labels = np.argmin(distances, axis=1)

    return labels, centroids

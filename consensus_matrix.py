import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans


# Section 2.2: Consensus Matrix:
def consensus_cluster(X, cluster_from, cluster_to):
    consensus_matrix = np.zeros((X.shape[0], X.shape[0]))
    clusters = list([])
    # print("consensus matrix is", consensus_matrix.shape)
    for k in range(cluster_from, cluster_to + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        cluster_labels = kmeans.labels_
        cluster_labels_new = np.squeeze(np.asarray(cluster_labels))
        clusters.append(cluster_labels_new)
    for i in tqdm(range(0, X.shape[0])):
        for j in range(0, X.shape[0]):
            count = 0
            for cluster in clusters:
                if cluster[i] == cluster[j]:
                    count += 1
            consensus_matrix[i, j] = int(count)
    return consensus_matrix


def consensus_noise(consensus_matrix):
    consensus_matrix_new = consensus_matrix / consensus_matrix[0, 0]
    consensus_matrix_new[consensus_matrix_new < 0.1] = 0
    noise = np.zeros(consensus_matrix.shape[0])
    threshold = consensus_matrix_new.sum() / consensus_matrix_new.shape[0]
    for i in range(consensus_matrix_new.shape[0]):
        a = sum(consensus_matrix_new[i, :])
        if a < threshold:
            noise[i] = 1
    return noise


def clean_tweet_noise(tweets, noise):
    if len(tweets) == len(noise):
        tweets_new = list([])
        for i in range(len(noise)):
            if noise[i] == 0:
                tweets_new.append(tweets[i])
        return tweets_new

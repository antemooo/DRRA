import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans


# This function is to get the consensus matrix of different KMeans with different K.
# The value of each element is how many times the value is same
# The "cluster_from" stands for the least k and the "cluster_to" stands for the largest k
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


# This function is to find the noise of the consensus matrix.
# if the value is less than the threshold, its position will be set to 1 in the noise vector.
# otherwise its position is set to 0 in the noise vector.
# the result is a noise vectore that indicates the positions of noise.
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


# Clean up the noise in the tweets.
# excluding all the values that are set to 1 in the noise vector.
def clean_tweet_noise(tweets, noise):
    if len(tweets) == len(noise):
        tweets_new = list([])
        for i in range(len(noise)):
            if noise[i] == 0:
                tweets_new.append(tweets[i])
        return tweets_new

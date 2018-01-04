import csv
import pandas as pd
from tqdm import tqdm
# This file is to cluster all the tweets

# This function is to make a cluster of tweets and generate a CSV file, the tweets_clean stands for the old_data we have
# the cluster_number stands for the old_data cluster number we need and the cluster_labels stands for the labels we got,
# it's a list of clusters of each tweet, the name is the name of this file
def generate_cluster(tweets_clean, cluster_number, cluster_labels, name):
    # all the tweets needs to be write into 9 files
    all_tweets_cluster = list([])
    for x in range(cluster_number):
        cluster_tweets = list([])
        all_tweets_cluster.append(cluster_tweets)

    for cluster_index in range(len(cluster_labels)):
        cluster = cluster_labels[cluster_index]
        tweet_info = tweets_clean[cluster_index]
        tweet_info_new = []
        for b in tweet_info:
            tweet_info_new.append(b)
        all_tweets_cluster[cluster - 1].append(tweet_info_new)
    i = 1
    cluster_group = list([])
    for a in tqdm(all_tweets_cluster):
        cluster_name = name + ': cluster_' + str(i) + '.csv'
        i = i + 1
        cluster_group.append(a)
        with open(cluster_name, 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(a)
    return cluster_group

# This function is to generate a CSV file that has name of each cluster(the common word) and the ID of each tweet
def generate_node(most_words, cluster_label, name):
    id = range(1, len(cluster_label) + 1)
    label = []
    for i in tqdm(cluster_label):
        label.append(most_words[i - 1])  # add correspond word into the list
    df = pd.DataFrame({'ID': id, 'Label': label})
    df.to_csv(name + '.csv', mode='w', index=False)
    return df

# This function is to generate the edges(from source node to target node in the same cluster), it uses the consensus
# function and the threshold stands for the value in the consensus matrix of KMeans, and if it's big enough the 2 nodes
# will be saved
def generate_edge(name, consensus_matrix, cluster_label, threshold):
    index_list = list([])
    for a in range(9):
        same_cluster = [index for index in range(len(cluster_label)) if cluster_label[index] == a]
        index_list.append(same_cluster)
    source = list([])
    target = list([])
    for i in tqdm(range(consensus_matrix.shape[0])):
        for j in range(consensus_matrix.shape[1]):
            if (consensus_matrix[i, j] > threshold) and (i < j):
                for clusters in index_list:
                    if i in clusters and j in clusters:
                        source.append(i + 1)
                        target.append(j + 1)

    df = pd.DataFrame({'Source': source, 'Target': target})
    df.to_csv(name + '.csv', mode='w', index=False)
    return df

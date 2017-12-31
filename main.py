import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from csv_helper import load_csv, load_csv2
from consensus_matrix import consensus_cluster, consensus_noise, clean_tweet_noise
from sklearn.cluster import KMeans
from DBSCAN import DBSCAN_matrix,db_noise
from WordsFrequency import cluster_words
from cluster import generate_cluster, generate_node, generate_edge
from clean_tweet import tokenizate, clean_tweet, combine

# Method 1: clean the digits and signatures

tweets = load_csv("Tweets_2016London.csv")
tweets_tokenize = tokenizate(tweets)
tweet_clean = clean_tweet(tweets_tokenize)

# Method 2: get the data set for the further use:



# get the combined tweet content
tweets_combine = combine(tweet_clean)

# get the vectorizer of TF-IDF
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')

X = vectorizer.fit_transform(tweets_combine)

X_new = X.todense()

consensus_matrix = consensus_cluster(X_new, 2, 10)

noise_kmeans = consensus_noise(consensus_matrix)

tweets_kmeans_clean = clean_tweet_noise(tweet_clean, noise_kmeans)

tweets_combine_kmeans = combine(tweets_kmeans_clean)

X_kmeans = vectorizer.fit_transform(tweets_combine_kmeans)

X_kmeans_new = X_kmeans.todense()

consensus_matrix1 = consensus_cluster(X_kmeans_new, 2, 12)

kmeans_new = KMeans(n_clusters=9, random_state=0).fit(consensus_matrix1)
cluster_labels_kmeans = kmeans_new.labels_

# print(cluster_labels_kmeans)

########################################################################################
# DBSCAN

db = DBSCAN_matrix(X_new,5,15)

db_noise = db_noise(db)

tweets_db_clean = clean_tweet_noise(tweet_clean, db_noise)

tweets_combine_db = combine(tweets_db_clean)

X_db = vectorizer.fit_transform(tweets_combine_db)

X_db_new = X_db.todense()

consensus_matrix2 = consensus_cluster(X_db_new, 2, 12)

db_new = KMeans(n_clusters=9, random_state=0).fit(consensus_matrix2)
cluster_labels_db = kmeans_new.labels_

#########################################################################################

cluster_group_kmeans = generate_cluster(tweets_kmeans_clean, 9, cluster_labels_kmeans,'kmeans')
cluster_group_db = generate_cluster(tweets_kmeans_clean, 9, cluster_labels_kmeans,'dbscan')

# the words that appeared mostly in each cluster group


most_words_kmeans = cluster_words(cluster_group_kmeans)
most_words_db = cluster_words(cluster_group_db)


node_kmeans = generate_node(most_words_kmeans, cluster_labels_kmeans, 'kmeans_node')
edge_kmeans = generate_edge('kmeans_edge_new', consensus_matrix1, cluster_labels_kmeans)

node_db = generate_node(most_words_db, cluster_labels_kmeans, 'db_scan')
edge_db = generate_edge('kmeans_edge_new', consensus_matrix2, cluster_labels_db)
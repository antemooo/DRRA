import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from csv_helper import load_csv, load_csv2
from consensus_matrix import consensus_cluster, consensus_noise, clean_tweet_noise
from sklearn.cluster import KMeans
from DBSCAN import DBSCAN_matrix, db_noise
from WordsFrequency import cluster_words
from cluster import generate_cluster, generate_node, generate_edge
from clean_tweet import tokenizate, clean_tweet, combine

# Method 1: clean the digits and punctuations in the tweets:
# In this part, what we need to achieve is to clean all the punctuations, digits and emjo in the tweet. And what I also
# need to do is to clean the 'http' in each tweet:

# load the csv file via use the code has been provided
tweets = load_csv("Tweets_2016London.csv")
# make a tokenize on the code, this method can be found on clean_tweet.py
tweets_tokenize = tokenizate(tweets)
# clean up the digits, punctuations and https, this method can be found on clean_tweet.py
tweet_clean = clean_tweet(tweets_tokenize)
########################################################################################################################


# Method 2: get the data set for the further use:
# In this part, we need to use 2 methods separately to achieve noisy clean, so it will give us 2 results: one is
# achieved via kmeans and consensus matrix another method is achieved via DBSCAN:

# At first I need to achieve TF-IDF method to get the percent of each words and can be used for the further use
# get the combined tweet content into one list, this method is to get all the words in a file for TF-IDF, the method can
# be found on clean_tweet.py
tweets_combine = combine(tweet_clean)
# get the vectorizer of TF-IDF, this method is from sklearn
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
# Attach the vectorizer to tweets files and we can get the sparse matrix of each tweets
X = vectorizer.fit_transform(tweets_combine)
# Transfer the sparse matrix to normal matrix
X_new = X.todense()

##################################
# Kmeans

# This method is to get teh consensus matrix via running kmeans 9 times, the k is from 2 to 10
consensus_matrix = consensus_cluster(X_new, 2, 10)
# This step is to clean the noise inside the consensus matrix via using the idea in the instruction file on
# point carré, the method can be found on consensus_matrix.py the result is a list with 0 and 1s, 1 means this tweet is
# a noise and should be removed, this method can be found on consensus_matrix.py
noise_kmeans = consensus_noise(consensus_matrix)
# Using the noise we get and attach it into the tweets which has been cleaned non-stop words and we get a clean tweet
# that has been removed noise, this method can be found on consensus_matrix.py
tweets_kmeans_clean = clean_tweet_noise(tweet_clean, noise_kmeans)

##################################
# DBSCAN

# This method is to get teh DBSCAN matrix via running DBSCAN 11 times, the number of elements in the range is from
# 5 to 15
db = DBSCAN_matrix(X_new, 5, 15)
# This step is to clean the noise inside the consensus matrix via using the idea in the instruction file on
# point carré, the method can be found on consensus_matrix.py the result is a list with 0 and 1s, 1 means this tweet is
# a noise and should be removed, this method can be found on consensus_matrix.py
db_noise = db_noise(db)
# Using the noise we get and attach it into the tweets which has been cleaned non-stop words and we get a clean tweet
# that has been removed noise, this method can be found on consensus_matrix.py
tweets_db_clean = clean_tweet_noise(tweet_clean, db_noise)

########################################################################################################################


# Method 3: cluster the tweets
# In this part, we used the results we got from the tweet data we cleaned, we generated 2 consensus matrix that from the
# data we got via KMeans with consensus matrix and DBSCAN:

# Get the combined tweets for the TF-IDF, the tweets are the noise clean result of Kmeans and consensus matrix method
tweets_combine_kmeans = combine(tweets_kmeans_clean)
# Get the sparse matrix of kmeans method
X_kmeans = vectorizer.fit_transform(tweets_combine_kmeans)
# Transfer the sparse matrix to normal matrix
X_kmeans_new = X_kmeans.todense()
# Get the consensus matrix by using kmeans which k from 2 to 12, this consensus matrix is from the data via kmeans
consensus_matrix1 = consensus_cluster(X_kmeans_new, 2, 12)
# Get the combined tweets for the TF-IDF, the tweets are the noise clean result of DBSCAN matrix method
tweets_combine_db = combine(tweets_db_clean)
# Get the sparse matrix of DBSCAN method
X_db = vectorizer.fit_transform(tweets_combine_db)
# Transfer the sparse matrix to normal matrix
X_db_new = X_db.todense()
# Get the consensus matrix by using KMeans which k from 2 to 12, this consensus matrix is from the data via DBSCAN
consensus_matrix2 = consensus_cluster(X_db_new, 2, 12)
# Generate a KMeans method for the first Consensus Matrix(the KMean and Consensus Matrix one), the KMeans is
# from sklearn
kmeans_new = KMeans(n_clusters=9, random_state=0).fit(consensus_matrix1)
# The label stands for the cluster of each tweets, which is from 1 to 9
cluster_labels_kmeans = kmeans_new.labels_
# Generate a KMeans method for the second Consensus Matrix(the DBSCAN one), the KMeans is from sklearn
db_new = KMeans(n_clusters=9, random_state=0).fit(consensus_matrix2)
# The label stands for the cluster of each tweets, which is from 1 to 9
cluster_labels_db = db_new.labels_
# Generate the cluster group, which is clusters for each tweets, this method can be found on cluster.py, the results are
# 9 csv files, which saved the tweets that been clustered in a group
cluster_group_kmeans = generate_cluster(tweets_kmeans_clean, 9, cluster_labels_kmeans, 'kmeans')
cluster_group_db = generate_cluster(tweets_kmeans_clean, 9, cluster_labels_kmeans, 'dbscan')

# Find the words that appeared mostly in each cluster group, the method can be found on WordsFrequency.py
most_words_kmeans = cluster_words(cluster_group_kmeans)
most_words_db = cluster_words(cluster_group_db)
# Generate nodes and edges for 2 methods, the information are in 2 csv files, one saved nodes of each cluster, one saved
# edges that connected the nodes in a cluster together these files are used for Gephi
node_kmeans = generate_node(most_words_kmeans, cluster_labels_kmeans, 'kmeans_node')
edge_kmeans = generate_edge('kmeans_edge', consensus_matrix1, cluster_labels_kmeans, 8)
node_db = generate_node(most_words_db, cluster_labels_kmeans, 'dbscan_node')
edge_db = generate_edge('dbscan_edge', consensus_matrix2, cluster_labels_db, 8)

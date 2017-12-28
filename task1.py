# Part 1 data cleaning
# In this part, my target is to make process of my data
# To achieve my target, I need to write 3 methods below:
# 1. make tokenizing of the data: first separate the data into sentences and each sentence is a list of words
# 2. remove stop words in the each sentences
# 3. use stem on each word in each sentence
import csv

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from csv_helper import load_csv
from nltk.tokenize import TweetTokenizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
# from kmeans import kmeans
# from kmeans import kmeans2
import matplotlib.pyplot as plt

import re


def remove_urls(vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return vTEXT


tweets = load_csv("Tweets_2016London.csv")


def tokenizate(tweets):
    tknzer = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=True)
    tokens = list([])
    for tweet in tweets:
        tweet = remove_urls(tweet)
        token = tknzer.tokenize(tweet)
        if token is not None:
            tokens.append(token)
    return tokens


tweets_tokenize = tokenizate(tweets)


# print(tweets_tokenize)


def stop_words_removal(sentence):
    english_stopwords = stopwords.words('english')
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '-', '_',
                            '___', '__', '"', '/', '...', ",", "∞", "'", 'ö', '\\', '__']
    content = list([])

    for word in sentence:
        if (word.lower() not in english_stopwords) & (word not in english_punctuations):
            word = re.sub(r'[^\w]', '', word)  # remove signs in a word
            word = re.sub(r'\w*\d\w*', '', word).strip()  # remove all the words that contains digits
            word = word.encode('ascii', 'ignore').decode('ascii')
            if not word.isdigit():
                if word != '':
                    content.append(word)
    return content


def stemming(sentence):
    st = LancasterStemmer()
    content = list([])
    for word in sentence:
        new_word = st.stem(word)
        content.append(new_word)
    return content


# tokenize the original tweets information into few sentences
# tweets_tokenize = tokenizate(tweets)
# print(tweets_tokenize)
# call the final tweets tweet_clean
tweet_clean = list([])
for tweet in tweets_tokenize:
    # remove all the stop words in each tweet sentences
    tweet_svr = stop_words_removal(tweet)
    # stem each word in these tweet sentences
    tweet_s = stemming(tweet_svr)
    # add the clean tweets into the list
    tweet_clean.append(tweet_s)


# ############################################################################# #

# Part 2
# Section 1 imply the TF-IDF

# Combine single words in each tweet
def combine(clean_tweet):
    tweets_combine = []
    for tweet_list in clean_tweet:
        tweet_combine = ' '.join(tweet_list)
        tweets_combine.append(tweet_combine)
    return tweets_combine


# get the combined tweet content
tweets_combine = combine(tweet_clean)

# get the vectorizer of TF-IDF
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
# print(tweets_combine)
# print(tweets)
X = vectorizer.fit_transform(tweets_combine)

# print("n_samples: %d, n_features: %d" % X.shape)

# Task 2.1 imply the k-means algorithm



# print(X_new)
X_new = X.todense()


# print("X_new shape is:", X_new.shape)

# Section 2.1: k means
# cluster_labels, centroids = kmeans(X_new, 5)
# print(cluster_labels)
# print(centroids)

# labels = ["cluster_"+str(x) for x in range(5)]
# population = [np.sum(cluster_labels == x) for x in range(5)]
# y_pos = np.arange(len(labels))
# barlist = plt.bar(y_pos, population, align='center',width=0.3)
# plt.xticks(y_pos, labels)
# plt.ylabel('Number of examples')
# plt.title('Sklearn digits dataset.')
# plt.show()

#############

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
            consensus_matrix[i, j] = count
    return consensus_matrix


consensus_matrix = consensus_cluster(X_new, 2, 10)


# print(consensus_matrix)


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


noise = consensus_noise(consensus_matrix)

print(noise)


def clean_tweet_noise(tweets, noise):
    if len(tweets) == len(noise):
        tweets_new = list([])
        for i in range(len(noise)):
            if noise[i] == 0:
                tweets_new.append(tweets[i])
        return tweets_new


tweets_new_clean = clean_tweet_noise(tweet_clean, noise)
print(len(tweets_new_clean))


##########
# Section 2.3: DBSCAN

def DBSCAN_matrix(X, eps_from, eps_to):
    db_matrix = np.zeros((X.shape[0], eps_to - eps_from + 1))
    j = 0
    for eps in tqdm(range(eps_from, eps_to + 1)):
        db = DBSCAN(eps=1, min_samples=eps).fit(X)
        core_index = db.core_sample_indices_
        for x in core_index:
            db_matrix[x - 1, j] = 1
        j = j + 1
    return db_matrix


# db = DBSCAN(eps=1, min_samples=20).fit(X_new)
# print(len(db.core_sample_indices_), db.core_sample_indices_)
# print(db.labels_)
#
# db_matrix = DBSCAN_matrix(X_new, 5, 15)
# print(db_matrix)


def db_noise(db_matrix):
    noise = np.sum(np.square(db_matrix), axis=1) / db_matrix.shape[1]
    noise_new = list([])
    for i in noise:
        if i > 0:
            noise_new.append(1)
        else:
            noise_new.append(0)

    return noise_new


# noise2 = db_noise(db_matrix)

# print(noise2)
###########################################################################

tweets_combine_new = combine(tweets_new_clean)
# print(tweets_combine_new)

# the new data cleaned via kmeans and consensus matrix
X1 = vectorizer.fit_transform(tweets_combine_new)
# print(X1)
X1_new = X1.todense()
# print(X1_new)

# print(X1_new.shape)

consensus_matrix1 = consensus_cluster(X1_new, 2, 12)
#
# print(consensus_matrix1)
#
kmeans1 = KMeans(n_clusters=9, random_state=0).fit(X1_new)
cluster_labels_new = kmeans1.labels_


print(len(cluster_labels_new))




#############################################################################

# Part 4

def generate_cluster(tweets_clean, cluster_number, cluster_labels):
    # all the tweets needs to be write into 9 files
    all_tweets_cluster = list([])
    for x in range(cluster_number):
        cluster_tweets = list([])
        all_tweets_cluster.append(cluster_tweets)
    #
    for cluster_index in range(len(cluster_labels)):
        cluster = cluster_labels[cluster_index]
        tweet_info = tweets_clean[cluster_index]
        tweet_info_new = []
        for b in tweet_info:
            tweet_info_new.append(b)
        all_tweets_cluster[cluster - 1].append(tweet_info_new)
    i = 1
    for a in all_tweets_cluster:
        cluster_name = 'cluster_' + str(i) + '.csv'
        i = i + 1
        with open(cluster_name, 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerows(a)


generate_cluster(tweets_new_clean, 9, cluster_labels_new)



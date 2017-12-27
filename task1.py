# Part 1 data cleaning
# In this part, my target is to make process of my data
# To achieve my target, I need to write 3 methods below:
# 1. make tokenizing of the data: first separate the data into sentences and each sentence is a list of words
# 2. remove stop words in the each sentences
# 3. use stem on each word in each sentence
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from csv_helper import load_csv
from nltk.tokenize import TweetTokenizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
import scipy.sparse.linalg
from sklearn.cluster import KMeans
from kmeans import kmeans
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
                            '___', '__', '"', '/', '...', ",", "∞", "'", 'ö']
    content = list([])

    for word in sentence:
        if (word.lower() not in english_stopwords) & (word not in english_punctuations):
            word = re.sub(r'[^\w]', '', word)  # remove signs in a word
            word = re.sub(r'\w*\d\w*', '', word).strip()  # remove all the words that contains digits
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
print(tweets_combine)
# print(tweets)
X = vectorizer.fit_transform(tweets_combine)

# print("n_samples: %d, n_features: %d" % X.shape)

# Task 2.1 imply the k-means algorithm


# X_new = scipy.sparse.linalg.inv(X)
# print(X_new)
X_new = X.todense()


# print("X_new shape is:", X_new.shape)

# Section 2.1: k means
# cluster_labels, centroids = kmeans(X_new, num_clusters)
# print(cluster_labels)
# print(centroids)

# labels = ["cluster_"+str(x) for x in range(num_clusters)]
# population = [np.sum(cluster_labels == x) for x in range(num_clusters)]
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
    print("consensus matrix is", consensus_matrix.shape)
    for k in range(cluster_from, cluster_to + 1):
        cluster_labels, centroids = kmeans(X, k)
        cluster_labels_new = np.squeeze(np.asarray(cluster_labels))
        clusters.append(cluster_labels_new)
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[0]):
            count = 0
            for cluster in clusters:
                if cluster[i] == cluster[j]:
                    count += 1
            consensus_matrix[i, j] = count
    return consensus_matrix


consensus_matrix = consensus_cluster(X_new, 2, 10)


# print(consensus_matrix)


def consensus_noise_noise(consensus_matrix):
    consensus_matrix_new = consensus_matrix / consensus_matrix[0, 0]
    consensus_matrix_new[consensus_matrix_new < 0.1] = 0
    noise = np.zeros(consensus_matrix.shape[0])
    threshold = consensus_matrix_new.sum() / consensus_matrix_new.shape[0]
    for i in range(consensus_matrix_new.shape[0]):
        a = sum(consensus_matrix_new[i, :])
        if a < threshold:
            noise[i] = 1
    return noise


noise = consensus_noise_noise(consensus_matrix)


# print(noise)


def clean_tweet_noise(tweets, noise):
    if len(tweets) == len(noise):
        tweets_new = list([])
        for i in range(len(noise)):
            if noise[i] == 0:
                tweets_new.append(tweets[i])
        return tweets_new


tweets_new_clean = clean_tweet_noise(tweet_clean, noise)
# print(len(tweets_new_clean))

############
tweets_combine_new = combine(tweets_new_clean)
# print(tweets_combine_new)

# get the vectorizer of TF-IDF
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')

# the new data cleaned via kmeans and consensus matrix
X1 = vectorizer.fit_transform(tweets_combine_new)
# print(X1)
X1_new = X1.todense()

consensus_matrix1 = consensus_cluster(X1_new, 2, 12)

print(consensus_matrix1)


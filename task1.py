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

X = vectorizer.fit_transform(tweets)

# print("n_samples: %d, n_features: %d" % X.shape)

# Task 2.1 imply the k-means algorithm


num_clusters = 5
# X_new = scipy.sparse.linalg.inv(X)
# print(X_new)
X_new = X.todense()

# print("X_new shape is:", X_new.shape)
cluster_labels, centroids = kmeans(X_new, num_clusters)
# print(cluster_labels)
# print(centroids)

labels = ["cluster_"+str(x) for x in range(num_clusters)]
population = [np.sum(cluster_labels == x) for x in range(num_clusters)]
y_pos = np.arange(len(labels))
barlist = plt.bar(y_pos, population, align='center',width=0.3)
plt.xticks(y_pos, labels)
plt.ylabel('Number of examples')
plt.title('Sklearn digits dataset.')
plt.show()

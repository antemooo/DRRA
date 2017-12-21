# Part 1 data cleaning
# In this part, my target is to make process of my data
# To achieve my target, I need to write 3 methods below:
# 1. make tokenizing of the data: first separate the data into sentences and each sentence is a list of words
# 2. remove stop words in the each sentences
# 3. use stem on each word in each sentence
from sklearn.feature_extraction.text import TfidfVectorizer
from csv_helper import load_csv
from nltk.tokenize import TweetTokenizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from optparse import OptionParser
import re


def remove_urls(vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return (vTEXT)


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


def stopWordsRemoval(sentence):
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
    tweet_svr = stopWordsRemoval(tweet)
    # stem each word in these tweet sentences
    tweet_s = stemming(tweet_svr)
    # add the clean tweets into the list
    tweet_clean.append(tweet_s)

print(tweet_clean)

# ############################################################################# #

# Part 2
# Section 1 imply the TF-IDF

tweets_clean = list([])
for tweet in tweet_clean:
    tweet = ''.join(tweet)
    tweet_clean += tweet


tfidf = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
X = tfidf.fit_transform(tweets_clean)
print(X)
# print(X)
# Section 2 imply the k-means algorithm

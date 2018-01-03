from csv_helper import load_csv, load_csv2
from nltk.tokenize import TweetTokenizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
import re


# This file contains all the methods that clean the useless information inside a tweet:

# This function is to remove the URLs inside a line of words
def remove_urls(vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return vTEXT

# Tokenizate words in a Tweet, which will return a list which contains lists that contain words in a sentence
def tokenizate(tweets):
    tknzer = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=True)
    tokens = list([])
    for tweet in tweets:
        tweet = remove_urls(tweet)
        token = tknzer.tokenize(tweet)
        if token is not None:
            tokens.append(token)
    return tokens

# Remove all the stop words which is not useful, and also remove the punctuations
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

# This function is to stem all the words, eg: loving and love can be stemmed as lov
def stemming(sentence):
    st = LancasterStemmer()
    content = list([])
    for word in sentence:
        new_word = st.stem(word)
        content.append(new_word)
    return content


# tokenize the original tweets information into few sentences
# call the final tweets tweet_clean
def clean_tweet(tweets_tokenize):
    tweet_clean = list([])
    for tweet in tweets_tokenize:
        # remove all the stop words in each tweet sentences
        tweet_svr = stop_words_removal(tweet)
        # stem each word in these tweet sentences
        tweet_s = stemming(tweet_svr)
        # add the clean tweets into the list
        if len(tweet_s) != 0:  # to judge is this tweet empty?
            tweet_clean.append(tweet_s)
    return tweet_clean


# Combine single words in each tweet, which will return a list that contains all the words in a tweet, this function is
# for TF-IDF
def combine(clean_tweet):
    tweets_combine = []
    for tweet_list in clean_tweet:
        tweet_combine = ' '.join(tweet_list)
        tweets_combine.append(tweet_combine)
    return tweets_combine

# This file contains 3 methods, and the cluster_words will be used in the main.py to find the words
# appeared mostly in a CSV file

# Count every words in a tweet, and the result is a dictionary like {words: count number}
def countWords(tweets):
    wordDict = {}
    for line in tweets:
        for word in line:
            if word in wordDict:
                wordDict[word] += 1
            else:
                wordDict[word] = 1
    return wordDict

# Find the most common word in a dictionary (the result of count words function)
def findMostCommon(charDict):
    mostFreq = ''
    mostFreqCount = 0
    for k in charDict:
        if charDict[k] > mostFreqCount:
            mostFreqCount = charDict[k]
            mostFreq = k
    return mostFreq

# Find the most common words in a cluster
# Takes a cluster group
# for each cluster find the most common word
# returns a list of common words in a cluster group
def cluster_words(cluster_group):
    most_words = list([])

    for cluster in cluster_group:
        common_word = findMostCommon(countWords(cluster))
        most_words.append(common_word)
    return most_words

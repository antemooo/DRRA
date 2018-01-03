# This file contains 3 methods, and the cluster_words will be used for the main.py, which is used to find the words
# appeared mostly in a CSV file

# Count every words in a tweet, and the result looks like {words: count number}
def countWords(tweets):
    wordDict = {}
    for line in tweets:
        for word in line:
            if word in wordDict:
                wordDict[word] += 1
            else:
                wordDict[word] = 1
    return wordDict

# Find the word most common in the result from count words function
def findMostCommon(charDict):
    mostFreq = ''
    mostFreqCount = 0
    for k in charDict:
        if charDict[k] > mostFreqCount:
            mostFreqCount = charDict[k]
            mostFreq = k
    return mostFreq

# Find the most common words in a cluster
def cluster_words(cluster_group):
    most_words = list([])

    for cluster in cluster_group:
        common_word = findMostCommon(countWords(cluster))
        most_words.append(common_word)
    return most_words

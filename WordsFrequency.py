# This file is to achieve the rest targets

def countWords(tweets):
    wordDict = {}
    for line in tweets:
        for word in line:
            if word in wordDict:
                wordDict[word] += 1
            else:
                wordDict[word] = 1
    return wordDict


def findMostCommon(charDict):
    mostFreq = ''
    mostFreqCount = 0
    for k in charDict:
        if charDict[k] > mostFreqCount:
            mostFreqCount = charDict[k]
            mostFreq = k
    return mostFreq


def cluster_words(cluster_group):
    most_words = list([])
    for cluster in cluster_group:
        common_word = findMostCommon(countWords(cluster))
        most_words.append(common_word)
    return most_words

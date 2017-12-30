# This file is to achieve the rest targets

from csv_helper import load_csv2

cluster1 = load_csv2('cluster_1.csv')
cluster2 = load_csv2('cluster_2.csv')
cluster3 = load_csv2('cluster_3.csv')
cluster4 = load_csv2('cluster_4.csv')
cluster5 = load_csv2('cluster_5.csv')
cluster6 = load_csv2('cluster_6.csv')
cluster7 = load_csv2('cluster_7.csv')
cluster8 = load_csv2('cluster_8.csv')
cluster9 = load_csv2('cluster_9.csv')


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


most_cluster1 = findMostCommon(countWords(cluster1))
most_cluster2 = findMostCommon(countWords(cluster2))
most_cluster3 = findMostCommon(countWords(cluster3))
most_cluster4 = findMostCommon(countWords(cluster4))
most_cluster5 = findMostCommon(countWords(cluster5))
most_cluster6 = findMostCommon(countWords(cluster6))
most_cluster7 = findMostCommon(countWords(cluster7))
most_cluster8 = findMostCommon(countWords(cluster8))
most_cluster9 = findMostCommon(countWords(cluster9))

print(most_cluster1,most_cluster2,most_cluster3,most_cluster4,most_cluster5,most_cluster6,most_cluster7,most_cluster8,most_cluster9)


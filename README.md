# DRRA
Data representation and reduction analysis

Copyright ©VUB - Data representation and reduction analysis course 2017.

**Authors:** 
- **Ahmed K. A Abdullah @github/antemooo.**
- **Rencong Tang @github/rencongtang.**


An example to data clustering.

The data will be clustered into 9 clusters through kmeans algorithm.


**The example is written and tested with Python 3.6.4 && Python 3.5.2 **

--------------------------------------------------------------------------------------------

-----------------------------
## To setup the dependencies: 
-----------------------------

`pip3 install -r requirements.txt`

OR

`pip install -r requirements.txt`

--------------------------------------------------------------------------------------------

----------
## main.py
----------

A full example is established.

Needs to use other python files: csv_helper.py, consensus_matrix.py, WordsFrequency.py, cluster.py, clean_tweet.py

A well explained example of how to obtain get the cluster files by using the original data
**To run the code:**

`python3 main.py`

The script will produce 3 .txt files:

- name+ cluster1.csv to cluster9.csv: contains the tweets which have been clustered into 9 clusters, the name corresponds to .

- kmeans_edge: contains all the kmeans edges information for the further use

- kmeans_nodes: contains all the kmeans nodes information for the further use 

- DBSCAN_edge: contains all the DBSCAN edges information for the further use

- DBSCAN_nodes: contains all the DBSCAN nodes information for the further use 

-------------------
## csv_helper.py
-------------------

A class contains the basic method needed to get the tweets in the .csv file.

--------------------
###### load_csv: 
--------------------

- get the information in .csv file tweet by tweet

--------------------
###### load_csv2:
--------------------
- get the information in .csv file word by word


--------------------
## consensus_matrix.py
--------------------

A simple script that contains different method to calculate the kmenas consensus matrix 

The script itself is well documented and each method has a comment explaining the functionality.

-----------------
## WordsFrequency.py
-----------------

A simple script that contains different method to calculate the the most frequency words in each cluster

The script itself is well documented and each method has a comment explaining the functionality.

-----------------
## cluster.py
-----------------

A simple script that contains different method to generate different files 

The script itself is well documented and each method has a comment explaining the functionality.







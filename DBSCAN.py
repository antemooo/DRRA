import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
# This file contains the functions that achieve DBSCAN and clean the noise of DBSCAN

# This function is to get the DBSCAN matrix, and the eps from means the least eps and the eps_to means the largest
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

# This function is to clean up the noise via DBSCAN, and the threshold is 0.5
def db_noise(db_matrix):
    noise = np.sum(np.square(db_matrix), axis=1) / db_matrix.shape[1]
    noise_new = list([])
    for i in noise:
        if i > 0.5:
            noise_new.append(0)
        else:
            noise_new.append(1)

    return noise_new

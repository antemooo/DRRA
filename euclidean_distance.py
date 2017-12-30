# This is the vectorized version of function calculating the Euclidean distance between to matrice. You had implemented
# this function in the previous exercise, and it will be used in the kmeans' implementation.
import numpy as np
from matplotlib import offsetbox


def euclidean_vectorized(A, B):
    n, d = A.shape
    m, d1 = B.shape
    assert d == d1, 'Incompatible shape'
    A_squared = np.sum(np.square(A), axis=1)
    B_squared = np.sum(np.square(B), axis=1)
    # A_squared.reshape(A_squared.shape[0], 1)
    # B_squared.reshape(B_squared.shape[0], 1)
    AB = np.matmul(A, B.T)
    distances = np.sqrt(A_squared - 2 * AB + B_squared.T)
    return distances


def euclidean_vectorized2(A, B):
    n, d = A.shape
    m, d1 = B.shape
    assert d == d1, 'Incompatible shape'
    A_squared = np.sum(np.square(A), axis=1, keepdims=True)
    B_squared = np.sum(np.square(B), axis=1, keepdims=True)
    A_squared = A_squared.reshape(A_squared.shape[0], 1)
    B_squared = B_squared.reshape(B_squared.shape[0], 1)
    # print(A_squared.shape)
    # print(B_squared.shape)
    AB = np.matmul(A, B.T)
    # print(AB.shape)
    # print(AB.shape)
    distances = np.sqrt(A_squared - 2 * AB + B_squared.T)
    # print("Distances shape is", distances.shape)
    return distances


def cosin_vectorized(A, B):
    n, d = A.shape
    m, d1 = B.shape
    assert d == d1, 'Incompatible shape'
    A_squared = np.sum(np.square(A), axis=1, keepdims=True)
    B_squared = np.sum(np.square(B), axis=1, keepdims=True)
    AB = np.matmul(A, B.T)
    distances = AB / (A_squared * B_squared)
    return distances

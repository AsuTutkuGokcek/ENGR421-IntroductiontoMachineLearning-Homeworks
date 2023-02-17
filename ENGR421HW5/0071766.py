import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spat
from scipy.stats import multivariate_normal

X = np.genfromtxt("hw05_data_set.csv", delimiter = ",")
mean_values = np.genfromtxt("hw05_initial_centroids.csv", delimiter = ',')

N = X.shape[0]
D = X.shape[1]
K = mean_values.shape[0]

i_means = np.array([[+0.0, +5.5],
                    [-5.5, +0.0],
                    [+0.0, +0.0],
                    [+5.5, +0.0],
                    [+0.0, -5.5]]) 

i_covariance_matrices = np.array([[[+4.8, +0.0], [+0.0, +0.4]],
                                  [[+0.4, +0.0], [+0.0, +2.8]],
                                  [[+2.4, +0.0], [+0.0, +2.4]],
                                  [[+0.4, +0.0], [+0.0, +2.8]],
                                  [[+4.8, +0.0], [+0.0, +0.4]]])

plt.figure(figsize = (8, 8))
plt.plot(X[:, 0], X[:, 1],".", color = "k")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

clusters = np.argmin(spat.distance_matrix(mean_values, X), axis = 0)

covariance_matrices = []
I = [[0.0, 0.0], [0.0, 0.0]]
for i in range(K):
    for j in range(X[clusters == i].shape[0]):
        I += np.matmul(((X[clusters == i])[j,:] - mean_values[i,:])[:, None], ((X[clusters == i])[j,:] - mean_values[i,:][None,:]))
    covariance_matrices.append(I / X[clusters == i].shape[0])
    I = [[0.0, 0.0], [0.0, 0.0]]
    
prior_probabilities = []
for k in range(K):
    prior_probabilities.append(X[clusters == k].shape[0] / N)
prior_probabilities = np.array(prior_probabilities)

a = 0
while a < 100:
    mtx = []
    for k in range(K):
        post = multivariate_normal(mean_values[k], covariance_matrices[k]).pdf(X) * prior_probabilities[k]
        mtx.append(post)
    H = np.vstack([mtx[k] / np.sum(mtx, axis = 0) for k in range(K)])
    mean_values = (np.vstack([np.matmul(H[k], X) / np.sum(H[k], axis = 0) for k in range(K)]))
    covariance_matrices = []
    for k in range(K):
        I = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(N):
            I += np.matmul((X[i] - mean_values[k])[:, None], (X[i] - mean_values[k])[None,:]) * H[k][i]
        covariance_matrices.append(I / np.sum(H[k], axis = 0))
    prior_probabilities = []
    for k in range(K):
        prior_probabilities.append(np.sum(H[k], axis = 0) / N)
    prior_probabilities = np.array(prior_probabilities)
    a += 1
    
print(mean_values)

clusters=np.argmax(H, axis = 0)

plt.figure(figsize = (8, 8))
colours = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"])
for c in range(K):
    plt.plot(X[clusters == c, 0], X[clusters == c, 1], ".", markersize = 10, color = colours[c])
    x, y = np.meshgrid(np.linspace(-8, 8, 421), np.linspace(-8, 8, 421))
    plt.contour(x, y, multivariate_normal.pdf(np.concatenate((x.flatten()[:, None], y.flatten()[:, None]), axis = 1), mean_values[c], covariance_matrices[c]).reshape(421,421), levels = [0.05], colors = colours[c])
    plt.contour(x, y, multivariate_normal.pdf(np.concatenate((x.flatten()[:, None], y.flatten()[:, None]), axis = 1), i_means[c], i_covariance_matrices[c]).reshape(421,421), levels = [0.05], colors = "k", linestyles = "dashed")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
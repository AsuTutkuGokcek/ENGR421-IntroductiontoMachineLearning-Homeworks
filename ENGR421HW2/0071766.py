# %% [markdown]
# ## Libraries

# %%
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pandas as pd

def safelog(x):
    return(np.log(x + 1e-100))

# %% [markdown]
# ## Import Data

# %%
data_set = np.genfromtxt(fname = "hw02_data_points.csv", delimiter = ",")
class_labels = np.genfromtxt(fname = "hw02_class_labels.csv").astype(int)

w_data = np.genfromtxt(fname = "hw02_W_initial.csv", delimiter = ",")
w0_data = np.genfromtxt(fname = "hw02_w0_initial.csv", delimiter = ",")

training_data = data_set[0:10000]
test_data = data_set[10000:]

# %% [markdown]
# ## Algorithm Parameters

# %%
eta = 0.00001
iteration_count = 1000

D = data_set.shape[1]
X = data_set[0:10000,:]
N = X.shape[0]
K = np.max(class_labels)

#y_truth = shows which cloth is in which class
y_truth = np.zeros((N, K)).astype(int)
y_truth[range(N), class_labels[0:10000]-1] = 1

w = w_data[:,:]
w0 = w0_data[:]

# %% [markdown]
# ## Sigmoid Function

# %%
def sigmoid(X, w_set, w0_set):
    return(1 / (1 + np.exp(-(np.matmul(X, w_set) + w0_set))))

# %% [markdown]
# ## Gradient Functions

# %%
def gradient_w0(y_truth, y_predicted):
    return np.sum((y_truth - y_predicted) * y_predicted * (y_predicted - 1), axis = 0)

def gradient_W(X, y_truth, y_predicted):
    return np.asarray([-np.matmul(X.transpose(), (y_truth[:, c] - y_predicted[:,c]) * y_predicted[:, c] * (y_predicted[:,c]-1)) for c in range(K)]).transpose()

# %% [markdown]
# ## Iterative Algorithm (Question 4)

# %%
iteration = 0
objective_values = []

while iteration < iteration_count:
    y_predicted = sigmoid(X, w_data, w0_data)
    objective_values = np.append(objective_values, 0.5 * np.sum((y_truth - y_predicted)**2))
    w0_data = w0_data - eta * gradient_w0(y_truth, y_predicted)
    w_data = w_data + eta * gradient_W(X, y_truth, y_predicted)
    iteration = iteration + 1

print(w_data)
print(w0_data)

# %% [markdown]
# ## Question 5

# %%
plt.figure(figsize = (8, 4))
plt.plot(range(1, iteration + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()

# %% [markdown]
# ## Question 6

# %%
Truth_array = np.argmax(y_predicted, axis = 1) + 1
confusion_matrix = pd.crosstab(Truth_array.T, class_labels[0:10000].T,
                               rownames = ["y_pred"],
                               colnames = ["y_truth"])
print(confusion_matrix)

# %% [markdown]
# ## Question 7

# %%
iteration = 0
objective_values = []
X = data_set[10000:,:]
N = X.shape[0]

y_truth = np.zeros((N, K)).astype(int)
y_truth[range(N), class_labels[10000:]-1] = 1

y_predicted = sigmoid(X, w_data, w0_data)
Truth_array = np.argmax(y_predicted, axis = 1) + 1

confusion_matrix = pd.crosstab(Truth_array.T, class_labels[10000:].T,
                               rownames = ["y_pred"],
                               colnames = ["y_truth"])
print(confusion_matrix)



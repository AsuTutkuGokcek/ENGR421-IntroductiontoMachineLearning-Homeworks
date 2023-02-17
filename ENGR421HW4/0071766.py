import numpy as np
import matplotlib.pyplot as plt

def safelog2(x):
    if x == 0:
        return(0)
    else:
        return(np.log2(x))

data_set = np.genfromtxt("hw04_data_set.csv", delimiter = ",")

# get number of samples and number of features
N = data_set.shape[0]

train_set = data_set[1:151]
test_set = data_set[151:]

# get X and y values
x_train  = train_set[:,:1]
y_train = train_set[:,1:].astype(int)

x_test  = test_set[:,:1]
y_test = test_set[:,1:].astype(int)

minimum_value = 1.5
maximum_value = 5.2

# get numbers of train and test samples
N_train = len(train_set)
N_test = len(test_set)

node_indices = {}
is_terminal = {}
need_split = {}

node_features = {}
node_splits = {}

# put all training instances into the root node
node_indices[1] = np.array(range(N_train))
is_terminal[1] = False
need_split[1] = True

def learning_algorithm(P):
    while True:

        # find nodes that need splitting
        split_nodes = [key for key, value in need_split.items()
                       if value == True]
        # check whether we reach all terminal nodes
        if len(split_nodes) == 0:
            break
        # find best split positions for all nodes
        for split_node in split_nodes:
            data_indices = node_indices[split_node]
            need_split[split_node] = False
            if len(data_indices) <= P:
                is_terminal[split_node] = True
            else:
                is_terminal[split_node] = False

                unique_values = np.sort(np.unique(x_train[data_indices, 0]))
                split_positions = (unique_values[1:] + unique_values[:-1]) / 2
                split_scores = np.repeat(0.0, len(split_positions))

                for s in range(len(split_positions)):
                    left_indices = data_indices[x_train[data_indices,0] >= split_positions[s]]

                    right_indices = data_indices[x_train[data_indices,0] < split_positions[s]]

                    split_scores[s] = -(len(left_indices) / len(data_indices) * np.sum([np.mean(y_train[left_indices]) * np.log(np.mean(y_train[left_indices]))]) + len(right_indices) / len(data_indices) * np.sum([np.mean(y_train[right_indices]) * np.log(np.mean(y_train[right_indices]))]))

                best_score = np.min(split_scores)
                best_split = split_positions[np.argmin(split_scores)]

                # decide where to split on which feature
                node_features[split_node] = best_split
                node_splits[split_node] = best_split

                # create left node using the selected split
                left_indices = data_indices[x_train[data_indices,0] > best_split]

                node_indices[2 * split_node] = left_indices
                is_terminal[2 * split_node] = False
                need_split[2 * split_node] = True

                # create right node using the selected split
                right_indices = data_indices[x_train[data_indices,0] <= best_split]

                node_indices[2 * split_node + 1] = right_indices
                is_terminal[2 * split_node + 1] = False
                need_split[2 * split_node + 1] = True

learning_algorithm(25)
plt.figure(figsize=(8, 4))

plt.plot(x_train, y_train, "b.", markersize = 10)
plt.plot(x_test, y_test, "r.", markersize = 10)
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")

array = []
array.append(np.min(x_train))
for a in node_splits:
    array.append(node_splits[a])

array.sort()
left_borders = array[:]
right_borders = array[1:]
right_borders.append(maximum_value)

Regressogram_numerator = np.asarray([np.sum(((left_borders[i] <= x_train) & (x_train < right_borders[i])) * y_train) for i in range(len(left_borders))])
Regressogram_denominator = np.asarray([np.sum(((left_borders[i] <= x_train) & (x_train < right_borders[i]))) for i in range(len(left_borders))])
Score_function = np.divide(Regressogram_numerator, Regressogram_denominator)

for i in range(len(array) - 1):
    plt.plot([array[i], array[i+1]], [Score_function[i], Score_function[i]], "k-")

for i in range(len(right_borders)):
    plt.plot([left_borders[i], right_borders[i]], [Score_function[i], Score_function[i]], "k-")

for j in range(len(right_borders) - 1):
    plt.plot([right_borders[j], right_borders[j]], [Score_function[j],Score_function[j+1] ], "k-")

plt.show()

y_predicted_training = [((left_borders <= x_train[i]) & (x_train[i] < right_borders)) for i in range(len(x_train))] * Score_function
upper1 = ([max(y_predicted_training[b]) for b in range(len(y_predicted_training))] - y_train[:, 0]) ** 2
Error1 = np.sqrt(np.sum(upper1) / len(x_train))

y_predicted_test = [(left_borders < x_test[i]) & (x_test[i] <= right_borders) for i in range(len(x_test))] * Score_function
upper2 = ([max(y_predicted_test[b]) for b in range(len(y_predicted_test))] - y_test[:, 0]) ** 2
Error2 = np.sqrt(np.sum(upper2) / len(x_test))

print("RMSE on training set is", Error1, "when P is 25")
print("RMSE on test set is", Error2, "when P is 25")

plt.figure(figsize=(6, 6))

temp_error1 = []
temp_error2 = []

for n in range(5,55,5):

    node_indices = {}
    is_terminal = {}
    need_split = {}

    node_features = {}
    node_splits = {}

    # put all training instances into the root node
    node_indices[1] = np.array(range(N_train))
    is_terminal[1] = False
    need_split[1] = True

    learning_algorithm(n)
    array = []
    array.append(np.min(x_train))
    for a in node_splits:
        array.append(node_splits[a])

    array.sort()

    left_borders = array[:]
    right_borders = array[1:]

    right_borders.append(maximum_value)

    Regressogram_numerator = np.asarray([np.sum(((left_borders[i] <= x_train) & (x_train < right_borders[i])) * y_train) for i in range(len(left_borders))])
    Regressogram_denominator = np.asarray([np.sum(((left_borders[i] <= x_train) & (x_train < right_borders[i]))) for i in range(len(left_borders))])
    Score_function = np.divide(Regressogram_numerator, Regressogram_denominator)
   
    y_predicted_training = [((left_borders <= x_train[i]) & (x_train[i] < right_borders)) for i in range(len(x_train))] * Score_function
    upper1 = ([max(y_predicted_training[b]) for b in range(len(y_predicted_training))] - y_train[:, 0]) ** 2
    Error1 = np.sqrt(np.sum(upper1) / len(x_train))
    temp_error1.append(Error1)

    y_predicted_test = [(left_borders < x_test[i]) & (x_test[i] <= right_borders) for i in range(len(x_test))] * Score_function
    upper2 = ([max(y_predicted_test[b]) for b in range(len(y_predicted_test))] - y_test[:, 0]) ** 2
    Error2 = np.sqrt(np.sum(upper2) / len(x_test))
    temp_error2.append(Error2)

    plt.plot(n, Error1, "b.", markersize = 10)
    plt.plot(n, Error2, "r.", markersize = 10)

i = 0
for n in range(5,55,5):
    if(i != 9):
        plt.plot([n, n + 5], [temp_error2[i], temp_error2[i + 1]], "r-")
        plt.plot([n, n + 5], [temp_error1[i], temp_error1[i + 1]], "b-")
    i += 1

plt.ylabel("RMSE")
plt.xlabel("Pre-pruning size (P)")
plt.show()
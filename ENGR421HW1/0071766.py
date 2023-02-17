# ## Libraries

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# ## Import Data 

data_set = np.genfromtxt(fname = "hw01_data_points.csv", dtype=str, delimiter = ",")
class_labels = np.genfromtxt(fname = "hw01_class_labels.csv").astype(int)

# ## Question 3

training_data = data_set[0:300,:]
test_data = data_set[300:400,:]

# ## Question 4
print("Question 4")
#check how many 1 and 2 classes there are in the first 300 rows
training_one_counter = 0
training_two_counter = 0
for i in class_labels[0:300] :
    if i==1 :
        training_one_counter += 1
    else :
        training_two_counter += 1

class_priors = [training_one_counter/300, training_two_counter/300]

training_count = [[0 for i in range(7)] for j in range(8)]
# temp selects row of training_count
# j selects column of training count
for i in range(300) :
    for j in range(7) : 
        if training_data[i,j] == "A" :
            temp = 0
        elif training_data[i,j] == "C" :
            temp = 2
        elif training_data[i,j] == "G" :
            temp = 4
        elif training_data[i,j] == "T" :
            temp = 6   
        if class_labels[i] == 2 :
            temp += 1
        training_count[temp][j] += 1
result = np.array(training_count)/150   #150 in each class 
pAcd = result[0:2,:]
pCcd = result[2:4,:]
pGcd = result[4:6,:]
pTcd = result[6:8,:]
print(pAcd)
print(pCcd)
print(pGcd)
print(pTcd)
print(class_priors)


# ## Question 5
print("Question 5")
confusion_train = [[0,0],[0,0]]

for i in range(300) :
    probability_vector = 1
    prediction = 0
    for j in range(7) :
        if training_data[i,j] == "A" :
            probability_vector *= pAcd[:,j]
        elif training_data[i,j] == "C" :
            probability_vector *= pCcd[:,j]
        elif training_data[i,j] == "G" :
            probability_vector *= pGcd[:,j]
        elif training_data[i,j] == "T" :
            probability_vector *= pTcd[:,j]
            
    if probability_vector[0] > probability_vector[1]:
        prediction = 1
    else:
        prediction = 2
        
    if class_labels[i] == prediction :
        if prediction == 1:
            confusion_train[0][0] += 1
        else:
            confusion_train[1][1] += 1
    else:
        if prediction == 1:
            confusion_train[0][1] += 1
        else:
            confusion_train[1][0] += 1
print(np.array(confusion_train))

# ## Question 6
print("Question 6")
confusion_test = [[0,0],[0,0]]

for i in range(100) :
    probability_vector = 1
    prediction = 0
    for j in range(7) :
        if test_data[i,j] == "A" :
            probability_vector *= pAcd[:,j]
        elif test_data[i,j] == "C" :
            probability_vector *= pCcd[:,j]
        elif test_data[i,j] == "G" :
            probability_vector *= pGcd[:,j]
        elif test_data[i,j] == "T" :
            probability_vector *= pTcd[:,j]
            
    if probability_vector[0] > probability_vector[1]:
        prediction = 1
    else:
        prediction = 2
        
    if class_labels[i+300] == prediction :
        if prediction == 1:
            confusion_test[0][0] += 1
        else:
            confusion_test[1][1] += 1
    else:
        if prediction == 1:
            confusion_test[0][1] += 1
        else:
            confusion_test[1][0] += 1
print(np.array(confusion_test))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)

predicted_probablities = np.genfromtxt("hw06_predicted_probabilities.csv", delimiter = ",")
true_labels = np.genfromtxt("hw06_true_labels.csv", delimiter = ",")

minimum_value = 0
maximum_value = 1

data_interval = np.linspace(minimum_value, maximum_value, 5000)

def draw_roc_curve(true_labels, predicted_probablities):
    plt.figure(figsize=(8, 8))
    a = 0
    area = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    prev_tp_rate = 0
    prev_fp_rate = 0

    for i in data_interval:
        for j in range(len(predicted_probablities)):
            if predicted_probablities[j] >= i:
                a = 1
            else:
                a = -1
            if a == 1:
                if true_labels[j] == 1:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if true_labels[j] == 1:
                    false_negative += 1
                else:
                    true_negative += 1
            a = 0

        if true_positive+false_negative!= 0:
            tp_rate = true_positive/(true_positive+false_negative)

        if false_positive+true_negative!= 0:
            fp_rate = false_positive/(false_positive+true_negative)

        if prev_tp_rate != 0 and prev_fp_rate != 0:
            plt.plot([prev_fp_rate,fp_rate],[prev_tp_rate,tp_rate], "k-")

        area += (fp_rate-prev_fp_rate)*tp_rate
        prev_tp_rate = tp_rate
        prev_fp_rate = fp_rate

        true_positive = 0
        true_negative = 0
        false_negative = 0
        false_positive = 0

    print ("The area under the ROC curve is" , 1-area)
    plt.xlabel("FP Rate")
    plt.ylabel("TP Rate")
    plt.show()

def draw_pr_curve(true_labels, predicted_probablities):
    plt.figure(figsize=(8, 8))
    area = 0
    a = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    prev_recall = 0
    prev_precision = 0

    for i in data_interval:
        for j in range(len(predicted_probablities)):
            if predicted_probablities[j] > i:
                a = 1
            else:
                a = -1
            if a == 1:
                if true_labels[j] == 1:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if true_labels[j] == 1:
                    false_negative += 1
                else:
                    true_negative += 1
            a = 0

        if true_positive+false_positive!= 0:
            precision = true_positive/(true_positive+false_positive)
        if true_positive+false_negative!= 0:
            recall = true_positive/(true_positive+false_negative)
        if prev_precision != 0 and prev_recall != 0:
            plt.plot([prev_recall,recall],[prev_precision,precision], "k-")

        area += (precision-prev_precision)*recall
        prev_precision = precision
        prev_recall = recall

        true_positive = 0
        true_negative = 0
        false_negative = 0
        false_positive = 0
    print("The area under the PR curve is" ,area)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

draw_roc_curve(true_labels, predicted_probablities)
draw_pr_curve(true_labels, predicted_probablities)
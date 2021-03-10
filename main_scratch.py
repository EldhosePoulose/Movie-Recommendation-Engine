"""
We begin with a toy example
            ID  |   m1  /   m2  /   m3  /   User Interest (Yes/No or Y/N)

Train Data: 1       0       1       1       Y
            2       0       0       1       N
            3       0       0       0       Y
            4       1       1       0       Y
Test Data:  5       1       1       0       ?

here m1, m2, m3 is our feature and class represents Y or N
"""

import numpy as np
X_train = np.array([
    [0,1,1],
    [0,0,1],
    [0,0,0],
    [1,1,0]
])
Y_train = ['Y','N','Y','Y']
X_test = np.array([[1,1,0]])

def get_label_indices(labels):
    """
    Group samples based on their labels and return indices
    :param labels: list of labels
    :return: dict, {class1: [indices], class2: [indices]}
    """
    from collections import defaultdict
    label_indices = defaultdict(list)
    for index, label in enumerate(labels):
        label_indices[label].append(index)
    return label_indices

# To check the labels that we provided
label_indices= get_label_indices(Y_train)
#print("label_indices: \n", label_indices)

def get_prior(label_indices):
    """
    Compute Prior based on the train data
    :param label_indices: group of sample indices by class
    :return: dictionary, with class label as key, corresponding prior as the value
    """
    prior = {label: len(indices) for label, indices in label_indices.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= total_count
    return prior

prior= get_prior(label_indices)
#print("prior: \n", prior)

#Now we calculate the likelihood, which is the conditional probability, P(feature|class)

def get_likelihood(features, label_indices, smoothing=0):
    """
    Compute likelihood based on training data
    :param features: matrix of features
    :param label_indices: grouped sample indices by class
    :param smoothing: integer, additive smoothing parameter
    :return: dictionary, with class as KEY, corresponding conditional probability vector as VALUE
    """

    likelihood= {}
    for label, indices in label_indices.items():
        likelihood[label] = features[indices, :].sum(axis=0)+smoothing
        total_count = len(indices)
        likelihood[label] = likelihood[label] / (total_count + 2 * smoothing)

    return likelihood


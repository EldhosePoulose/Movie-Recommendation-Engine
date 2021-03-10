import numpy as np
X_train= np.array([
    [0,1,1],
    [0,0,1],
    [0,0,0],
    [1,1,0]
])
Y_train= ['Y','N','Y','Y']
X_test= np.array([[1,1,0]])

def get_label_indices(labels):
    """
    Group samples based on their labels and return indices
    :param labels: list of labels
    :return: dict, {class1: [indices], class2: [indices]}
    """
    from collections import defaultdict
    label_indices= defaultdict(list)
    for index, label in enumerate(labels):
        label_indices[label].append(index)
    return label_indices


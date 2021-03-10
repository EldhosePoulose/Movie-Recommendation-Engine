"""
We use a toy example
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
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 0],
    [1, 1, 0]
])
Y_train = ['Y', 'N', 'Y', 'Y']
X_test = np.array([[0, 0, 1]])

from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB(alpha = 1.0, fit_prior=True)
# We now train the NB model using our train data and label using fit method
clf.fit(X_train, Y_train)

# for showing the prediction probabilities: we use predict_prob method and to see the prediction: predict method
pred_prob = clf.predict_proba(X_test)
#print("Predicted Probabilities [scikit-learn]: \n", pred_prob)

pred = clf.predict(X_test)
print("Prediction [scikit-learn]: \n", pred)

# Self Learning: 10.03.2021

import numpy as np
from collections import defaultdict

data_path = "./metadata/ml-1m/ratings.dat"

import pandas as pd

# Reading ratings dataset into a pandas dataframe object.
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('./metadata/ml-1m/ratings.dat', sep='::', names=r_cols,
                      encoding='latin-1')

# Getting number of users and movies from the dataset. https://grouplens.org/datasets/book-crossing/
# https://grouplens.org/datasets/movielens/
n_users = ratings.user_id.unique().tolist()
n_movies = ratings.movie_id.unique().tolist()
print(type(n_users))
print(type(n_movies))
print('Number of Users: {}'.format(len(n_users)))
print('Number of Movies: {}'.format(len(n_movies)))


# function to preprocess the ratings data
def load_rating_data(data_path, n_users, n_movies):
    """
    We use this function to load the ratings and return the number of ratings for each movie and movie_id index mapping
    :param data_path: Path of the dataset
    :param n_users: number of users participated
    :param n_movies: number of movies
    :return: rating data in the numpy array of [user, movie];
             movie_n_rating, {movie_id: number of ratings};
             movie_id_mapping, {movie_id: col_index in rating data}
    """

    data = np.zeros([len(n_users), len(n_movies)], dtype=np.float)
    movie_id_mapping = {}
    movie_n_rating = defaultdict(int)
    with open(data_path, 'r') as file:
        for line in file.readlines()[1:]:
            user_id, movie_id, rating, _ = line.split("::")
            user_id = int(user_id) - 1
            if movie_id not in movie_id_mapping:
                movie_id_mapping[movie_id] = len(movie_id_mapping)
            rating = int(rating)
            data[user_id, movie_id_mapping[movie_id]] = rating
            if rating > 0:
                movie_n_rating[movie_id] += 1
    return data, movie_n_rating, movie_id_mapping


data, movie_n_rating, movie_id_mapping = load_rating_data(data_path, n_users, n_movies)


# data distribution
def display_distribution(data):
    values, counts = np.unique(data, return_counts=True)
    for value, count in zip(values, counts):
        print(f'Number of rating {int(value)}: {count}')


display_distribution(data)

# Check if NO Ratings available i.e, rating 0 for this dataset
# How the other ratings are distributed
# we set the movie with most known ratings as our TARGET/GOAL movie.
# for this we use sorting
movie_id_most, n_rating_most = sorted(movie_n_rating.items(), key=lambda d: d[1], reverse=True)[0]
print(f'Movie ID {movie_id_most} has {n_rating_most} ratings.')

# Now we reconstruct the dataset according to this information ( that is placing of target variable)

X_raw = np.delete(data, movie_id_mapping[movie_id_most], axis=1)
Y_raw = data[:, movie_id_mapping[movie_id_most]]

# clean the test data i.e. remove samples without ratings
X = X_raw[Y_raw > 0]
Y = Y_raw[Y_raw > 0]

print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)
# Visualize the Target movie rating distribution after cleaning
display_distribution(Y)

# Based on the distribution we decide above which rating we say the movie is being recommended or
# I recommend this movie because a lot of people rated it (Note: we don't know what they rated or their comments all about)

recommend = 3
Y[Y <= recommend] = 0  # Not the Best and Not recommend it
Y[Y > recommend] = 1  # Best and recommend it
n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()
# Note: Analyse the label distribution and see how balanced or in balanced it is.
print(f'{n_pos} positive samples and {n_neg} negative samples')

# SPLIT the dataset

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# see test and train sizes

print(len(Y_train), len(Y_test))

# Na??ve Bayes
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB(alpha=1.0, fit_prior=True)
# Multinomial since we have 5 classes
# We now train the NB model using our train data and label using fit method
clf.fit(X_train, Y_train)

# for showing the prediction probabilities: we use predict_prob method and to see the prediction: predict method
prediction_prob = clf.predict_proba(X_test)
print("Predicted Probabilities [scikit-learn]: \n", prediction_prob[0:10])

prediction = clf.predict(X_test)
print("Prediction [scikit-learn]: \n", prediction[:10])

# Classification Accuracy

accuracy = clf.score(X_test, Y_test)
print(f'The accuracy is: {accuracy * 100:.1f}%')

## EVALUATING CLASSIFICATION PERFORMANCE
# Date: 11.03.2020

# Confusion Matrix
from sklearn.metrics import confusion_matrix

print(confusion_matrix(Y_test, prediction, labels=[0, 1]))  # prediction variable used from above

#
from sklearn.metrics import precision_score, recall_score, f1_score
# Precision
# precision_score(Y_test, prediction, pos_label=1)
# Recall
# recall_score(Y_test, prediction, pos_label=1)
# F1 Score
# f1_score(Y_test, prediction, pos_label=1)
# f1_score(Y_test, prediction, pos_label=0) # negative/dislike class '0'

from sklearn.metrics import classification_report

report = classification_report(Y_test, prediction)
print(report)

# Area Under the Curve (AUC)

pos_prob = prediction_prob[:, 1]
thresholds = np.arange(0.0, 1.1, 0.05)
true_pos, false_pos = [0] * len(thresholds), [0] * len(thresholds)
for pred, y in zip(pos_prob, Y_test):
    for i, threshold in enumerate(thresholds):
        if pred >= threshold:
            # if truth and prediction are both 1
            if y == 1:
                true_pos[i] += 1
            # if truth is 0 while prediction is 1
            else:
                false_pos[i] += 1
        else:
            break

n_pos_test = (Y_test == 1).sum()
n_neg_test = (Y_test == 0).sum()
print(f'{n_pos_test} positive test samples and {n_neg_test} negative test samples')

true_pos_rate = [tp / n_pos_test for tp in true_pos]
false_pos_rate = [fp / n_neg_test for fp in false_pos]

import matplotlib.pyplot as plt

plt.figure()
lw = 2
plt.plot(false_pos_rate, true_pos_rate, color="red", lw=lw)
plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (fp)')
plt.ylabel('True Positive Rate (tp)')
plt.title("Receiver Operating Characteristics")
plt.legend(loc="lower right")
#plt.show()

# AUC Score
from sklearn.metrics import roc_auc_score

print(f' AUC-Score: {roc_auc_score(Y_test, pos_prob)}')

# Date: 12.03.2020
# Tuning Model using Cross-Validation

# k-fold CV

from sklearn.model_selection import StratifiedKFold

k = 5
k_fold = StratifiedKFold(n_splits=k)

# alpha: Smoothing factor
# fit_prior: whether to use prior tailored to the training data
smoothing_factor_alpha = [1, 2, 3, 4, 5, 6]
fit_prior_option = [True, False]
auc_record = {}

for train_indices, test_indices in k_fold.split(X, Y):
    #print("Train:", train_indices, "Test:", test_indices)
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    for alpha in smoothing_factor_alpha:
        if alpha not in auc_record:
            auc_record[alpha] = {}
        for fit_prior in fit_prior_option:
            clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            clf.fit(X_train, Y_train)
            prediction_prob = clf.predict_proba(X_test)
            pos_prob = prediction_prob[:, 1]
            auc = roc_auc_score(Y_test, pos_prob)
            auc_record[alpha][fit_prior] = auc + auc_record[alpha].get(fit_prior, 0.0)

for smoothing, smoothing_record in auc_record.items():
    for fit_prior, auc in smoothing_record.items():
        print(f'        {smoothing}        {fit_prior}      {auc / k:.5f}')

# see which smoothing and fit_prior or the hyper parameters are providing the best AUC value.
# here it is AUC= 0.65823 for [2, False]
# so we retrain the model with this values

clf = MultinomialNB(alpha=2.0, fit_prior=False)
clf.fit(X_train, Y_train)
pos_prob = clf.predict_proba(X_test)[:, 1]
print('AUC with the best model: ', roc_auc_score(Y_test, pos_prob))

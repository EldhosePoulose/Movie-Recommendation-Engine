# Self Learning: 10.03.2021

import numpy as np
from collections import defaultdict
data_path ="./metadata/ml-1m/ratings.dat"

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
    with open(data_path,'r') as file:
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

#data distribution
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

#clean the test data i.e. remove samples without ratings
X = X_raw[Y_raw > 0]
Y = Y_raw[Y_raw > 0]

print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)
# Visualize the Target movie rating distribution after cleaning
display_distribution(Y)

# Based on the distribution we decide above which rating we say the movie is being recommended or
# I recommend this movie because a lot of people rated it (Note: we don't know what they rated or their comments all about)

recommend = 3
Y[Y <= recommend] = 0 # Not the Best and Not recommend it
Y[Y > recommend] = 1 # Best and recommend it
n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()
# Note: Analyse the label distribution and see how balanced or in balanced it is.
print(f'{n_pos} positive samples and {n_neg} negative samples')

# SPLIT the dataset

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#see test and train sizes

print(len(Y_train), len(Y_test))


# Naïve Bayes
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB(alpha = 1.0, fit_prior=True)
# Multinomial since we have 5 classes
# We now train the NB model using our train data and label using fit method
clf.fit(X_train, Y_train)

# for showing the prediction probabilities: we use predict_prob method and to see the prediction: predict method
pred_prob = clf.predict_proba(X_test)
print("Predicted Probabilities [scikit-learn]: \n", pred_prob[0:10])

prediction = clf.predict(X_test)
print("Prediction [scikit-learn]: \n", prediction[:10])

# Classification Accuracy

accuracy = clf.score(X_test, Y_test)
print(f'The accuracy is: {accuracy*100:.1f}%')


## EVALUATING CLASSIFICATION PERFORMANCE

# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, prediction, labels=[0,1])) # prediction variable used from above

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





# Tuning Model using Cross-Validation


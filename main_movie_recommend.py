import numpy as np
from collections import defaultdict
data_path ="./metadata/ml-1m/ratings.dat"

import pandas as pd

# Reading ratings dataset into a pandas dataframe object.
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('./metadata/ml-1m/ratings.dat', sep='::', names=r_cols,
 encoding='latin-1')
# Getting number of users and movies from the dataset.
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

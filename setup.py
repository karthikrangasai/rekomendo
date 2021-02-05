import os
import numpy as np
import copy
import time

from src import svd_model
from src import cur_model

FILE_NAMES = {
	'movies': 'movies.dat',
	'ratings': 'ratings.dat',
	'users': 'users.dat',
}

SAVED_FILE_NAMES = {
	'movies': 'movies.dat',
	'ratings': 'ratings.dat',
	'users': 'users.dat',
}

movie_ids = {}
user_ids = {}

ratings_matrix_original_loc = os.path.join(os.getcwd(), 'ratings_matrix_original.npy')

if not os.path.exists(ratings_matrix_original_loc):
	def get_line_count(file_path):
		_id = 1
		num_lines = 0
		with open(file_path, mode='r') as f:
			line = f.readline()
			while line:
				line = line.split('::')
				
				if 'movies' in file_path:
					movie_ids[line[0]] = _id
				else:
					user_ids[line[0]] = _id
				
				num_lines += 1
				_id += 1
				line = f.readline()
		return num_lines

	# Counting the number of the movies and users
	# Movies: 3883 and Users: 6040
	dir_path = os.path.join(os.getcwd(), 'dataset')
	num_movies = get_line_count(os.path.join(dir_path, FILE_NAMES['movies']))
	num_users = get_line_count(os.path.join(dir_path, FILE_NAMES['users']))
	print("Movies: %d and Users: %d" % (num_movies, num_users))

	# Creating rating matrix and populating it
	ratings_matrix_original = np.zeros((num_movies, num_users))
	with open(os.path.join(dir_path, FILE_NAMES['ratings'])) as f:
		line = f.readline()
		while line:
			line = line.split("::")
			ratings_matrix_original[movie_ids[line[1]] - 1, user_ids[line[0]] - 1] = int(line[2])
			line = f.readline()

	np.save('ratings_matrix_original.npy', ratings_matrix_original)
else:
	ratings_matrix_original = np.load(ratings_matrix_original_loc)

train = ratings_matrix_original.T[:4000]
test = ratings_matrix_original.T[4000:, :2500]

corr = np.zeros((test.shape[0], train.shape[0]))
for i,t in enumerate(test):
	for j in range(train.shape[0]):
		corr[i,j] = np.correlate(test[i], train[j,:2500])
np.save('test_train_correlation_normal.npy', corr)


U, S, Vt = svd_model.SVD(train, False)
reconstructed = (U*S)@Vt
corr = np.zeros((test.shape[0], reconstructed.shape[0]))
for i,t in enumerate(test):
	for j in range(reconstructed.shape[0]):
		corr[i,j] = np.correlate(test[i], reconstructed[j,:2500])
np.save('test_train_correlation_svd_normal.npy', corr)


U, S, Vt = svd_model.SVD(train, True)
reconstructed = (U*S)@Vt
corr = np.zeros((test.shape[0], reconstructed.shape[0]))
for i,t in enumerate(test):
	for j in range(reconstructed.shape[0]):
		corr[i,j] = np.correlate(test[i], reconstructed[j,:2500])
np.save('test_train_correlation_svd_90.npy', corr)


C,U,R = cur_model.CUR(train, 1000, False)
reconstructed = C@U@R
corr = np.zeros((test.shape[0], reconstructed.shape[0]))
for i,t in enumerate(test):
	for j in range(reconstructed.shape[0]):
		corr[i,j] = np.correlate(test[i], reconstructed[j,:2500])
np.save('test_train_correlation_cur_normal.npy', corr)


C,U,R = cur_model.CUR(train, 1000, True)
reconstructed = C@U@R
corr = np.zeros((test.shape[0], reconstructed.shape[0]))
for i,t in enumerate(test):
	for j in range(reconstructed.shape[0]):
		corr[i,j] = np.correlate(test[i], reconstructed[j,:2500])
np.save('test_train_correlation_cur_90.npy', corr)
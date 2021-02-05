import os
import numpy as np
import copy
from sklearn.metrics import mean_squared_error
import time

from src import collaborative_model

ratings_matrix_original = np.load('ratings_matrix_original.npy')

def collaborative():
	correlation_matrix = np.load('test_train_correlation_normal.npy')
	start_time = time.time()
	rmse, spearmann = collaborative_model.collaborative_filtering(ratings_matrix_original.T[:4000], ratings_matrix_original.T[4000:, :2500], ratings_matrix_original.T[4000:, 2500:], correlation_matrix, 50)
	print("RMSE: %f" % (rmse))
	print("SPEARMANN: %f" % (spearmann))
	end_time = time.time()
	print("Collab time: " + str(end_time - start_time))
	
	correlation_matrix = np.load('test_train_correlation_svd_normal.npy')
	start_time = time.time()
	rmse, spearmann = collaborative_model.collaborative_filtering(ratings_matrix_original.T[:4000], ratings_matrix_original.T[4000:, :2500], ratings_matrix_original.T[4000:, 2500:], correlation_matrix, 50)
	print("RMSE: %f" % (rmse))
	print("SPEARMANN: %f" % (spearmann))
	end_time = time.time()
	print("Collab time: " + str(end_time - start_time))
	
	correlation_matrix = np.load('test_train_correlation_svd_90.npy')
	start_time = time.time()
	rmse, spearmann = collaborative_model.collaborative_filtering(ratings_matrix_original.T[:4000], ratings_matrix_original.T[4000:, :2500], ratings_matrix_original.T[4000:, 2500:], correlation_matrix, 50)
	print("RMSE: %f" % (rmse))
	print("SPEARMANN: %f" % (spearmann))
	end_time = time.time()
	print("Collab time: " + str(end_time - start_time))
	
	correlation_matrix = np.load('test_train_correlation_cur_normal.npy')
	start_time = time.time()
	rmse, spearmann = collaborative_model.collaborative_filtering(ratings_matrix_original.T[:4000], ratings_matrix_original.T[4000:, :2500], ratings_matrix_original.T[4000:, 2500:], correlation_matrix, 50)
	print("RMSE: %f" % (rmse))
	print("SPEARMANN: %f" % (spearmann))
	end_time = time.time()
	print("Collab time: " + str(end_time - start_time))
	
	correlation_matrix = np.load('test_train_correlation_cur_90.npy')
	start_time = time.time()
	rmse, spearmann = collaborative_model.collaborative_filtering(ratings_matrix_original.T[:4000], ratings_matrix_original.T[4000:, :2500], ratings_matrix_original.T[4000:, 2500:], correlation_matrix, 50)
	print("RMSE: %f" % (rmse))
	print("SPEARMANN: %f" % (spearmann))
	end_time = time.time()
	print("Collab time: " + str(end_time - start_time))


if __name__ == "__main__":
	collaborative()
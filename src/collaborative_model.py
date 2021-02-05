import numpy as np
import scipy as sp
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.metrics import precision_score

''' The following functions are used to calculate metrics used to evaluate our recommender systems'''

def rmse(y, y_pred):
	'''Root Mean Squared Error Calculation'''
	return np.sqrt(mean_squared_error(y[np.where(y>0)], y_pred[np.where(y>0)]))

def spearman_corrleation(y, y_pred):
	'''Spearmann Coreelation Calculation'''
	src = stats.spearmanr(y[np.where(y>0)], y_pred[np.where(y>0)])
	return src.correlation


'''A basic implementation of collaborative filtering that doesn't consider deviations from global parameters'''
'''This approach is used to find the K most similar users to the required user, and use their ratings to predict the required user's rating'''
def collaborative_filtering(train, test_given, test_known, correlation_matrix, K):
	num_test_users = test_given.shape[0]
	test_predicted = np.zeros(test_known.shape)
	for i in range(num_test_users):
		'''returns closest K users'''
		closest_K_users = np.argsort(-correlation_matrix[i])[:K] 
		closest_K_ratings = train[closest_K_users, 2500:]
		correlation_values = correlation_matrix[i][closest_K_users]
		'''multiplies correlations with their ratings'''
		closest_K_ratings = np.multiply(correlation_values, closest_K_ratings.T).T 
		'''weighted average of ratings is the rating predicted for our user'''
		test_predicted[i] = np.sum(closest_K_ratings, axis=0)/np.sum(correlation_values)
	'''metric evaluations'''
	RMSE = rmse(test_known, test_predicted)
	SRC = spearman_corrleation(test_known, test_predicted)
	return RMSE, SRC

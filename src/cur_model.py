import numpy as np
import scipy as sp
import copy

from src import svd_model

def column_select(ratings_matrix, k, retained_90_percent=False):
	c = int(k * np.log(k))
	_U,_S,Vh = svd_model.SVD(ratings_matrix, retained_90_percent=retained_90_percent)
	scores = np.divide(np.sum(np.square(Vh[:k,:]), axis=0), k)
	probs = np.minimum(np.multiply(scores, c), 1)
	probs = probs / probs.sum()
	indices = np.random.choice(Vh.shape[1], c, p=probs)
	shift = np.sqrt(np.multiply(scores, np.square(c)))
	actual_shift = shift[indices]
	C = np.divide(ratings_matrix[:,indices], actual_shift)
	return C

def CUR(input_ratings_matrix, k, retained_90_percent=False):
	C = column_select(input_ratings_matrix, k, retained_90_percent=retained_90_percent)
	R = column_select(input_ratings_matrix.T, k, retained_90_percent=retained_90_percent)
	U = np.linalg.pinv(C) @ input_ratings_matrix @ np.linalg.pinv(R.T)
	
	if retained_90_percent:
		np.save('cur_90_C.npy', C)
		np.save('cur_90_U.npy', U)
		np.save('cur_90_R.npy', R.T)
	else:	
		np.save('cur_C.npy', C)
		np.save('cur_U.npy', U)
		np.save('cur_R.npy', R.T)
	return C,U,R.T
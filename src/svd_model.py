import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigs
import copy

def SVD(input_ratings_matrix, retained_90_percent=False):
	print("\n>>> SVD Started: ")
	ratings_matrix = copy.deepcopy(input_ratings_matrix)

	AT_A = np.matmul(np.transpose(ratings_matrix), ratings_matrix)
	S_, V = np.linalg.eigh(AT_A)
	V = V.T

	S_temp = S_[np.where(S_>0)]
	V_temp = V[np.where(S_>0)]

	S_ = np.sort(S_temp)[::-1]
	V = V_temp[np.argsort(S_temp)][::-1]
	S_ = np.sqrt(S_)
	V = V.T	

	S_inv = np.linalg.inv(np.diag(S_))
	U = ratings_matrix @ (V @ S_inv)

	if retained_90_percent:
		# 90% Energy retention part
		denominator = np.sum(np.square(S_))
		for retain_thresh in range(S_.shape[0]):
			numerator = np.sum(np.square(S_[:(retain_thresh + 1)]))
			thresh = np.divide(numerator, denominator)
			if thresh >= 0.9:
				break
		
		S_ = S_[:retain_thresh]
		V = V[:,:retain_thresh]
		U = U[:,:retain_thresh]
		
		np.save('svd_90_U.npy', U)
		np.save('svd_90_S.npy', S_)
		np.save('svd_90_Vt.npy', V.T)
		
		del S_temp
		del V_temp
		del AT_A
		del ratings_matrix
		return U, S_, V.T

	np.save('svd_U.npy', U)
	np.save('svd_S.npy', S_)
	np.save('svd_Vt.npy', V.T)

	del S_temp
	del V_temp
	del AT_A
	del ratings_matrix
	return U, S_, V.T
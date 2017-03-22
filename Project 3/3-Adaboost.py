import numpy as np
import data_preprocess as dp


data = dp.getNumpyArrayFromFile('pima-indians-diabetes.data')
X = data[:, 0:-1]
Y = data[:, -1]

w = np.ones(X.shape[0]) / X.shape[0]
for d in range(0, X.shape[1]):
	dim_d = X[:, d]
	print np.min(dim_d), np.max(dim_d)
	r = np.max(dim_d) - np.min(dim_d)
	for i in range(np.min(dim_d), np.max(dim_d), r/10):
		print i
	break

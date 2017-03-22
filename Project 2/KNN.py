from scipy import stats
import numpy
import data_preprocess as dproc


def KNN_Search(K, X_train, Y_train, X_test):
	lenDtest = numpy.shape(X_test)[0]
	lenDtrain = numpy.shape(X_train)[0]
	Y_test = numpy.array([0.] * lenDtest)

	for i in range(0, lenDtest):
		x_test = X_test[i]
		l2_x_train = numpy.array([0.] * lenDtrain)
		for j in range(0, lenDtrain):
			x_train = X_train[j]
			l2_x_train[j] = dproc.L2(x_train, x_test)
		k_index = l2_x_train.argsort()[:K]
		Y_test[i] = stats.mode(Y_train[k_index])[0]
	return Y_test


def KNN_Weighted_Search(X_train, Y_train, X_test, bandwidth):
	lenDtest = numpy.shape(X_test)[0]
	Y_test = numpy.array([0.] * lenDtest)

	X_train0 = X_train[numpy.where(Y_train == 0), :]
	X_train1 = X_train[numpy.where(Y_train == 1), :]
	X_train0 = X_train0[0]
	X_train1 = X_train1[0]
	lenDtrain0 = numpy.shape(X_train0)[1]
	lenDtrain1 = numpy.shape(X_train1)[1]

	for i in range(0, lenDtest):
		x_test = X_test[i]
		weight0 = 0
		weight1 = 0
		for j0 in range(0, lenDtrain0):
			x_train = X_train0[j0]
			diff = dproc.L2(x_train, x_test)
			weight0 = weight0 + numpy.exp(-1. * numpy.square(diff) / numpy.square(bandwidth))
			# print weight0, numpy.exp(-1. * numpy.square(diff) / numpy.square(bandwidth))
		for j1 in range(0, lenDtrain1):
			x_train = X_train1[j1]
			diff = dproc.L2(x_train, x_test)
			weight1 = weight1 + numpy.exp(-1. * numpy.square(diff) / numpy.square(bandwidth))

		weight0 = weight0 / lenDtrain0
		weight1 = weight1 / lenDtrain1
		if weight0 > weight1:
			Y_test[i] = 0
		else:
			Y_test[i] = 1
		# break
	return Y_test

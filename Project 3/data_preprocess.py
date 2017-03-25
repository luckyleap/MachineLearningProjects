import random
import numpy
from sklearn import preprocessing


def reshapeCol(x):
	return numpy.reshape(x, (numpy.shape(x)[0], 1))


def reshapeRow(x):
	return numpy.reshape(x, (1, numpy.shape(x)[0]))


# Given a txt file of float values, return as numpy array
def getNumpyArrayFromFile(fileStr):
	txt_file = open(fileStr, 'r')
	data = []
	for lines in txt_file:
		lines = lines.strip().split(',')
		data.append(lines)
	data = numpy.array(data)
	data = data.astype(numpy.float)

	return data


# Return folded data, with test data as the input i index
def get_split_fold_data(data, test_index):
	test_data = data[test_index]
	folds = numpy.shape(data)[0]

	if test_index != 0:
		train_data = data[0]
	else:
		train_data = data[1]

	for i in range(0, folds):
		if test_index != i:
			curr_data = data[i]
			train_data = numpy.concatenate((train_data, curr_data), 0)

	return (test_data, train_data)


# Returns in n-fold array of the data split
def split_fold(data, fold):
	fold_array = []

	data_len = len(data)
	bin_len = data_len / fold

	d = numpy.array(data)

	start_index = 0

	for i in range(0, fold):
		end_index = (i + 1) * bin_len
		fold_array.append(d[start_index:end_index, :])
		start_index = end_index

	return fold_array


# normalize all columns
def normalizeData(data):
	for d in range(0, len(data[0])):
		data[:, d] = preprocessing.normalize(data[:, d])

	return data


# normalize all columns, keep last intact (labels)
def normalizeDataExceptLast(data):
	for d in range(0, len(data[0]) - 1):
		data[:, d] = preprocessing.normalize(data[:, d])

	return data


def flatten3Dto2D(data):
	d = numpy.array(data)
	ans = d.reshape(-1, 3)
	return ans


def flatten2Dto3D(data, shape):
	d = numpy.array(data)
	ans = d.reshape(shape[0], shape[1], 3)
	return ans


def stripLastColAsTest(data):
	data_train = data[:, :-1]
	data_ans = data[:, -1]

	return [data_train, data_ans]


def shuffle(array, ans):
	c = list(zip(array, ans))
	random.shuffle(c)
	array, ans = zip(*c)

	array = cleanArray(array)
	ans = cleanArray(ans)
	return [array, ans]


def cleanArray(array):
	ans = []
	for data in array:
		if len(data) != 0:
			ans.append(data)

	return ans


def split_data(data, percent):
	length = len(data)
	train_l = int(round(length * percent))

	train = data[0:train_l]
	test = data[train_l:]

	return [train, test]


# Returns array of indexes of all matching values
def findIndex(data, value):
	ans = []
	for i in range(0, len(data)):
		if data[i] == value:
			ans.append(i)

	return ans


def L2(y, y_pred):
	diff = numpy.square(numpy.subtract(y, y_pred))
	s = numpy.sum(diff)
	l2 = numpy.sqrt(s)
	return l2


def findAccuracy(y, y_pred):
	total = len(y)
	correct = 0.
	for i in range(0, total):
		curr_pred = y_pred[i]
		curr_true = y[i]
		if curr_pred == curr_true:
			correct = correct + 1

	return correct / total

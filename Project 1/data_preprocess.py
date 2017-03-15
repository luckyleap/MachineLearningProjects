import random
import numpy


def flatten3Dto2D(data):
	d = numpy.array(data)
	ans = d.reshape(-1, 3)
	return ans


def flatten2Dto3D(data, shape):
	d = numpy.array(data)
	ans = d.reshape(shape[0], shape[1], 3)
	return ans



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

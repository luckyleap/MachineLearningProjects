import random


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

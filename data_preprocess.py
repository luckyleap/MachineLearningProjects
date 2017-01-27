import random

def shuffle( array, ans ):
	c = list(zip(array, ans))
	random.shuffle(c)
	array, ans = zip(*c)

	return [array, ans]

def split_data( data, percent ):
	l = len(data)
	train_l = int(round(l * percent))


	train = data[0:train_l]
	test = data[train_l:]

	return [train, test]
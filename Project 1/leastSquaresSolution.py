import numpy


def leastSquareSolve(data, labels):
	opt_weight = numpy.dot(numpy.linalg.pinv(data), numpy.array(labels))
	return opt_weight
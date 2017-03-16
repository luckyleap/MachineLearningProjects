import numpy


def initializeWeights(data):
	dims = len(data[0])
	weights = [0] * dims
	return weights

def hasConverged(prev_error, curr_error, iteration):
	if iteration > 10000:
		return True

	if numpy.abs(prev_error - curr_error) < 0.0000000000001 and iteration > 1000:
		return True
	return False


def gradient(data, weights, groundTruth):
	N = data.shape[0]
	# Returns partial derivative values of each weight
	y_diff = groundTruth - numpy.dot(data, numpy.array(weights))
	gradient = numpy.dot(-1 * numpy.transpose(data), y_diff) / N
	return gradient

def getLearningRate(iteration):
	# Caps the lowest learning rate
	return max(1/iteration, 0.0001)


def getOptimalWeights(data, groundTruth):
	weights = initializeWeights(data)
	# print weights
	iteration = 1
	learningRate = 1
	curr_error = 0
	prev_error = float('inf')

	while(hasConverged(prev_error, curr_error, iteration) is False):
		# print '----Iteration---'
		# print 'Learning Rate', learningRate
		# print 'ERROR', curr_error
		g = gradient(data, weights, groundTruth)
		weights = numpy.subtract(weights, learningRate * g)
		# print weights
		prev_error = curr_error
		[y, curr_error] = getEstimateY(data, weights, groundTruth)
		iteration += 1
		learningRate = getLearningRate(iteration)

	return weights


def getEstimateY(data, weights, groundTruth):
	# print weights.shape, data.shape
	y = numpy.dot(data, numpy.array(weights))
	error = numpy.sum(numpy.square(numpy.subtract(groundTruth, y)))
	return [y, error]

import numpy
import leastSquaresSolution as ls
import gradient_descent as gd
import data_preprocess as dp

# Data splitting
data = numpy.loadtxt(open("winequality-red.csv", "rb"), delimiter=";", skiprows=1)
[data_train, data_ans] = dp.stripLastColAsTest(data)
[train, test] = dp.split_data(data_train, 0.5)
[train_ans, test_ans] = dp.split_data(data_ans, 0.5)

opt_weight = ls.leastSquareSolve(train, train_ans)

opt_w = numpy.transpose(opt_weight)
test_t = numpy.transpose(test)

predict = numpy.dot(opt_w, test_t)
error = dp.L2(test_ans, predict)

print 'Least Squares Solution L2 Error: ', error

opt_weight = gd.getOptimalWeights(train, train_ans)

opt_w = numpy.transpose(opt_weight)
test_t = numpy.transpose(test)

predict = numpy.dot(opt_w, test_t)
error = dp.L2(test_ans, predict)

print 'Gradient Descent Solution L2 Error: ', error

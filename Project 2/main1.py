import numpy
import data_preprocess as dproc
import KNN

data = dproc.getNumpyArrayFromFile('pima-indians-diabetes.data')
data = dproc.normalizeDataExceptLast(data)
d2 = dproc.split_fold(data, 10)

avg = []

# for i in range(0, 10):
# 	[test, train] = dproc.get_split_fold_data(d2, i)

# 	y_test = test[:, -1]
# 	x_test = test[:, 0:-1]
# 	y_train = train[:, -1]
# 	x_train = train[:, 0:-1]

# 	y_pred = KNN.KNN_Search(5, x_train, y_train, x_test)
# 	accuracy = dproc.findAccuracy(y_pred, y_test)
# 	avg.append(accuracy)

# avg_accuracy = numpy.average(avg)
# print 'Unweighted Accuracy', avg_accuracy

# Part 5 - Weighted KNN

avg = []

for i in range(0, 10):
	[test, train] = dproc.get_split_fold_data(d2, i)

	y_test = test[:, -1]
	x_test = test[:, 0:-1]
	y_train = train[:, -1]
	x_train = train[:, 0:-1]

	y_pred = KNN.KNN_Weighted_Search(x_train, y_train, x_test, 1000)
	accuracy = dproc.findAccuracy(y_pred, y_test)
	avg.append(accuracy)

avg_accuracy = numpy.average(avg)
print 'Weighted Accuracy', avg_accuracy

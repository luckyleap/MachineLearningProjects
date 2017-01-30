from sklearn import svm
import data_preprocess

# Load dataset
data = []
ans = []
file = open('iris.data.txt', 'rt')
for line in file:
	line = line.strip()
	array = line.split(',')
	data.append(array[0:-1])
	ans.append(array[-1])


file.close()

# Data set splitting

# Shuffle array
[data, ans] = data_preprocess.shuffle(data, ans)

# Split to train, test, then split train to validation
[trainData, testData] = data_preprocess.split_data(data, 0.6)
[trainAns, testAns] = data_preprocess.split_data(ans, 0.6)

[trainDataSmall, validData] = data_preprocess.split_data(trainData, (2.0 / 3.0))
[trainAnsSmall, validAns] = data_preprocess.split_data(trainAns, (2.0 / 3.0))

# Iterates through all possible combinations of parameters
C_Param_Values = [1, 50, 200, 500, 1000]
accuracy_a = []

for param in C_Param_Values:
	clf = svm.SVC(C=param)
	clf.fit(trainDataSmall, trainAnsSmall)
	accuracy = clf.score(validData, validAns, sample_weight=None)
	accuracy_a.append(accuracy)

param_value = C_Param_Values[data_preprocess.findIndex(accuracy_a, max(accuracy_a))[0]]
print param_value

# Run actual param on training
clf = svm.SVC(C=param_value)
clf.fit(trainData, trainAns)
train_error = clf.score(trainData, trainAns)
test_error = clf.score(testData, testAns)

generalization_error = train_error - test_error
print(generalization_error)


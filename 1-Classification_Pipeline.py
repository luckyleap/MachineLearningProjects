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
	ans.append(array[-1]);


file.close()

# Data set splitting

# Shuffle array
[data, ans] = data_preprocess.shuffle(data, ans)

# Split to train, test, then split train to validation
[trainData, testData] = data_preprocess.split_data(data, 0.6)
[trainAns, testAns] = data_preprocess.split_data(ans, 0.6)

[trainData, validData] = data_preprocess.split_data(trainData, (2.0/3.0))
[trainAns, validAns] = data_preprocess.split_data(trainAns, (2.0/3.0))




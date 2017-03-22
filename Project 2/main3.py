import numpy as np
import matplotlib.pyplot as plt
from random import randint


def getPred(y_pred):
	y_pos = np.where(y_pred >= 0)
	y_neg = np.where(y_pred < 0)
	y_pred[y_pos] = 1
	y_pred[y_neg] = -1
	return y_pred

def graphCurrData(w, x_data, y_data, fig, index):
	lenD = np.shape(x_data)[0]
	y_pred = np.dot(x_data, w)
	y_pred = getPred(y_pred)

	y_positive = np.where(y_data == 1)[0]
	y_negative = np.where(y_data == -1)[0]

	y_pred_positive = np.where(y_pred == 1)[0]
	y_pred_negative = np.where(y_pred == -1)[0]

	ax = fig.add_subplot(1, 7, index)
	str_var = 'Iteration: ' + str(index)
	ax.set_title(str_var)

	y_correct_positive = np.intersect1d(y_pred_positive, y_positive)
	y_correct_negative = np.intersect1d(y_pred_negative, y_negative)
	y_w_positive = np.setdiff1d(y_pred_positive, y_correct_positive)
	y_w_negative = np.setdiff1d(y_pred_negative, y_correct_negative)

	# # decision line
	x_line = [-5, 5]
	y_line = [x_line[0] * w[0], x_line[1] * w[1]]
	ax.plot(x_line, y_line, linestyle = 'dashed')

	if y_correct_positive.size > 0 and y_pred_positive.size > 0:
		print 'here'
		ax.plot(x_data[y_correct_positive, 0], x_data[y_correct_positive, 1], "go", markersize=7, markeredgewidth=1, markeredgecolor='g', markerfacecolor="None")

	if y_correct_negative.size > 0 and y_pred_negative.size > 0:
		print 'here2'
		ax.plot(x_data[y_correct_negative, 0], x_data[y_correct_negative, 1], "go", markersize=7, markeredgewidth=1, markeredgecolor='g')

	if y_w_positive.size > 0 and y_pred_positive.size > 0:
		print 'here3'
		ax.plot(x_data[y_w_positive, 0], x_data[y_w_positive], "ro", markersize=7, markeredgewidth=1, markeredgecolor='r')

	if y_w_negative.size > 0 and y_pred_negative.size > 0:
		print 'here4'
		ax.plot(x_data[y_w_negative, 0], x_data[y_w_negative], "ro", markersize=7, markeredgewidth=1, markeredgecolor='r', markerfacecolor="None")



data = np.array([[4, 1, -1], [2, 4, -1], [2, 3, -1], [3, 6, -1], [4, 4, -1], [9, 10, 1], [6, 8, 1], [9, 5, 1], [8, 7, 1], [10, 8, 1]])
x_data = data[:, 0:2] - 5
y_data = data[:, 2]

w = np.array([0, 0])


y_predict = np.array([0] * np.shape(x_data)[0])

data_size = np.shape(x_data)[0]
index = 0
y_pred = np.dot(x_data, w)

fig = plt.figure()
iteration = 0
fig_index = 0
hasMadeError = False
while(True):
	if np.mod(iteration, 3) == 0:
		graphCurrData(w, x_data, y_data, fig, fig_index + 1)
		fig_index = fig_index + 1

	next_index = np.mod(iteration, data_size)
	print next_index
	point = x_data[next_index]
	y_pred = np.dot(point, w)
	y_label = y_data[next_index]
	iteration = iteration + 1

	if (y_pred >= 0 and y_label == -1) or (y_pred < 0 and y_label == 1):
		# Incorrect needs to update w
		print 'wrong', next_index
		w = w + 0.1 * point * y_label
		hasMadeError = True

	# Break condition
	if np.mod(iteration, data_size) == 0 and hasMadeError is False:
		break
	elif np.mod(iteration, data_size) == 0:
		hasMadeError = False


plt.axis('equal')
plt.grid(True)
plt.show()



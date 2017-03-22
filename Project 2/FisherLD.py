import numpy as np



def findW(data1, data2):
	m1 = np.average(data1)
	m2 = np.average(data2)
	cov1 = np.cov(data1.T)
	cov2 = np.cov(data2.T)
	sw = cov1 + cov2
	w = np.linalg.inv(sw) * (m2 - m1)
	return w


def findSW(m1, m2, data1, data2):
	m1_len = np.shape(data1)[0]
	m2_len = np.shape(data2)[0]

	stotal = 0

	for i in range(0, m1_len):
		data = data1[i]
		s = np.transpose(data - m1) * ((data - m1))
		stotal += s

	for i in range(0, m2_len):
		data = data2[i]
		s = np.transpose(data - m2) * ((data - m2))
		stotal += s

	return stotal



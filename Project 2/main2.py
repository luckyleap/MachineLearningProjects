import matplotlib.pyplot as plt
import FisherLD as fisher
import numpy as np

data1 = np.array([[4, 1], [2, 4], [2, 3], [3, 6], [4, 4]])
data2 = np.array([[9, 10], [6, 8], [9, 5], [8, 7], [10, 8]])

w = fisher.findW(data1, data2)
print w

x = data1[:, 0]
y = w[0][1] / w[1][0] * x

x2 = data2[:, 1]
y2 = w[0][1] / w[1][0] * x2

#draw axis
plt.axis('equal')
plt.axis([0, 10, 0, 10])


plt.arrow(4.5, 4.5, -2, 2 * w[0][1]/w[1][0], head_width=0.3, head_length=0.3, fc='y', ec='y')
plt.plot(x, y, 'rs')
plt.plot(x2, y2, 'bs')

#plot a grid
plt.grid(True)

plt.show()

plt.close('all')
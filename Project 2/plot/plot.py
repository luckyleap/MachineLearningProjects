import matplotlib.pyplot as plt

#draw axis
plt.axis('equal')
plt.axis([0, 10, 0, 10])
#plot pointes
plt.plot([4, 2 , 2, 3, 4], [1, 4, 3, 6, 4], 'ro')
plt.plot([9, 6 , 9, 8, 10], [10, 8, 5, 7, 8], 'go')


#draw line
plt.plot([0, 10], [0, 10], linestyle='dashed')
#draw arrow
ax = plt.axes();
ax.arrow(5, 5, -2.5, 2.5, head_width=0.3, head_length=0.3, fc='y', ec='y');
#plot a grid
plt.grid(True)

plt.show()
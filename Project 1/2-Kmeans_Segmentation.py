import kmeans
import data_preprocess
import numpy
from scipy import misc

img_data = misc.imread('2092.jpg')
flatD = data_preprocess.flatten3Dto2D(img_data)
k = 4
[cluster_assignments, clusters] = kmeans.kmeansConvergence(flatD, k)

for i in range(0, k):
	# Replace value of image point with cluster point
	cluster_index = numpy.where(cluster_assignments == i)
	print cluster_index, clusters[i]
	flatD[cluster_index] = clusters[i]

print flatD
# print clusters
# print(kmeans.calcEuclideanSquare(clusters[0], clusters[1]))

notFlat = data_preprocess.flatten2Dto3D(flatD, img_data.shape)
img = misc.imsave('test.jpg', notFlat)
# [membership, mean_centers, ssd] = kmeans.fit(img_data, 3)

# print '--------MEMBERSHIP TEST-----------'
# print membership
# print '--------Center     Test-----------'
# print mean_centers
# print '--------Error      Test-----------'
# print ssd

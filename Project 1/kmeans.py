from random import randint
import numpy
import math

def initializeClusters(data, k):
	clusters = []
	d_len = data.shape[1]
	dimensions = [0] * d_len

	# Find max in the ranges of the D
	for i in range(0, d_len):
		dimensions[i] = (min(data[:, i]), max(data[:, i]))

	# Initilize clusters points
	for i in range(0, k):
		point = [0] * d_len
		for j in range(0, d_len):
			r = dimensions[j]
			point[j] = randint(r[0], r[1])
			while(math.isnan(point[j])):
				point[j] = randint(r[0], r[1])

		clusters.append(point)

	return clusters


def calcEuclideanSquare(p1, p2):
	sum_diff = numpy.sum(numpy.square(numpy.subtract(p1, p2)))
	return sum_diff


def findClosestCluster(clusters, data):
	# Returns a N * K array assigning the index of the cluster closest in association with the data
	data_l = len(data)
	closest_clusters = [0] * data_l

	for i in range(0, data_l):
		d = data[i]
		cluster_index = 0
		min_dist = float('inf')

		for j in range(0, len(clusters)):
			c = clusters[j]
			dist = calcEuclideanSquare(d, c)
			if dist < min_dist:
				min_dist = dist
				cluster_index = j

		closest_clusters[i] = cluster_index

	return closest_clusters


def findNewCentroids(data, cluster_assignments, clusters, k):
	new_clusters = []
	dims = len(clusters[0])
	# print 'dims', dims
	# print 'Cluster Shapes', clusters.shape
	for i in range(0, k):
		cluster_index = numpy.where(cluster_assignments == i)
		cluster_data = data[numpy.array(cluster_index)].reshape(-1, dims)
		m = numpy.mean(cluster_data, axis=0)
		new_clusters.append(m)

	return new_clusters


def isConvergent(prev_clusters, clusters):
	for i in range(0, len(prev_clusters)):
		prev = prev_clusters[i]
		curr = clusters[i]
		sum_diff = numpy.sum(numpy.absolute(numpy.subtract(prev, curr)))

		if (sum_diff > 1):
			return False

	return True


def kmeansConvergence(data, k):
	clusters = numpy.array(initializeClusters(data, k))
	cluster_assignments = numpy.array(findClosestCluster(clusters, data))
	prev_clusters = numpy.copy(clusters)

	# Create a temp cluster that will result to false for while loop
	prev_clusters[0][0] = prev_clusters[0][0] + 2

	while(isConvergent(prev_clusters, clusters) is False):
		print 'ITERATION---------------'
		prev_clusters = clusters
		clusters = numpy.array(findNewCentroids(data, cluster_assignments, clusters, k))
		print 'New Clusters', clusters
		cluster_assignments = numpy.array(findClosestCluster(clusters, data))
		print 'New cluster Assignments', cluster_assignments
		print 'Cluster Assig Shape', cluster_assignments.shape

	return [cluster_assignments, clusters]













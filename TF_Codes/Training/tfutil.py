import tensorflow as tf
import numpy as np
from functools import partial
def compute_pairwise_distances(x, y):
	"""Computes the squared pairwise Euclidean distances between x and y.

	Args:
	x: a tensor of shape [num_x_samples, num_features]
	y: a tensor of shape [num_y_samples, num_features]

	Returns:
	a distance matrix of dimensions [num_x_samples, num_y_samples].

	Raises:
	ValueError: if the inputs do no matched the specified dimensions.
	"""

	if not len(x.get_shape()) == len(y.get_shape()) == 2:
		raise ValueError('Both inputs should be matrices.')

	if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
		raise ValueError('The number of features should be the same.')

	def norm(x):
		return tf.reduce_sum(tf.square(x), 1)

	return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))


def gaussian_kernel_matrix(x, y, sigmas):
	"""Computes a Guassian Radial Basis Kernel between the samples of x and y. We
	create a sum of multiple gaussian kernels each having a width sigma_i.

	Args:
	x: a tensor of shape [num_samples, num_features]
	y: a tensor of shape [num_samples, num_features]
	sigmas: a tensor of floats which denote the widths of each of the
	  gaussians in the kernel.

	Returns:
	A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
	"""
	beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

	dist = compute_pairwise_distances(x, y)

	s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

	return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))




def maximum_mean_discrepancy(x,y, kernel=gaussian_kernel_matrix, name='maximum_mean_discrepancy'):
	r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.

	Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
	the distributions of x and y. Here we use the kernel two sample estimate
	using the empirical mean of the two distributions.

	MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
		= \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },

	where K = <\phi(x), \phi(y)>,
	is the desired kernel function, in this case a radial basis kernel.

	Args:
	x: a tensor of shape [num_samples, num_features]
	y: a tensor of shape [num_samples, num_features]
	kernel: a function which computes the kernel in MMD. Defaults to the
		GaussianKernelMatrix.

	Returns:
	a scalar denoting the squared maximum mean discrepancy loss.
	"""
	with tf.name_scope(name):
		# \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
		cost = tf.reduce_mean(kernel(x, x))
		cost += tf.reduce_mean(kernel(y, y))
		cost -= 2 * tf.reduce_mean(kernel(x, y))

		# We do not allow the loss to become negative.
		cost = tf.where(cost > 0, cost, 0, name='value')
	return cost


def mmd_loss(source_samples, target_samples, weight, name='mmd_loss'):
	"""Adds a similarity loss term, the MMD between two representations.

	This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
	different Gaussian kernels.

	Args:
	source_samples: a tensor of shape [num_samples, num_features].
	target_samples: a tensor of shape [num_samples, num_features].
	weight: the weight of the MMD loss.
	scope: optional name scope for summary tags.

	Returns:
	a scalar tensor representing the MMD loss value.
	"""
	with tf.name_scope(name):
		sigmas = [
		1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6
		]
		gaussian_kernel = partial(gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

		loss_value = maximum_mean_discrepancy(source_samples, target_samples, kernel=gaussian_kernel)
		loss_value = tf.maximum(1e-4, loss_value) * weight
	assert_op = tf.Assert(tf.is_finite(loss_value), [loss_value])
	with tf.control_dependencies([assert_op]):
		tag = 'MMDLoss'
		barrier = tf.no_op(tag)
	return loss_value

def optimizeAdam(cost, learning_rate = 0.001):
	return tf.train.AdamOptimizer(learning_rate).minimize(cost)

def optimizeMomentum(cost, learning_rate = 0.01, momentum = 0.9):
	return tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cost)



import tensorflow as tf
import tfutil
import numpy as np
import os

def testImgtype(x):
	pristinecnt = 0
	tamperedcnt = 0
	n_classes = 2 # number of classes to be classified


	l1 = 0.7		# Lambda 1
	l2 = 0.4		# lambda 2

	x = x/255.0
	weights = {
	    "wc1": tf.Variable(tf.random_normal([5, 5, 3, 32], stddev=0.01), name="wc1"),
	    "wc2": tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01), name="wc2"),
	    "wc3": tf.Variable(tf.random_normal([3, 3, 64, 192], stddev=0.01), name="wc3"),
	    "wc4": tf.Variable(tf.random_normal([3, 3, 192, 128], stddev=0.01), name="wc4"),
	    "wc5": tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=0.01), name="wc5"),
	    "wf1": tf.Variable(tf.random_normal([6272, 512], stddev=0.01), name="wf1"),
	    "wdas": tf.Variable(tf.random_normal([512, 256], stddev=0.01), name="wdas"),
	    "wdat": tf.Variable(tf.random_normal([512, 256], stddev=0.01), name="wdat"),
	    "wcs": tf.Variable(tf.random_normal([256, n_classes],   stddev=0.01), name="wcs"),
	    "wct": tf.Variable(tf.random_normal([256, n_classes],   stddev=0.01), name="wct")

	}

	# Bias parameters as devised in the original research paper
	biases = {
	    "bc1": tf.Variable(tf.random_normal([32], stddev=0.01), name="bc1"),
	    "bc2": tf.Variable(tf.random_normal([64], stddev=0.01), name="bc2"),
	    "bc3": tf.Variable(tf.random_normal([192], stddev=0.01), name="bc3"),
	    "bc4": tf.Variable(tf.random_normal([128], stddev=0.01), name="bc4"),
	    "bc5": tf.Variable(tf.random_normal([128], stddev=0.01), name="bc5"),
	    "bf1": tf.Variable(tf.random_normal([512], stddev=0.01), name="bf1"),
	    "bdas": tf.Variable(tf.random_normal([256], stddev=0.01), name="bdas"),
	    "bdat": tf.Variable(tf.random_normal([256], stddev=0.01), name="bdat"),
	    "bcs": tf.Variable(tf.random_normal([n_classes], stddev=0.01), name="bcs"),
	    "bct": tf.Variable(tf.random_normal([n_classes], stddev=0.01), name="bct")

	}


	# input and output vector placeholders for target domain
	x_t = tf.placeholder(tf.float32, [None, 64, 64, 3])
	y_t = tf.placeholder(tf.float32, [None, n_classes])

	keepprob = tf.placeholder(tf.float32)

	# fully connected layer
	def fc_layer(x, W, b, name="fc"):
		return tf.nn.bias_add(tf.matmul(x, W), b)


	# Common path
	def alexnet(img, weights, biases, keep_prob):
		# 1st convolutional layer
		conv1 = tf.nn.conv2d(img, weights["wc1"], strides=[1, 1, 1, 1], padding="SAME", name="conv1")
		conv1 = tf.nn.bias_add(conv1, biases["bc1"])
		conv1 = tf.nn.relu(conv1)
		conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

		# 2nd convolutional layer
		conv2 = tf.nn.conv2d(conv1, weights["wc2"], strides=[1, 1, 1, 1], padding="SAME", name="conv2")
		conv2 = tf.nn.bias_add(conv2, biases["bc2"])
		conv2 = tf.nn.relu(conv2)
		conv2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

		# 3rd convolutional layer
		conv3 = tf.nn.conv2d(conv2, weights["wc3"], strides=[1, 1, 1, 1], padding="SAME", name="conv3")
		conv3 = tf.nn.bias_add(conv3, biases["bc3"])
		conv3 = tf.nn.relu(conv3)

		# 4th convolutional layer
		conv4 = tf.nn.conv2d(conv3, weights["wc4"], strides=[1, 1, 1, 1], padding="SAME", name="conv4")
		conv4 = tf.nn.bias_add(conv4, biases["bc4"])
		conv4 = tf.nn.relu(conv4)

		# 5th convolutional layer
		conv5 = tf.nn.conv2d(conv4, weights["wc5"], strides=[1, 1, 1, 1], padding="SAME", name="conv5")
		conv5 = tf.nn.bias_add(conv5, biases["bc5"])
		conv5 = tf.nn.relu(conv5)
		conv5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

		# stretching out the 5th convolutional layer into a long n-dimensional tensor
		shape = [-1, weights['wf1'].get_shape().as_list()[0]]
		flatten = tf.reshape(conv5, shape)

		# 1st fully connected layer
		fc1 = fc_layer(flatten, weights["wf1"], biases["bf1"], name="fc1")
		fc1 = tf.nn.relu(fc1)
		fc1 = tf.nn.dropout(fc1, keep_prob=keep_prob)
		return fc1

	def domainAdapt(fcs, fct, weights, biases, keep_prob):
		# Source Domain Adaptation fully connected layer
		fcdas = fc_layer(fcs, weights["wdas"], biases["bdas"], name="fcdas")
		fcdas = tf.nn.relu(fcdas)
		fcdas = tf.nn.dropout(fcdas, keep_prob=keep_prob)

		# Target Domain Adaptation fully connected layer
		fcdat = fc_layer(fct, weights["wdat"], biases["bdat"], name="fcdat")
		fcdat = tf.nn.relu(fcdat)
		fcdat = tf.nn.dropout(fcdat, keep_prob=keep_prob)

		mmdloss = tfutil.mmd_loss(fcdas, fcdat, 1)
		# Source head
		fccs = fc_layer(fcdas, weights["wcs"], biases["bcs"], name="fccs")
		# Target head
		fcct = fc_layer(fcdat, weights["wct"], biases["bct"], name="fcct")


		# Return source head, target head and the calculated mmd loss
		return fccs, fcct, mmdloss



	def targetpath(fct, weights, biases, keep_prob):
		# Target Domain Adaptation fully connected layer
		fcdat = fc_layer(fct, weights["wdat"], biases["bdat"], name="fcdat")
		fcdat = tf.nn.relu(fcdat)
		fcdat = tf.nn.dropout(fcdat, keep_prob=keep_prob)

		fcct = fc_layer(fcdat, weights["wct"], biases["bct"], name="fcct")
		return fcct
	


	targetcom = alexnet(x_t, weights, biases, keepprob)	# Target image passing path

	# Checking accuracy
	targetheadtest = targetpath(targetcom, weights, biases, keepprob)
	targetheadtest = tf.nn.softmax(targetheadtest)

	saver = tf.train.Saver()
	sess = tf.Session()
	saver.restore(sess, "fm/try.ckpt")
	accrr =  sess.run(targetheadtest, feed_dict={x_t:x, keepprob:1.})
	sess.close()
	tf.reset_default_graph()
	for i in range(accrr.shape[0]):
		if accrr[i][0] > accrr[i][1]:
			pristinecnt = pristinecnt+1
		else:
			tamperedcnt = tamperedcnt+1
	if float(tamperedcnt)/float(accrr.shape[0]) >= 0.25:
		return 0, pristinecnt, tamperedcnt
	else:
		return 1, pristinecnt, tamperedcnt 
	'''
	if pristinecnt >= tamperedcnt:
		return 1, pristinecnt, tamperedcnt
	else:
		return 0, pristinecnt, tamperedcnt
	'''
	

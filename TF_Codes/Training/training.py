import tensorflow as tf
import tfutil
import numpy as np
import os


n_classes = 2 # number of classes to be classified


l1 = 0.7		# Lambda 1
l2 = 0.4		# lambda 2

batchsize = 100
incr = batchsize


ll1 = 0
ll2 = 0

source = np.load("/media/shubh/PranayHDD/numpy_npz/training_data_0.npz")
sourceimages = None
sourcelabels = None
targetimages = None
targetlabels = None

maxpatches = None



def nextbatch():
	global ll1
	global ll2
	global incr
	im = np.transpose((sourceimages[ll1:ll1+incr,...]), (0, 3, 2, 1))
	lb = sourcelabels[ll1:ll1+incr,...]
	batch_xs = im

	batch_ys = np.zeros((incr,2))
	batch_ys[np.arange(incr),lb.astype('int')] =1


	im = np.transpose((targetimages[ll2:ll2+incr,...]), (0, 3, 2, 1))
	lb = targetlabels[ll2:ll2+incr,...]
	batch_xt = im

	batch_yt = np.zeros((incr,2))
	batch_yt[np.arange(incr),lb.astype('int')] =1


	ll1 = ll1+incr
	ll2 = ll2+incr
	


	return batch_xs, batch_ys, batch_xt, batch_yt



weights = {
	"wc1": tf.Variable(tf.random_normal([5, 5, 3, 32], stddev=0.1), name="wc1"),
	"wc2": tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.1), name="wc2"),
	"wc3": tf.Variable(tf.random_normal([3, 3, 64, 192], stddev=0.1), name="wc3"),
	"wc4": tf.Variable(tf.random_normal([3, 3, 192, 128], stddev=0.1), name="wc4"),
	"wc5": tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=0.1), name="wc5"),
	"wf1": tf.Variable(tf.random_normal([6272, 512], stddev=0.1), name="wf1"),
	"wdas": tf.Variable(tf.random_normal([512, 256], stddev=0.1), name="wdas"),
	"wdat": tf.Variable(tf.random_normal([512, 256], stddev=0.1), name="wdat"),
	"wcs": tf.Variable(tf.random_normal([256, n_classes],   stddev=0.1), name="wcs"),
	"wct": tf.Variable(tf.random_normal([256, n_classes],   stddev=0.1), name="wct")

}

# Bias parameters as devised in the original research paper
biases = {
	"bc1": tf.Variable(tf.random_normal([32], stddev=0.1), name="bc1"),
	"bc2": tf.Variable(tf.random_normal([64], stddev=0.1), name="bc2"),
	"bc3": tf.Variable(tf.random_normal([192], stddev=0.1), name="bc3"),
	"bc4": tf.Variable(tf.random_normal([128], stddev=0.1), name="bc4"),
	"bc5": tf.Variable(tf.random_normal([128], stddev=0.1), name="bc5"),
	"bf1": tf.Variable(tf.random_normal([512], stddev=0.1), name="bf1"),
	"bdas": tf.Variable(tf.random_normal([256], stddev=0.1), name="bdas"),
	"bdat": tf.Variable(tf.random_normal([256], stddev=0.1), name="bdat"),
	"bcs": tf.Variable(tf.random_normal([n_classes], stddev=0.1), name="bcs"),
	"bct": tf.Variable(tf.random_normal([n_classes], stddev=0.1), name="bct")

}

# input and output vector placeholders for source domain
x_s = tf.placeholder(tf.float32, [None, 64, 64, 3])
y_s = tf.placeholder(tf.float32, [None, n_classes])

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
	print conv5.shape
	# stretching out the 5th convolutional layer into a long n-dimensional tensor
	shape = [-1, weights['wf1'].get_shape().as_list()[0]]
	flatten = tf.reshape(conv5, shape)
	print flatten.shape
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
	


sourcecom = alexnet(x_s, weights, biases, keepprob)	# Source image passing path
targetcom = alexnet(x_t, weights, biases, keepprob)	# Target image passing path
sourcehead, targethead, mmdloss = domainAdapt(sourcecom, targetcom, weights, biases, keepprob) #Source, target head  mmd loss

soucl_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=sourcehead, labels=y_s))
tarcl_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=targethead, labels=y_t))

loss = tarcl_loss + l1*soucl_loss + l2*mmdloss
optm = tfutil.optimizeAdam(loss)

# Checking accuracy
targetheadtest = targetpath(targetcom, weights, biases, keepprob)
correct_pred = tf.equal(tf.argmax(targetheadtest,1), tf.argmax(y_t,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()
saver = tf.train.Saver()

sess = tf.Session()
saver.restore(sess, "./Models/try2.ckpt")
sess.run(init)

GG = 40
source_list = [ np.load("/media/shubh/PranayHDD/numpy_npz/training_data_{}.npz".format(i)) for i in range(GG)]

for nepochs in range(200):
	
	print(nepochs)
	for i in range(0,200,2):
		print("Npz file:",i)

		source = source_list[i%GG]
		sourceimages = source['arr_0']
		sourcelabels = source['arr_1']
		targetimages = source['arr_2']
		targetlabels = source['arr_3']
		maxpatches  = max(targetimages.shape[0], sourceimages.shape[0])
		for numbatch in range(maxpatches/incr):

			batch_xs, batch_ys, batch_xt, batch_yt = nextbatch()
			#print "Batch loaded"
			if ll1+incr>=sourceimages.shape[0]:
				ll1 = 0
			if ll2+incr>=targetimages.shape[0]:
				ll2 = 0
		
			sess.run(optm, feed_dict = {x_s:batch_xs, y_s:batch_ys, x_t:batch_xt, y_t:batch_yt, keepprob:.5})
				#print "Test Accuracy: ", sess.run(accuracy, feed_dict={x_t:batch_xt, y_t:batch_yt, keepprob:1.})	
			
		
		ll1 = 0
		ll2 = 0
		print "Test Accuracy: ", sess.run(accuracy, feed_dict={x_t:batch_xt, y_t:batch_yt, keepprob:1.})	
		# print "Test Accuracy: ", sess.run(targethead, feed_dict={x_t:batch_xt, y_t:batch_yt, keepprob:1.})	
		saver.save(sess, "Models/try2.ckpt")	
	


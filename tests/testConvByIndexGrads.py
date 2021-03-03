import numpy
import tensorflow as tf
import cliffordConvolution as cc

import scipy.ndimage
import time
import operator
from functools import reduce


# insert_zeros = tf.load_op_library("./insert_zeros.so")
# a = insert_zeros.insert_zeros(numpy.ones([100,30,30,100], dtype=numpy.float32), numpy.array([[21,3,15,90]]).astype(numpy.int32))
# a = numpy.ones(shape=[64,56,56,64], dtype=numpy.float32)	#input
a = numpy.random.rand(64,56,56,64).astype(numpy.float32)	#input
# b = numpy.zeros(shape=[10,3,3,64,32], dtype=numpy.float32)	#weights
c = numpy.ones(shape=[64,56,56,32], dtype=numpy.int32)*2		#indexes
d = numpy.ones(shape=[64,56,56,32], dtype=numpy.int32)		#mask
t = numpy.ones(shape=[64,56,56,32], dtype=numpy.float32)		#thetas2
# e = numpy.ones(shape=[3,3,64,32], dtype=numpy.float32)		#flatWeights
gt = numpy.ones(shape=[64,56,56,32], dtype=numpy.float32)
# gt[:,:,:,::2] = -1

gt = (numpy.random.rand(64,56,56,32).astype(numpy.float32) - 0.5)*2
e = numpy.random.rand(3,3,64,32).astype(numpy.float32)
c = numpy.floor(numpy.random.rand(64,56,56,32).astype(numpy.float32)*33).astype(numpy.int32) #indexes
d = numpy.floor(numpy.random.rand(64,56,56,32).astype(numpy.float32)*2).astype(numpy.int32) #mask

# for i in range(32):
	# weightSet = numpy.random.rand(3,3,64).astype(numpy.float32)
	# b[i%10,:,:,:,i] = weightSet
	# b[(i+1)%10,:,:,:,i] = weightSet
	# e[:,:,:,i] = weightSet
	# c[:,:,:,i] = i%10
	# c[:,20:40,12:56,i] = (i+1)%10

inpt = tf.placeholder(tf.float32, [64,56,56,64])
# rotatedWeights = tf.placeholder(tf.float32, [10,3,3,64,32])
quantized = tf.placeholder(tf.int32, [64,56,56,32])
thetas2 = tf.placeholder(tf.float32, [64,56,56,32])
convMask = tf.placeholder(tf.int32, [64,56,56,32])
flatWeights = tf.placeholder(tf.float32, [3,3,64,32])
groundTruth = tf.placeholder(tf.float32, [64,56,56,32])
lab = tf.placeholder(tf.int32, [])

num_bins = 32
num_angles = 4
offset = num_bins//(2*num_angles)
angles = [tf.constant([(a-offset)*2*numpy.pi/num_bins]) for a in range(num_bins+1)]
rotatedWeightsL = []
for angle in angles:
	rotatedWeightsL.append(cc.transformations.rotateVectorField(flatWeights, angle))

rotatedWeights = tf.stack(rotatedWeightsL, axis=0)

with tf.device('gpu:0'):
	convByIndexCudaRes = cc.ops.convByIndex(inpt, rotatedWeights, quantized, convMask, thetas2, [1,1,1,1], "SAME")
	flatConvCudaResL = []
	for rotation in range(num_bins+1):
		rMask = tf.cast(tf.equal(quantized, rotation), tf.float32)
		flatConvCudaResL.append(tf.nn.conv2d(inpt, rotatedWeightsL[rotation], [1,1,1,1], "SAME")*rMask)
	flatConvCudaRes = tf.reduce_sum(tf.stack(flatConvCudaResL, axis=0), axis=0) * tf.cast(convMask, tf.float32)
	# myErrorGPU = tf.reduce_sum(tf.pow(groundTruth - convByIndexCudaRes,1))
	# tfErrorGPU = tf.reduce_sum(tf.pow(groundTruth - flatConvCudaRes,1))
	myErrorGPU = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reduce_sum(convByIndexCudaRes, axis=[0,1,2]), labels=lab)
	tfErrorGPU = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reduce_sum(flatConvCudaRes, axis=[0,1,2]), labels=lab)
	flatInputGradsGPU = tf.gradients(tfErrorGPU, inpt)
	indexedInputGradsGPU = tf.gradients(myErrorGPU, inpt)
	flatWeightGradsGPU = tf.gradients(tfErrorGPU, flatWeights)
	indexedWeightGradsGPU = tf.gradients(myErrorGPU, flatWeights)

with tf.device('cpu:0'):
	convByIndexCPURes = cc.ops.convByIndex(inpt, rotatedWeights, quantized, convMask, thetas2, [1,1,1,1], "SAME")
	flatConvCPUResL = []
	for rotation in range(num_bins+1):
		rMask = tf.cast(tf.equal(quantized, rotation), tf.float32)
		flatConvCPUResL.append(tf.nn.conv2d(inpt, rotatedWeightsL[rotation], [1,1,1,1], "SAME")*rMask)
	flatConvCPURes = tf.reduce_sum(tf.stack(flatConvCPUResL, axis=0), axis=0) * tf.cast(convMask, tf.float32)
	myErrorCPU = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reduce_sum(convByIndexCPURes, axis=[0,1,2]), labels=lab)
	tfErrorCPU = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.reduce_sum(flatConvCPURes, axis=[0,1,2]), labels=lab)
	# myErrorCPU = tf.reduce_sum(tf.pow(groundTruth - convByIndexCPURes,1))
	# tfErrorCPU = tf.reduce_sum(tf.pow(groundTruth - flatConvCPURes,1))
	flatInputGradsCPU = tf.gradients(tfErrorCPU, inpt)
	indexedInputGradsCPU = tf.gradients(myErrorCPU, inpt)
	flatWeightGradsCPU = tf.gradients(tfErrorCPU, flatWeights)
	indexedWeightGradsCPU = tf.gradients(myErrorCPU, flatWeights)

# resGrads = tf.gradients(myErrorCPU, convByIndexCPURes)

myconfig = tf.ConfigProto(log_device_placement=False)
myconfig.gpu_options.allow_growth = True
sess = tf.Session(config=myconfig)

# res = sess.run([test_grads, flatWeightGradsGPU], feed_dict={inpt: a, quantized: c, convMask: d, flatWeights: e, groundTruth: gt})
st1 = time.time()
results = sess.run([indexedWeightGradsGPU, flatWeightGradsGPU, indexedWeightGradsCPU, flatWeightGradsCPU], feed_dict={inpt: a, quantized: c, convMask: d, flatWeights: e, groundTruth: gt, thetas2: t, lab: 12})
t1 = time.time() - st1

numpy.mean(numpy.abs(results[0][0]))
results = sess.run([indexedInputGradsGPU, flatInputGradsGPU, indexedInputGradsCPU, flatInputGradsCPU], feed_dict={inpt: a, quantized: c, convMask: d, flatWeights: e, groundTruth: gt})
numpy.mean(numpy.abs(results[0][0] - results[1][0])/numpy.abs(results[0][0]))
numpy.mean(numpy.abs(results[0][0] - results[2][0])/numpy.abs(results[0][0]))
numpy.mean(numpy.abs(results[0][0] - results[3][0])/numpy.abs(results[0][0]))
numpy.mean(numpy.abs(results[1][0] - results[2][0])/numpy.abs(results[0][0]))
numpy.mean(numpy.abs(results[1][0] - results[3][0])/numpy.abs(results[0][0]))
numpy.mean(numpy.abs(results[2][0] - results[3][0])/numpy.abs(results[0][0]))

results[0][0][0,40:50,40:50,1]

numpy.max(numpy.sum(results[0][0], axis=0) - results[1][0])
numpy.sum(numpy.sum(results[0][0], axis=0) != results[1][0])
numpy.unique(results[0][0])
numpy.sum(e[:,:,1,:])
st2 = time.time()
t2 = time.time() - st2

results = sess.run([indexedGradsGPU, flatGradsGPU, indexedGradsCPU, flatGradsCPU, convByIndexCudaRes, flatConvCudaRes], feed_dict={inpt: a, rotatedWeights: b, quantized: c, convMask: d, flatWeights: e, groundTruth: gt})
results = sess.run([flatGradsCPU], feed_dict={inpt: a, rotatedWeights: b, quantized: c, convMask: d, flatWeights: e, groundTruth: gt})
results = sess.run([indexedGradsCPU], feed_dict={inpt: a, rotatedWeights: b, quantized: c, convMask: d, flatWeights: e, groundTruth: gt})
results = sess.run([indexedGradsGPU, flatGradsGPU], feed_dict={inpt: a, rotatedWeights: b, quantized: c, convMask: d, flatWeights: e, groundTruth: gt})
results = sess.run([myErrorGPU], feed_dict={inpt: a, rotatedWeights: b, quantized: c, convMask: d, flatWeights: e, groundTruth: gt})
numpy.sum(results[0] != results[1])
numpy.max(results[0][0] - results[1][0])
numpy.mean(results[0][0] - results[1][0])
numpy.sum(results[2] != results[3])
numpy.max(results[2] - results[3])
numpy.mean(results[2] - results[3])
numpy.sum(results[1] != results[3])
numpy.max(results[1] - results[3])
numpy.mean(results[1] - results[3])
numpy.sum(results[0] != results[2])
numpy.max(results[0] - results[2])
numpy.mean(results[0] - results[2])

results = sess.run([convByIndexCPURes, flatConvCPURes], feed_dict={inpt: a, rotatedWeights: b, quantized: c, convMask: d, flatWeights: e})
results = sess.run([convByIndexCudaRes, convByIndexCPURes, flatConvCudaRes, flatConvCPURes], feed_dict={inpt: a, rotatedWeights: b, quantized: c, convMask: d, flatWeights: e})
results = sess.run([convByIndexCudaRes], feed_dict={inpt: a, rotatedWeights: b, quantized: c, convMask: d, flatWeights: e})
results = sess.run([convByIndexCPURes], feed_dict={inpt: a, rotatedWeights: b, quantized: c, convMask: d, flatWeights: e})
results = sess.run([flatConvCPURes], feed_dict={inpt: a, rotatedWeights: b, quantized: c, convMask: d, flatWeights: e})
results = sess.run([flatConvCudaRes], feed_dict={inpt: a, rotatedWeights: b, quantized: c, convMask: d, flatWeights: e})
results = sess.run([convByIndexCudaRes, flatConvCudaRes], feed_dict={inpt: a, rotatedWeights: b, quantized: c, convMask: d, flatWeights: e})
results = sess.run([convByIndexCudaRes, convByIndexCPURes], feed_dict={inpt: a, rotatedWeights: b, quantized: c, convMask: d, flatWeights: e})
numpy.sum(results[0] != results[1])

results[1][0]
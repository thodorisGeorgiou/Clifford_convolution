import numpy
import tensorflow as tf
import cliffordConvolution as cc

import scipy.ndimage
import time
import operator
from functools import reduce

def conv_2tf(inpt, weights, c_i, c_o, s_h, s_w, padding):
	if inpt.get_shape()[-1] % 2 != 0:
		print("conv_1 input must be a vector field")
	if inpt.get_shape()[-1] != 2*c_i:
		print("Number of inputs does not match input size")
	if weights.get_shape()[-1] != c_o:
		print("Number of outputs does not match weight tensor size")
	inptFeatMpas = tf.unstack(inpt, axis=-1)
	weightsSep = tf.unstack(weights, axis=-2)
	inptMona = tf.stack(inptFeatMpas[1::2], axis=-1)
	inptZiga = tf.stack(inptFeatMpas[::2], axis=-1)
	weightMona = tf.stack(weightsSep[1::2], axis=-2)
	weightZiga = tf.stack(weightsSep[::2], axis=-2)
	first = tf.nn.conv2d(inptZiga, weightMona, [1, s_h, s_w, 1], padding=padding)
	second = tf.nn.conv2d(inptMona, weightZiga, [1, s_h, s_w, 1], padding=padding)
	return first - second

def rotateVectorField(field, angle):
	inPlaceRotated = rotateVectors(field, angle)
	return tf.contrib.image.rotate(inPlaceRotated, angle)

def rotateVectors(vectors, theta):
	vectors = tf.transpose(vectors, [0,3,1,2])
	shape = vectors.get_shape()
	vectors = tf.reshape(vectors, [shape[0].value*shape[3].value//2*shape[1].value, shape[2].value, 2])
	rotation_matrix = tf.stack([tf.cos(theta), tf.sin(theta), -tf.sin(theta), tf.cos(theta)])
	rotation_matrix = tf.reshape(rotation_matrix, (2,2))
	rotation_matrix = tf.tile(tf.expand_dims(rotation_matrix, axis=0), [vectors.get_shape()[0],1,1])
	rotated = tf.reshape(tf.matmul(vectors, rotation_matrix), shape)
	return tf.transpose(rotated, [0,2,3,1])

def changeToCartesian(magnitude, angles):
	x = tf.multiply(magnitude, tf.cos(angles))
	y = tf.multiply(magnitude, tf.sin(angles))
	xs = tf.unstack(x, axis=-1)
	ys = tf.unstack(y, axis=-1)
	res = [None]*(len(xs)+len(ys))
	res[::2] = xs
	res[1::2] = ys
	return tf.stack(res, axis=-1)

def cliffordConvOne(inpt, weights, c_i, c_o, s_h, s_w, padding, num_angles=4, num_bins=32):
	offset = num_bins//(2*num_angles)
	angles = [tf.constant([(a-offset)*2*numpy.pi/num_bins]) for a in range(num_bins+1)]
	thetas = []
	rotatedWeights = []
	for angle in angles:
		rotatedWeights.append(rotateVectorField(weights, angle))
	rotatedWeights = tf.stack(rotatedWeights, axis=0)
	weightShape = rotatedWeights.get_shape()
	for angle in range(0, num_bins, num_bins//num_angles):
		weightSet = tf.gather(rotatedWeights, angle + offset)
		conv2 = conv_2tf(inpt, weightSet, c_i, c_o, s_h, s_w, padding)
		conv0 = tf.nn.conv2d(inpt, weightSet, [1,s_h,s_w,1], padding)
		thetas.append(tf.atan2(conv2, conv0))
	angles = tf.concat(angles, axis=0)
	thetas = tf.stack(thetas, axis=-1)
	winner = tf.argmin(tf.abs(thetas), axis=-1, output_type=tf.int32)
	thetas2 = cc.ops.reduceIndex(thetas, winner)
	thetas2, convMask = cc.ops.offsetCorrect(thetas2, [offset*2*numpy.pi/num_bins])
	quantized = tf.cast(tf.round(thetas2*num_bins/(2*numpy.pi)), tf.int32) + tf.cast(winner * (num_bins//num_angles), tf.int32) + offset
	res = cc.ops.convByIndex(inpt, rotatedWeights, quantized, convMask, [1,1,1,1], "SAME")
	return res, tf.gather(angles, quantized)

def cliffordConvTwo(inpt, weights, c_i, c_o, s_h, s_w, padding, num_angles=4, num_bins=32):
	offset = num_bins//(2*num_angles)
	angles = [tf.constant([(a-offset)*2*numpy.pi/num_bins]) for a in range(num_bins+1)]
	thetas = []
	rotatedWeights = []
	for angle in angles:
		rotatedWeights.append(rotateVectorField(weights, angle))
	rotatedWeights = tf.stack(rotatedWeights, axis=0)
	weightShape = rotatedWeights.get_shape()
	for angle in range(0, num_bins, num_bins//num_angles):
		weightSet = tf.gather(rotatedWeights, angle + offset)
		conv2 = conv_2tf(inpt, weightSet, c_i, c_o, s_h, s_w, padding)
		conv0 = tf.nn.conv2d(inpt, weightSet, [1,s_h,s_w,1], padding)
		thetas.append(tf.atan2(conv2, conv0))
	angles = tf.concat(angles, axis=0)
	thetas = tf.stack(thetas, axis=-1)
	winner = tf.argmin(tf.abs(thetas), axis=-1, output_type=tf.int32)
	thetas2 = cc.ops.reduceIndex(thetas, winner)
	thetas2, convMask = cc.ops.offsetCorrect(thetas2, [offset*2*numpy.pi/num_bins])
	quantized = tf.cast(tf.round(thetas2*num_bins/(2*numpy.pi)), tf.int32) + tf.cast(winner * (num_bins//num_angles), tf.int32) + offset
	res = cc.ops.convByIndex(inpt, rotatedWeights, quantized, convMask, [1,1,1,1], "SAME")
	return res, cc.ops.gatherAngles(angles, quantized, thetas2)

inpt = tf.placeholder(tf.float32, [128, 5, 5, 128])
weights = tf.nn.l2_normalize(tf.constant(numpy.random.rand(3,3,128,64), dtype=numpy.float32), axis=[0,1,2])
# weights2 = tf.nn.l2_normalize(tf.constant(numpy.random.rand(3,3,128,64), dtype=numpy.float32), axis=[0,1,2])

with tf.device('cpu:0'):
	convResOneCPU, anglesOneCPU = cliffordConvOne(tf.nn.l2_normalize(inpt, axis=[1,2,3]), weights, 64, 64, 1, 1, "SAME")
	# reluResOne = tf.nn.relu(convResOne)
	# cartResOne = changeToCartesian(reluResOne, anglesOne)
	# normResOne = tf.layers.batch_normalization(cartResOne)
	convResTwoCPU, anglesTwoCPU = cliffordConvTwo(tf.nn.l2_normalize(inpt, axis=[1,2,3]), weights, 64, 64, 1, 1, "SAME")
	# reluResTwo = tf.nn.relu(convResTwo)
	# cartResTwo = changeToCartesian(reluResTwo, anglesTwo)
	# normResTwo = tf.layers.batch_normalization(cartResTwo)

with tf.device('gpu:0'):
	convResOneGPU, anglesOneGPU = cliffordConvOne(tf.nn.l2_normalize(inpt, axis=[1,2,3]), weights, 64, 64, 1, 1, "SAME")
	# reluResOne = tf.nn.relu(convResOne)
	# cartResOne = changeToCartesian(reluResOne, anglesOne)
	# normResOne = tf.layers.batch_normalization(cartResOne)
	convResTwoGPU, anglesTwoGPU = cliffordConvTwo(tf.nn.l2_normalize(inpt, axis=[1,2,3]), weights, 64, 64, 1, 1, "SAME")
	# reluResTwo = tf.nn.relu(convResTwo)
	# cartResTwo = changeToCartesian(reluResTwo, anglesTwo)
	# normResTwo = tf.layers.batch_normalization(cartResTwo)



myconfig = tf.ConfigProto(log_device_placement=False)
myconfig.gpu_options.allow_growth = True
sess = tf.Session(config=myconfig)
init = tf.global_variables_initializer()
sess.run(init)
out = numpy.ones([128,5,5,128])
# out = numpy.random.rand(128,5,5,128)
res = sess.run([convResOneCPU, anglesOneCPU, convResTwoCPU, anglesTwoCPU, convResOneGPU, anglesOneGPU, convResTwoGPU, anglesTwoGPU], feed_dict={inpt: out})

numpy.sum(res[0] != res[2])
# st1 = time.time()
# res1 = sess.run([convResOne, anglesOne, reluResOne, cartResOne, normResOne], feed_dict={inpt: out})
# t1 = time.time() - st1

# st2 = time.time()
# res2 = sess.run([convResTwo, anglesTwo, reluResTwo, cartResTwo, normResTwo], feed_dict={inpt: out})
# t2 = time.time() - st2
# print(t2)

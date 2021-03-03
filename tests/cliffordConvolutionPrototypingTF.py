import tensorflow as tf
convByIndex = tf.load_op_library("./conv_by_index_2d.so")
reduce_index = tf.load_op_library("./reduce_index.so")
offset_correct = tf.load_op_library("./offset_correct.so")
# insert_zeros = tf.load_op_library("./insert_zeros.so")
# import gatherRedirectedGradients
import numpy
import scipy.ndimage
import time
import operator
from functools import reduce
# from matplotlib import pyplot as plt

def prod(iterable):
	return reduce(operator.mul, iterable, 1)

def conv_0(a,b):
	return numpy.sum(a*b)

def conv_2(a,b):
	part_sums = numpy.sum((a*(b[:,:,::-1])), axis=(0,1))
	return part_sums[0] - part_sums[1]

def rotate_vector_field_numpy(field, angle, order):
	angle = angle * numpy.pi
	cos_theta = numpy.cos(angle)
	sin_theta = numpy.sin(angle)
	r = numpy.array([[cos_theta, sin_theta],[-1*sin_theta, cos_theta]])
	rotated = numpy.matmul(field, r)
	return scipy.ndimage.interpolation.rotate(rotated, numpy.degrees(angle), reshape=False, order=order)

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


# inpt = tf.placeholder(tf.float32, [128,32,32,256])
# weights = tf.placeholder(tf.float32, [3,3,256,128])
# c_i = 128
# c_o = 128
# s_h = 1
# s_w = 1
# padding = "SAME"
# num_angles=4
# num_bins=32
# [['b0', 'b1'], ['d0', 'c1']]


def rotInvariantConv(inpt, weights, c_i, c_o, s_h, s_w, padding, num_angles=4, num_bins=32):
	angles = [tf.constant([a*2*numpy.pi/num_bins]) for a in range(num_bins)]
	rotatedWeights = []
	for angle in angles:
		rotatedWeights.append(rotateVectorField(weights, angle))
	rotatedWeights = tf.concat(rotatedWeights, axis=-1)
	conv0 = tf.nn.conv2d(inpt, rotatedWeights, [1,s_h,s_w,1], padding)
	shape = conv0.get_shape()
	conv0 = tf.transpose(tf.reshape(conv0, [shape[0],shape[1],shape[2], num_bins, c_o]), [3,0,1,2,4])
	res = tf.reduce_max(conv0, axis=0, keepdims=False)
	return res


def cliffordConv(inpt, weights, c_i, c_o, s_h, s_w, padding, num_angles=4, num_bins=32):
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
	thetas2 = reduce_index.reduce_index(thetas, winner)
	thetas2, convMask = offset_correct.offset_correct(thetas2, [offset*2*numpy.pi/num_bins])
	quantized = tf.cast(tf.round(thetas2*num_bins/(2*numpy.pi)), tf.int32) + tf.cast(winner * (num_bins//num_angles), tf.int32) + offset

	res = convByIndex.conv_by_index2d(inpt, rotatedWeights, quantized, convMask, [1,1,1,1], "SAME")
	# res = convByIndex.conv_by_index2d(inpt, rotatedWeights, quantized, convMask, indOrigine=None, [1,1,1,1], "SAME")
	# out_angles = outAngles.OutAngles(angles, quantized, thetas2)
	return res, tf.gather(angles, quantized)


def changeToCartesian(magnitude, angles):
	x = tf.multiply(magnitude, tf.cos(angles))
	y = tf.multiply(magnitude, tf.sin(angles))
	xs = tf.unstack(x, axis=-1)
	ys = tf.unstack(y, axis=-1)
	res = [None]*(len(xs)+len(ys))
	res[::2] = xs
	res[1::2] = ys
	return tf.stack(res, axis=-1)


with tf.device('gpu:0'):
	in2 = tf.placeholder(tf.float32, [128, 28, 28, 128])
	weights2 = tf.nn.l2_normalize(tf.constant(numpy.random.rand(3,3,128,64), dtype=numpy.float32), axis=[0,1,2])
	out2, angles = cliffordConv(in2, weights2, 64, 64, 1, 1, "SAME")
	out2 = tf.nn.relu(out2)
	out2 = changeToCartesian(out2, angles)
	out2 = tf.layers.batch_normalization(out2)
	for l in range(2):
		weights2 = tf.constant(numpy.random.rand(3,3,128,64), dtype=numpy.float32)
		out2, angles = cliffordConv(out2, weights2, 64, 64, 1, 1, "SAME")
		out2 = tf.nn.relu(out2)
		out2 = changeToCartesian(out2, angles)
		out2 = tf.layers.batch_normalization(out2)

with tf.device('gpu:1'):
	in1 = tf.placeholder(tf.float32, [128, 28, 28, 128])
	weights1 = tf.nn.l2_normalize(tf.constant(numpy.random.rand(3,3,128,128), dtype=numpy.float32), axis=[0,1,2])
	out1 = rotInvariantConv(in1, weights1, 128, 128, 1, 1, "SAME")
	out1 = tf.layers.batch_normalization(out1)
	out1 = tf.nn.relu(out1)
	for l in range(2):
		weights1 = tf.constant(numpy.random.rand(3,3,128,128), dtype=numpy.float32)
		out1 = rotInvariantConv(out1, weights1, 128, 128, 1, 1, "SAME")
		out1 = tf.layers.batch_normalization(out1)
		out1 = tf.nn.relu(out1)


myconfig = tf.ConfigProto(log_device_placement=False)
myconfig.gpu_options.allow_growth = True
sess = tf.Session(config=myconfig)
init = tf.global_variables_initializer()
sess.run(init)
out = numpy.random.rand(128,28,28,128)

st1 = time.time()
o = sess.run(out1, feed_dict={in1: out})
t1 = time.time() - st1
print(t1)
st2 = time.time()
o = sess.run(out2, feed_dict={in2: out})
t2 = time.time() - st2
print(str(t1)+"\n"+str(t2))

# angle = tf.placeholder(tf.float32, (2))
# tan = tf.placeholder(tf.float32, (2))



# with tf.device('gpu:0'):
# 	tftan = tf.tan(angle)
# 	tfatan = tf.atan(tan)


aa = tf.placeholder(tf.float32, [1,3,3,256])
ss = tf.placeholder(tf.float32, [3,3,256,1])
kk = conv_2tf(aa, ss, 128, 1, 1, 1, "VALID")
kk0 = tf.nn.conv2d(aa, ss, [1,1,1,1], "VALID")
ang = tf.atan(tf.divide(kk0,kk))


sess = tf.Session()
angle = numpy.zeros((3,10000), dtype=numpy.float32)
for rr in range(10000):
	print(rr)
	b = numpy.zeros([3,3,128], dtype=numpy.float32)
	c = numpy.zeros([3,3,128], dtype=numpy.float32)
	bb = numpy.zeros([3,3,128], dtype=numpy.float32)
	cc = numpy.zeros([3,3,128], dtype=numpy.float32)
	pfff = numpy.zeros([1,3,3,256], dtype=numpy.float32)
	pfffWeights = numpy.zeros([3,3,256,1], dtype=numpy.float32)
	c0=[]
	c2=[]
	for k in range(128):
		a = numpy.ones(shape=[3,3,2], dtype=numpy.float32)
		r = numpy.random.rand(3,3).astype(numpy.float32) * 2 * numpy.pi
		r2 = numpy.random.rand(3,3).astype(numpy.float32)
		a[:,:,0] = numpy.cos(r) * r2
		a[:,:,1] = numpy.sin(r) * r2
		e = (a / numpy.linalg.norm(a)).astype(numpy.float32)
		f = a  + (numpy.random.rand(3,3,2) - 0.5)*0.01
		g = rotate_vector_field_numpy(f, 1.0/4, 2)
		b[:,:,k] = e[:,:,0]
		c[:,:,k] = e[:,:,1]
		bb[:,:,k] = g[:,:,0]
		cc[:,:,k] = g[:,:,1]
		pfff[0,:,:,2*k:2*k+2] = e
		pfffWeights[:,:,2*k:2*k+2, 0] = g
	# 	c2.append(conv_2(e.astype(numpy.float32),g.astype(numpy.float32)))
	# 	c0.append(conv_0(e.astype(numpy.float32),g.astype(numpy.float32)))
	# tan1 = numpy.sum(c2)/numpy.sum(c0)
	mm = numpy.sum(b*bb)
	zz = numpy.sum(c*cc)
	mz = numpy.sum(b*cc)
	zm = numpy.sum(bb*c)
	cc0 = mm + zz
	cc2 = mz - zm
	tan2 = cc0/cc2
	tfconv22d, tfconv2d, tfangles = sess.run([kk, kk0, ang], feed_dict={aa: pfff, ss: pfffWeights})
	pfffWeights2 = (pfffWeights/numpy.linalg.norm(pfffWeights)).astype(numpy.float32)
	tfconv22d2, tfconv2d2, tfangles2 = sess.run([kk, kk0, ang], feed_dict={aa: pfff, ss: pfffWeights2})
	angle[0,rr] = numpy.arctan(tan2)
	angle[1,rr] = tfangles
	angle[2,rr] = tfangles2


numpy.mean(angle[0])
numpy.std(angle[0])
numpy.mean(angle[1])
numpy.std(angle[1])
numpy.mean(angle[2])
numpy.std(angle[2])

hist0 = numpy.zeros(32)
hist1 = numpy.zeros(32)
hist2 = numpy.zeros(32)
quantized0 = numpy.round(angle[0]*32/(2*numpy.pi))
quantized1 = numpy.round(angle[1]*32/(2*numpy.pi))
quantized2 = numpy.round(angle[2]*32/(2*numpy.pi))
q0 = quantized0.astype(numpy.int32)
q1 = quantized1.astype(numpy.int32)
q2 = quantized2.astype(numpy.int32)
numpy.add.at(hist0, q0, 1)
numpy.add.at(hist1, q1, 1)
numpy.add.at(hist2, q2, 1)

fr, = plt.plot(hist0, label="1")
mz, = plt.plot(hist1, label="2")
tef, = plt.plot(hist2, label="3")
plt.legend(handles=[fr, mz, tef])
plt.show()


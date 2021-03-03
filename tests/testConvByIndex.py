import numpy
import tensorflow as tf
import cliffordConvolution as cc

import scipy.ndimage
import time
import operator
from functools import reduce

def manual_rotate_vector_field(field, angle):
	angle = angle * numpy.pi
	cos_theta = numpy.cos(angle)
	sin_theta = numpy.sin(angle)
	r = numpy.array([[cos_theta, sin_theta],[-1*sin_theta, cos_theta]])
	rotated = numpy.matmul(field, r)
	center = numpy.array([1,1], dtype=numpy.float32)
	c = numpy.zeros(shape=field.shape, dtype=numpy.float32)
	cos_theta = numpy.cos(-angle)
	sin_theta = numpy.sin(-angle)
	r = numpy.array([[cos_theta, sin_theta],[-1*sin_theta, cos_theta]])
	for i in range(field.shape[0]):
		for j in range(field.shape[1]):
			_coor = numpy.array([i,j], dtype=numpy.float32)
			nCoor = numpy.matmul(_coor - center, r) + center
			coor = _coor.astype(numpy.int32)
			c[coor[0], coor[1],0] = bilinear_interpolation(rotated[:,:,0], nCoor)
			c[coor[0], coor[1],1] = bilinear_interpolation(rotated[:,:,1], nCoor)
	return c

def bilinear_interpolation(inpt, nCoor):
	# if numpy.any(numpy.logical_or(nCoor+1 > inpt.shape[:2], nCoor < 0)):
	# 	return 0
	a = numpy.zeros(shape=numpy.array(inpt.shape)+2, dtype=numpy.float32)
	a[1:inpt.shape[0]+1, 1:inpt.shape[1]+1] = inpt
	x1 = numpy.floor(nCoor[0]).astype(numpy.int32)
	y1 = numpy.floor(nCoor[1]).astype(numpy.int32)
	x = numpy.array([x1+1 - nCoor[0], nCoor[0] - x1])
	y = numpy.array([y1+1 - nCoor[1], nCoor[1] - y1])
	k = numpy.matmul(x,a[x1+1:x1+3, y1+1:y1+3])
	return numpy.matmul(k, numpy.transpose(y))


batch = 256
rows=56
cols=56
in_depth=64
out_depth=64
filter_rows=3
filter_cols=3
num_angles=32

# a = numpy.ones(shape=[128,56,56,64], dtype=numpy.float32)	#input
b = numpy.zeros(shape=[batch,rows,cols,out_depth], dtype=numpy.int32)	#indexes
e = numpy.zeros(shape=[filter_rows,filter_cols,in_depth,out_depth], dtype=numpy.float32)		#flatWeights
a = numpy.random.rand(batch,rows,cols,in_depth).astype(numpy.float32)		#input
w = numpy.zeros([num_angles+1,filter_rows,filter_cols,in_depth,out_depth], dtype=numpy.float32)			#weights
# b = (numpy.random.rand(batch,rows,cols,out_depth)*(num_angles+1)).astype(numpy.int32)	#indexes
d = numpy.ones(shape=[batch,rows,cols,out_depth], dtype=numpy.int32)		#mask
# e = numpy.random.rand(filter_rows,filter_cols,in_depth,out_depth).astype(numpy.float32)		#flatWeights
f = numpy.ones(shape=[batch,rows,cols,out_depth], dtype=numpy.float32)		#mask

for i in range(out_depth):
	weights = numpy.random.rand(filter_rows, filter_cols, in_depth)
	w[i%33,:,:,:,i] = weights
	b[:,:,:,i] = i%33
	e[:,:,:,i] = weights


# init_weights = numpy.ones([3,3,10,10], dtype=numpy.float32)
init_weights = numpy.random.rand(filter_rows,filter_cols,in_depth,out_depth).astype(numpy.float32)
for dout in range(init_weights.shape[3]):
	for din in range(0,init_weights.shape[2],2):
		field = init_weights[:,:,din:din+2,dout]
		for angle in range(num_angles):
			an = angle*numpy.pi*2/num_angles
			w[angle,:,:,din:din+2,dout] = manual_rotate_vector_field(field, an)
			if angle==0: w[-1,:,:,din:din+2,dout] = w[angle,:,:,din:din+2,dout]



inpt = tf.placeholder(tf.float32, [batch,rows,cols,in_depth])
rotatedWeights = tf.placeholder(tf.float32, [num_angles+1,filter_rows,filter_cols,in_depth,out_depth])
quantized = tf.placeholder(tf.int32, [batch,rows,cols,out_depth])
convMask = tf.placeholder(tf.int32, [batch,rows,cols,out_depth])
angles = tf.placeholder(tf.float32, [batch,rows,cols,out_depth])
flatWeights = tf.placeholder(tf.float32, [filter_rows,filter_cols,in_depth,out_depth])
with tf.device('gpu:0'):
	convByIndexCudaRes = cc.ops.convByIndex(inpt, rotatedWeights, quantized, convMask, angles, [1,1,1,1], "SAME")
	flatConvCudaRes = tf.nn.conv2d(inpt, flatWeights, [1,1,1,1], "SAME")

with tf.device('cpu:0'):
	convByIndexCPURes = cc.ops.convByIndex(inpt, rotatedWeights, quantized, convMask, angles, [1,1,1,1], "SAME")
	flatConvCPURes = tf.nn.conv2d(inpt, flatWeights, [1,1,1,1], "SAME")


myconfig = tf.ConfigProto(log_device_placement=False)
myconfig.gpu_options.allow_growth = True
sess = tf.Session(config=myconfig)

st1 = time.time()
results = sess.run([convByIndexCudaRes], feed_dict={inpt: a, rotatedWeights: w, quantized: b, convMask: d, flatWeights: e, angles: f})
t1 = time.time() - st1

st2 = time.time()
results = sess.run([flatConvCudaRes], feed_dict={inpt: a, rotatedWeights: w, quantized: b, convMask: d, flatWeights: e})
t2 = time.time() - st2

results = sess.run([convByIndexCudaRes, convByIndexCPURes, flatConvCudaRes, flatConvCPURes], feed_dict={inpt: a, rotatedWeights: w, quantized: b, convMask: d, flatWeights: e, angles: f})

t1
t2
numpy.max(numpy.abs(results[0] - results[1]))
numpy.mean(numpy.abs(results[0] - results[1]))
numpy.max(numpy.abs(results[2] - results[3]))
numpy.mean(numpy.abs(results[2] - results[3]))
numpy.max(numpy.abs(results[1] - results[3]))
numpy.mean(numpy.abs(results[1] - results[3]))
numpy.max(numpy.abs(results[0] - results[2]))
numpy.mean(numpy.abs(results[0] - results[2]))

# numpy.max(numpy.abs(results[2] - results[3]))
# numpy.mean(numpy.abs(results[2] - results[3]))
# numpy.max(numpy.abs(results[0] - results[2]))
# numpy.mean(numpy.abs(results[0] - results[2]))

numpy.sum(results[0] != results[1])
numpy.sum(results[2] != results[3])
numpy.max(results[2] - results[3])
numpy.mean(results[2] - results[3])
numpy.sum(results[1] != results[3])
numpy.max(results[1] - results[3])
numpy.mean(results[1] - results[3])
numpy.sum(results[0] != results[2])
numpy.max(results[0] - results[2])
numpy.mean(results[0] - results[2])

results = sess.run([convByIndexCPURes], feed_dict={inpt: a, rotatedWeights: b, quantized: c, convMask: d, flatWeights: e})
results = sess.run([flatConvCPURes], feed_dict={inpt: a, rotatedWeights: b, quantized: c, convMask: d, flatWeights: e})
results = sess.run([convByIndexCudaRes, flatConvCudaRes], feed_dict={inpt: a, rotatedWeights: b, quantized: c, convMask: d, flatWeights: e})
results = sess.run([convByIndexCudaRes, convByIndexCPURes], feed_dict={inpt: a, rotatedWeights: b, quantized: c, convMask: d, flatWeights: e})
numpy.sum(results[0] != results[1])

results[1][0]

4.54741358757019 - 1.2477607727050781
4.515273332595825 - 1.2423021793365479
3.988178253173828 - 1.2393715381622314
3.9707512855529785 - 1.2407095432281494
3.970672607421875 - 1.2421042919158936
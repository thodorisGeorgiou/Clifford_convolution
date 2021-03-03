import numpy
import tensorflow as tf
import cliffordConvolution as cc
import scipy.ndimage
import time
import operator
from functools import reduce

inpt2 = tf.placeholder(tf.float32, [32,56,56,32,4])
indexes2 = tf.placeholder(tf.int32, [32,56,56,32])

with tf.device('gpu:0'):
	out2 = cc.ops.reduceIndex(inpt2, indexes2)

with tf.device('cpu:0'):
	out = cc.ops.reduceIndex(inpt2, indexes2)


inum = numpy.ones([32,56,56,32], dtype=numpy.int32)
ii = numpy.zeros([32,56,56,32,4], dtype=numpy.float32)
n = 1
for i in range(inum.shape[0]):
	for j in range(inum.shape[1]):
		for k in range(inum.shape[2]):
			for l in range(inum.shape[3]):
				ind = (numpy.random.rand(1)*4).astype(numpy.int32)[0]
				ii[i,j,k,l,ind] = n
				inum[i,j,k,l] = ind
				n += 1

# ii[:,:,:,:,0] = 10
# ii[:,:,:,:,1] = 0
# ii[:,:,:,:,2] = 5
# ii[:,:,:,:,3] = 20
# i[0] = 0
# i[1] = 1
# i[2] = 2
# i[3] = 3


sess = tf.Session()

a = sess.run(out, feed_dict={inpt2: ii, indexes2: inum})
a2 = sess.run(out2, feed_dict={inpt2: ii, indexes2: inum})

numpy.sum(a != a2)

for i in range(a2.shape[0]):
	for j in range(a2.shape[1]):
		for k in range(a2.shape[2]):
			for l in range(a2.shape[3]):
				print(a2[i,j,k,l])

for i in range(ii.shape[0]):
	for j in range(ii.shape[1]):
		for k in range(ii.shape[2]):
			for l in range(ii.shape[3]):
				for m in range(ii.shape[4]):
					print(ii[i,j,k,l,m])


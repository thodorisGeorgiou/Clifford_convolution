import tensorflow as tf
import numpy
import cliffordConvolution as cc
import scipy.ndimage
import time
import operator
from functools import reduce


# insert_zeros = tf.load_op_library("./insert_zeros.so")
# a = insert_zeros.insert_zeros(numpy.ones([100,30,30,100], dtype=numpy.float32), numpy.array([[21,3,15,90]]).astype(numpy.int32))
a = numpy.random.rand(128,64,64,1024).astype(numpy.float32)
offset = 0.8
k = tf.placeholder(tf.float32, [128,64,64,1024])
with tf.device('gpu:0'):
	c = cc.ops.offsetCorrect(k, [offset])

with tf.device('cpu:0'):
	c2 = cc.ops.offsetCorrect(k, [offset])


myconfig = tf.ConfigProto(log_device_placement=False)
myconfig.gpu_options.allow_growth = True
sess = tf.Session(config=myconfig)


st1 = time.time()
b2 = sess.run(c2, feed_dict={k: a})
t1 = time.time() - st1
st2 = time.time()
b = sess.run(c, feed_dict={k: a})
t2 = time.time() - st2

numpy.sum(a>=offset)
numpy.sum(b[0]==0)
numpy.sum(b2[0]==0)

0.6734054088592529
0.6560304164886475

3.975929021835327
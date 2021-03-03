import numpy
import tensorflow as tf
test = tf.load_op_library("calculate_proportions.so").calculate_proportions

a = numpy.ones([10,10,10,10], dtype=numpy.float32)
b = numpy.zeros([10,10,10,10], dtype=numpy.int32)

for i in range(10):
	b[:,:,:,i] = i
	d = 0
	for ii in range(10):
		for iii in range(10):
			for iiii in range(10):
				a[ii,iii,iiii,i] = numpy.random.rand()
				# a[ii,iii,iiii,i] = float(d*i)
				d+=1

aa = tf.placeholder(tf.float32, [10,10,10,10])
bb = tf.placeholder(tf.int32, [10,10,10,10])
an = tf.placeholder(tf.float32, [10,3,3,10,10])
with tf.device('cpu:0'):
	c = test(aa,bb,an)

with tf.device('gpu:0'):
	e = test(aa,bb,an)

# init = tf.global_variables_initializer()
sess = tf.Session()
# sess.run(init)
res = sess.run([c, e], feed_dict={aa: a, bb: b, an: numpy.ones(shape=[10,3,3,10,10], dtype=numpy.float32)})

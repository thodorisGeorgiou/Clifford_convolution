import numpy
import tensorflow as tf
gatherAngles = tf.load_op_library("./gather_angles.so")
import time


# insert_zeros = tf.load_op_library("./insert_zeros.so")
# a = insert_zeros.insert_zeros(numpy.ones([100,30,30,100], dtype=numpy.float32), numpy.array([[21,3,15,90]]).astype(numpy.int32))
a = (numpy.random.rand(384,64,64,384)*10).astype(numpy.int32)
b = numpy.array([1/(i+1) for i in range(10)])
c = numpy.random.rand(384,64,64,384).astype(numpy.float32)

k = tf.placeholder(tf.int32, [384,64,64,384])
l = tf.placeholder(tf.float32, [10])
m = tf.placeholder(tf.float32, [384,64,64,384])

with tf.device('cpu:0'):
	d = gatherAngles.gather_angles(l, k, m)
	e = tf.gather(l,k)

with tf.device('gpu:0'):
	f = gatherAngles.gather_angles(l, k, m)
	g = tf.gather(l,k)

# with tf.device('gpu:0'):
# 	c = offset_correct.offset_correct(k, [offset])

myconfig = tf.ConfigProto(log_device_placement=False)
myconfig.gpu_options.allow_growth = True
sess = tf.Session(config=myconfig)


st1 = time.time()
r = sess.run(d, feed_dict={k: a, l: b, m: c})
t1 = time.time() - st1

st2 = time.time()
rr = sess.run(e, feed_dict={k: a, l: b, m: c})
t2 = time.time() - st2

st4 = time.time()
rrrr = sess.run(g, feed_dict={k: a, l: b, m: c})
t4 = time.time() - st4

st3 = time.time()
rrr = sess.run(f, feed_dict={k: a, l: b, m: c})
t3 = time.time() - st3



print(str(t1)+" "+str(t2)+" "+str(t3)+" "+str(t4))

numpy.sum(r != rr)
r, rr = sess.run([d,e], feed_dict={k: a, l: b, m: c})

b2 = sess.run(c2, feed_dict={k: a})


0.932471513748169 1.1623539924621582 3.120055675506592 2.700204610824585
0.9254395961761475 1.2072558403015137 3.0362231731414795 2.6977696418762207
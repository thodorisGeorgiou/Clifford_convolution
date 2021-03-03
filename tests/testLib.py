import numpy
import tensorflow as tf
import cliffordConvolution as cc
import time

inpt = tf.placeholder(tf.float32, [128, 64, 64, 128])
weights = tf.nn.l2_normalize(tf.constant(numpy.random.rand(3,3,128,64), dtype=numpy.float32), axis=[0,1,2])
# weights2 = tf.nn.l2_normalize(tf.constant(numpy.random.rand(3,3,128,64), dtype=numpy.float32), axis=[0,1,2])

with tf.device('cpu:0'):
# 	convResOneCPU, anglesOneCPU = cc.layers.cliffordConvOld(tf.nn.l2_normalize(inpt, axis=[1,2,3]), weights, 64, 64, 1, 1, "SAME")
# 	reluResOne = tf.nn.relu(convResOneCPU)
# 	cartResOne = cc.tranformations.changeToCartesian(reluResOne, anglesOneCPU)
# 	normResOne = tf.layers.batch_normalization(cartResOne)
	convResTwoCPU, anglesTwoCPU = cc.layers.cliffordConv(tf.nn.l2_normalize(inpt, axis=[1,2,3]), weights, 64, 64, 1, 1, "SAME")
	reluResTwo = tf.nn.relu(convResTwoCPU)
	cartResTwo = cc.tranformations.changeToCartesian(reluResTwo, anglesTwoCPU)
	normResTwo = tf.layers.batch_normalization(cartResTwo)

with tf.device('gpu:0'):
	# convResOneGPU, anglesOneGPU = cc.layers.cliffordConvOld(tf.nn.l2_normalize(inpt, axis=[1,2,3]), weights, 64, 64, 1, 1, "SAME")
	# reluResOneGPU = tf.nn.relu(convResOneGPU)
	# cartResOneGPU = cc.tranformations.changeToCartesian(reluResOneGPU, anglesOneGPU)
	# normResOneGPU = tf.layers.batch_normalization(cartResOneGPU)
	convResTwoGPU, anglesTwoGPU = cc.layers.cliffordConv(tf.nn.l2_normalize(inpt, axis=[1,2,3]), weights, 64, 64, 1, 1, "SAME")
	reluResTwoGPU = tf.nn.relu(convResTwoGPU)
	cartResTwoGPU = cc.tranformations.changeToCartesian(reluResTwoGPU, anglesTwoGPU)
	normResTwoGPU = tf.layers.batch_normalization(cartResTwoGPU)



myconfig = tf.ConfigProto(log_device_placement=False)
myconfig.gpu_options.allow_growth = True
sess = tf.Session(config=myconfig)
init = tf.global_variables_initializer()
sess.run(init)
out = numpy.random.rand(128,64,64,128)
# out = numpy.ones([128,64,64,128])
# out = numpy.random.rand(128,5,5,128)

st2 = time.time()
res2 = sess.run([convResTwoGPU, anglesTwoGPU, convResTwoCPU, anglesTwoCPU], feed_dict={inpt: out})
t2 = time.time() - st2
print(t2)

numpy.sum(res[0] != res[2])
st1 = time.time()
res1 = sess.run([normResOne, normResTwo, normResOneGPU, normResTwoGPU], feed_dict={inpt: out})
t1 = time.time() - st1
res = sess.run([convResOneCPU, anglesOneCPU, convResTwoCPU, anglesTwoCPU, convResOneGPU, anglesOneGPU, convResTwoGPU, anglesTwoGPU], feed_dict={inpt: out})
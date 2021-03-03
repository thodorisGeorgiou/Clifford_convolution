import numpy
import tensorflow as tf
import cliffordConvolution as cc

num_bins = 32
num_angles = 4
offset = num_bins//(2*num_angles)
angles = [tf.Variable([(a-offset)*2*numpy.pi/num_bins]) for a in range(num_bins+1)]

weights = tf.placeholder(tf.float32, shape=[3,3,10,10])
inds = tf.placeholder(tf.int32, shape=[10,5,5,10])
inpt1 = tf.placeholder(tf.float32, shape=[10,5,5,10])
inpt2 = tf.placeholder(tf.float32, shape=[10,5,5,10])
i = tf.placeholder(tf.float32, shape=[10,5,5,10])
# rotatedWeights = []
# for angle in angles:
# 	rotatedWeights.append(cc.tranformations.rotateVectorField(weights, angle))

# rotatedWeights = tf.stack(rotatedWeights, axis=0)

rotWeights = tf.contrib.image.rotate(weights, angles[4])


# thetas = tf.concat(angles, axis=0)
# conv0 = []
# for angle in range(0, num_bins, num_bins//num_angles):
# 	weightSet = tf.gather(rotatedWeights, angle + offset)
# 	conv0.append(tf.nn.conv2d(inpt1, weightSet, [1,1,1,1], "SAME"))

# res = conv0[0]
# for c in conv0[1:]:
# 	res += c

res = tf.nn.conv2d(inpt1, rotWeights, [1,1,1,1], "SAME")

e = tf.reduce_sum(i-res)

# grads = []
# for c in conv0:
# 	grads.append(tf.gradients(c, angles[1]))

error_grads = tf.gradients(e,weights)

ww = numpy.ones(shape=[3,3,10,10], dtype=numpy.float32)
ii = numpy.ones(shape=[10,5,5,10], dtype=numpy.int32)
i1 = numpy.ones(shape=[10,5,5,10], dtype=numpy.float32)/2
i2 = numpy.ones(shape=[10,5,5,10], dtype=numpy.float32)
i3 = numpy.ones(shape=[10,5,5,10], dtype=numpy.float32)*0.2

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

gr = sess.run(error_grads, feed_dict={weights:ww, inds:ii, inpt1:i1, i:i3})

gr = sess.run(grads+[error_grads], feed_dict={weights:ww, inds:ii, inpt1:i1, i:i3})
ee = sess.run(e, feed_dict={weights:ww, inds:ii, inpt1:i1, i:i3})

# er, mgt = sess.run([e, my_grad_thetas], feed_dict={b:bb, i:ii, t:tt})
# er, mgt, eg = sess.run([e, my_grad_thetas, error_grads], feed_dict={b:bb, i:ii, t:tt})

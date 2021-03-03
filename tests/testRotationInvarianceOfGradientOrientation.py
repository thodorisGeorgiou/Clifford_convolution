import os
import sys
import numpy
sys.path.append("/tank/georgioutk/cliffordConvolutionRegAngGrads/")
import tensorflow as tf
import cliffordConvolution as cc
import time
import pickle
import scipy
from matplotlib import pyplot as plt

def getGuessValue(kerStd,posX,posY):
	return 1./(2.*numpy.pi*(numpy.power(kerStd,2)))*numpy.exp(-(numpy.power(posX,2)+numpy.power(posY,2))/(2.*(numpy.power(kerStd,2))))

def getGuessKernel(kerStd):
	K11=getGuessValue(kerStd,-1,1)
	K12=getGuessValue(kerStd,0,1)
	K13=getGuessValue(kerStd,1,1)
	K21=getGuessValue(kerStd,-1,0)
	K22=getGuessValue(kerStd,0,0)
	K23=getGuessValue(kerStd,1,0)
	K31=getGuessValue(kerStd,-1,-1)
	K32=getGuessValue(kerStd,0,-1)
	K33=getGuessValue(kerStd,1,-1)
	kernel = tf.expand_dims(tf.expand_dims(tf.constant(numpy.array([[ K11, K12, K13],[ K21, K22, K23],[ K31, K32, K33]]),dtype=tf.float32),axis=-1),axis=-1)#3*3*4*4
	# zeros = tf.zeros_like(kernel)
	# k0 = tf.concat([kernel, zeros, zeros], axis=-2)
	# k1 = tf.concat([zeros, kernel, zeros], axis=-2)
	# k2 = tf.concat([zeros, zeros, kernel], axis=-2)
	# return tf.concat([k0, k1, k2], axis=-1)
	return kernel

kernel = getGuessKernel(1)


def plot_field(field):
	fig, axes = plt.subplots(nrows=field.shape[0], ncols=field.shape[1])
	for i in range(field.shape[0]):
		for j in range(field.shape[1]):
			axes[i,j].quiver(0, 0, field[i,j,0], -field[i,j,1], angles='xy', scale_units='xy', scale=1)
			axes[i,j].set_xlim(-1.5, 1.5)
			axes[i,j].set_ylim(-1.5, 1.5)
			axes[i,j].set_xticks([])
			axes[i,j].set_yticks([])
	plt.subplots_adjust(wspace=0, hspace=0)
	# plt.show()

# cifar = tf.keras.datasets.cifar10
# (x_train, y_train),(x_test, y_test) = cifar.load_data()
# x_train, x_test = (x_train / 255.0).astype(numpy.float32), (x_test / 255.0).astype(numpy.float32)
# y_train, y_test = y_train.astype(numpy.int32), y_test.astype(numpy.int32)

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = numpy.expand_dims((x_train / 255.0).astype(numpy.float32), -1), numpy.expand_dims((x_test / 255.0).astype(numpy.float32), -1)
y_train, y_test = numpy.expand_dims(y_train.astype(numpy.int32), -1), numpy.expand_dims(y_test.astype(numpy.int32), -1)

angle = tf.placeholder(tf.float32, [])

#block one
# grads0 = tf.placeholder(tf.float32, [10000, 32, 32, 3])
# grads1 = tf.placeholder(tf.float32, [10000, 32, 32, 3])
# xs = tf.unstack(grads0, axis=-1)
# ys = tf.unstack(grads1, axis=-1)
# res = [None]*(len(xs)+len(ys))
# res[::2] = xs
# res[1::2] = ys
# grads = tf.stack(res, axis=-1)
# rotSecond = cc.transformations.rotateVectorField(avgrads, angle, irelevantAxisFirst=True)
# sepGrads = tf.split(avgrads, 2, -1)

#block two
images = tf.placeholder(tf.float32, [None,None,None,None])
avImage = tf.layers.average_pooling2d(images, [3,3], [1,1], padding='SAME')
rotImages = tf.contrib.image.rotate(images, angle)
avgrads = tf.layers.average_pooling2d(images, [3,3], [1,1], padding='SAME')

#block three
# numpyWx = numpy.zeros([3,3], dtype=numpy.float32)
# numpyWx[0,0] = -numpy.sqrt(2)/10
# numpyWx[0,2] = numpy.sqrt(2)/10
# numpyWx[2,2] = numpy.sqrt(2)/10
# numpyWx[2,0] = -numpy.sqrt(2)/10
# numpyWx[1,0] = -3/10
# numpyWx[1,2] = 3/10
# numpyWy = numpy.zeros([3,3], dtype=numpy.float32)
# numpyWy[0,0] = -numpy.sqrt(2)/10
# numpyWy[0,2] = -numpy.sqrt(2)/10
# numpyWy[2,2] = numpy.sqrt(2)/10
# numpyWy[2,0] = numpy.sqrt(2)/10
# numpyWy[0,1] = -3/10
# numpyWy[2,1] = 3/10
# wx = tf.constant(numpyWx, dtype=tf.float32)
# wy = tf.constant(numpyWy, dtype=tf.float32)
# wz = tf.zeros_like(wx)
# wx0 = tf.stack([wx, wz, wz], axis=-1)
# wx1 = tf.stack([wz, wx, wz], axis=-1)
# wx2 = tf.stack([wz, wz, wx], axis=-1)
# wy0 = tf.stack([wy, wz, wz], axis=-1)
# wy1 = tf.stack([wz, wy, wz], axis=-1)
# wy2 = tf.stack([wz, wz, wy], axis=-1)
# w = tf.stack([wy0, wx0, wy1, wx1, wy2, wx2], axis=-1)
images3 = tf.placeholder(tf.float32, [None, 28, 28, 1])
# grads3 = tf.nn.conv2d(images3, w, [1,1,1,1], "SAME")
grads3 = cc.layers.calculateImageGradients(images3)


#normalize
fs = tf.placeholder(tf.float32, [])
field = tf.placeholder(tf.float32,[10000, 28, 28, 2])
normField = cc.layers.normalizeVectorField(field, fs, fs)

#rotate vector field
rotated = cc.transformations.rotateVectorField(field, angle, irelevantAxisFirst=True)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

a = 0.33*numpy.pi
#Rotate first
rotatedImages = sess.run(rotImages, feed_dict={images: x_test, angle: a})
rfg_test = numpy.gradient(rotatedImages, axis=[1,2])
rotFirstGrads = sess.run(avgrads, feed_dict={grads0: rfg_test[0], grads1: rfg_test[1]})
rotFirstGrads = sess.run(normField, feed_dict={field: rotFirstGrads})

#Rotate second
averageImages = sess.run(avImage, feed_dict={images: x_test})
g_test = numpy.gradient(averageImages, axis=[1,2])
rotSecondGrads = sess.run(rotSecond, feed_dict={grads0: g_test[0], grads1: g_test[1], angle: a})
rotSecondGrads = sess.run(normField, feed_dict={field: rotSecondGrads})



a = 0.33*numpy.pi
#Rotate first tf
rotatedImages = sess.run(rotImages, feed_dict={images: x_test, angle: a})
rotFirstGrads = sess.run(grads3, feed_dict={images3: rotatedImages})
# rotFirstGrads = sess.run(normField, feed_dict={field: rotFirstGrads, fs: 5})
rotFirstGrads = sess.run(avgrads, feed_dict={images: rotFirstGrads})

#Rotate second tf
averageImages = sess.run(avImage, feed_dict={images: x_test})
rotSecondGrads = sess.run(grads3, feed_dict={images3: x_test})
rotSecondGrads = sess.run(rotated, feed_dict={field: rotSecondGrads, angle: a})
# rotSecondGrads = sess.run(normField, feed_dict={field: rotSecondGrads, fs:5})
rotSecondGrads = sess.run(avgrads, feed_dict={images: rotSecondGrads})

numpy.mean(numpy.abs(rotFirstGrads[0,:,:,0]-rotSecondGrads[0,:,:,0]))
rotSecondGrads[0,10:15,13:17,1]
rotFirstGrads[0,10:15,13:17,1]


#No rotation tf
noRotGrads = sess.run(grads3, feed_dict={images3: x_test})
noRotGrads = sess.run(normField, feed_dict={field: noRotGrads, fs: 5})



rotFirstGrads[rotFirstGrads<1e-2] = 0
rotSecondGrads[rotSecondGrads<1e-2] = 0
n1 = numpy.abs(rotFirstGrads)
n1[n1==0] = 1
n2 = numpy.abs(rotSecondGrads)
n2[n2==0] = 1
numpy.mean(numpy.abs((rotSecondGrads-rotFirstGrads)/n1))
numpy.mean(numpy.abs((rotSecondGrads-rotFirstGrads)/n2))

# numpy.mean(numpy.abs(rotFirstGrads))
plot_field(rotFirstGrads[0,13:19,13:19,0:2])
plot_field(rotSecondGrads[0,13:19,13:19,0:2])
plt.show()

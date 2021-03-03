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

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = numpy.expand_dims((x_train / 255.0).astype(numpy.float32), -1), numpy.expand_dims((x_test / 255.0).astype(numpy.float32), -1)
y_train, y_test = y_train.astype(numpy.int32), y_test.astype(numpy.int32)

cifar = tf.keras.datasets.cifar10
(x_train, y_train),(x_test, y_test) = cifar.load_data()
x_train, x_test = (x_train / 255.0).astype(numpy.float32), (x_test / 255.0).astype(numpy.float32)
y_train, y_test = y_train.astype(numpy.int32), y_test.astype(numpy.int32)

g_train = numpy.gradient(x_train, axis=[1,2])
g_test = numpy.gradient(x_test, axis=[1,2])

grads0 = tf.placeholder(tf.float32, [None, None, None, None])
grads1 = tf.placeholder(tf.float32, [None, None, None, None])
# grads0 = tf.placeholder(tf.float32, [60000,28,28,1])
# grads1 = tf.placeholder(tf.float32, [60000,28,28,1])
grads = tf.concat([grads0, grads1], axis=-1)
avgrads = tf.layers.average_pooling2d(grads, [3,3], [1,1], padding='SAME')
rotSecond = cc.transformations.rotateVectorField(avgrads, angle, irelevantAxisFirst=True)
sepGrads = tf.split(avgrads, 2, -1)

sess = tf.Session()
# ag_train = sess.run(sepGrads, feed_dict={grads0: g_train[0], grads1:g_train[1]})
# ag_test = sess.run(sepGrads, feed_dict={grads0: g_test[0], grads1:g_test[1]})

cg_train = sess.run(grads, feed_dict={grads0: g_train[0], grads1:g_train[1]})
cg_test = sess.run(grads, feed_dict={grads0: g_test[0], grads1:g_test[1]})

mask_train = numpy.sqrt(ag_train[0]**2 + ag_train[1]**2) < 1e-2
mask_test = numpy.sqrt(ag_test[0]**2 + ag_test[1]**2) < 1e-2

mask_train[:,:,0,:] = False
mask_train[:,:,-1,:] = False
mask_train[:,0,:,:] = False
mask_train[:,-1,:,:] = False
mask_test[:,:,0,:] = False
mask_test[:,:,-1,:] = False
mask_test[:,0,:,:] = False
mask_test[:,-1,:,:] = False
# ag_trainV = numpy.concatenate(ag_train, axis=-1)

ag_train[0][mask_train] = numpy.nan
ag_train[1][mask_train] = numpy.nan
ag_test[0][mask_test] = numpy.nan
ag_test[1][mask_test] = numpy.nan

ag_trainWithMask = [numpy.ma.masked_invalid(ag_train[0]), numpy.ma.masked_invalid(ag_train[1])]
ag_testWithMask = [numpy.ma.masked_invalid(ag_test[0]), numpy.ma.masked_invalid(ag_test[1])]

x = numpy.arange(0, ag_train[0].shape[2])
y = numpy.arange(0, ag_train[0].shape[1])

xx, yy = numpy.meshgrid(x, y)

ag_trainInterpolated = [numpy.zeros(shape=ag_train[0].shape), numpy.zeros(shape=ag_train[1].shape)]
for index in range(ag_trainWithMask[0].shape[0]):
	example = ag_trainWithMask[0][index,:,:,0]
	#get only the valid values
	x1 = xx[~example.mask]
	y1 = yy[~example.mask]
	newarr = example[~example.mask]
	GD1 = scipy.interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='cubic')
	ag_trainInterpolated[0][index,:,:,0] = GD1

for index in range(ag_trainWithMask[1].shape[0]):
	example = ag_trainWithMask[1][index,:,:,0]
	#get only the valid values
	x1 = xx[~example.mask]
	y1 = yy[~example.mask]
	newarr = example[~example.mask]
	GD1 = scipy.interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='cubic')
	ag_trainInterpolated[1][index,:,:,0] = GD1

ag_testInterpolated = [numpy.zeros(shape=ag_test[0].shape), numpy.zeros(shape=ag_test[1].shape)]
for index in range(ag_testWithMask[0].shape[0]):
	example = ag_testWithMask[0][index,:,:,0]
	#get only the valid values
	x1 = xx[~example.mask]
	y1 = yy[~example.mask]
	newarr = example[~example.mask]
	GD1 = scipy.interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='cubic')
	ag_testInterpolated[0][index,:,:,0] = GD1

for index in range(ag_testWithMask[1].shape[0]):
	example = ag_testWithMask[1][index,:,:,0]
	#get only the valid values
	x1 = xx[~example.mask]
	y1 = yy[~example.mask]
	newarr = example[~example.mask]
	GD1 = scipy.interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='cubic')
	ag_testInterpolated[1][index,:,:,0] = GD1

a_train = numpy.arctan2(ag_train[1], ag_train[0])
a_trainInterpolated = numpy.arctan2(ag_trainInterpolated[1], ag_trainInterpolated[0])

a_test = numpy.arctan2(ag_test[1], ag_test[0])
a_testInterpolated = numpy.arctan2(ag_testInterpolated[1], ag_testInterpolated[0])




magn = tf.placeholder(tf.float32, [60000,28,28,1])
angs = tf.placeholder(tf.float32, [60000,28,28,1])
magn2 = tf.placeholder(tf.float32, [10000,28,28,1])
angs2 = tf.placeholder(tf.float32, [10000,28,28,1])
normMagn = cc.layers.normalizeVectorField(magn, 3, 3)
vField = cc.transformations.changeToCartesian(normMagn, angs)
normMagn2 = cc.layers.normalizeVectorField(magn2, 3, 3)
vField2 = cc.transformations.changeToCartesian(normMagn2, angs2)
# magn = tf.placeholder(tf.float32, [60000,28,28,1])
# angs = tf.placeholder(tf.float32, [60000,28,28,1])
# vField = cc.transformations.changeToCartesian(magn, angs)

v_trainInterpolated, v_testInterpolated = sess.run([vField, vField2], feed_dict={magn: x_train, angs: a_trainInterpolated, magn2: x_test, angs2: a_testInterpolated})
# v_testInterpolated = sess.run(vField, feed_dict={magn: x_test, angs: a_testInterpolated})
v_train = sess.run(vField, feed_dict={magn: x_train, angs: a_train})

pickle.dump(v_trainInterpolated, open("vMnistTrain3x3AP.pkl","wb"))
pickle.dump(v_testInterpolated, open("vMnistTest3x3AP.pkl","wb"))

pickle.dump(cg_train, open("gCifar10Train.pkl","wb"))
pickle.dump(cg_test, open("gCifar10Test.pkl","wb"))


plt.imshow(a_train[0,:,:,0])
plt.show()
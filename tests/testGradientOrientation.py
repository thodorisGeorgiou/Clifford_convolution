import os
import sys
import numpy
sys.path.append("/tank/georgioutk/cliffordConvolutionTrainable/")
import tensorflow as tf
import cliffordConvolution as cc
import time
from matplotlib import pyplot as plt

def plot_field(field):
	fig, axes = plt.subplots(nrows=field.shape[0], ncols=field.shape[1])
	for i in range(field.shape[0]):
		for j in range(field.shape[1]):
			axes[i,j].quiver(0, 0, field[i,j,0], field[i,j,1], angles='xy', scale_units='xy', scale=1)
			axes[i,j].set_xlim(-1.5, 1.5)
			axes[i,j].set_ylim(-1.5, 1.5)
			axes[i,j].set_xticks([])
			axes[i,j].set_yticks([])
	plt.subplots_adjust(wspace=0, hspace=0)
	# plt.show()

wx = tf.constant(numpy.expand_dims(numpy.expand_dims(numpy.array([[1],[1],[0]]).astype(numpy.float32), axis=-1), axis=-1))
wy = tf.constant(numpy.expand_dims(numpy.expand_dims(numpy.array([[1,1,0]]).astype(numpy.float32), axis=-1), axis=-1))
inpt = tf.placeholder(tf.float32, [1,28,28,1])
angle = tf.placeholder(tf.float32, [])

# avInpt = tf.layers.average_pooling2d(inpt, [5,5], [1,1], padding='SAME')
avInpt = inpt
opGrad = tf.image.image_gradients(avInpt)
tpGrad = []
tpGrad.append(tf.nn.conv2d(opGrad[0],wx,[1,1,1,1],"SAME")/2)
tpGrad.append(tf.nn.conv2d(opGrad[1],wy,[1,1,1,1],"SAME")/2)
# gradStrength = tf.sqrt(tpGrad[0]**2 + tpGrad[1]**2)
# pooledGrads, indexes = tf.nn.max_pool_with_argmax(gradStrength, [1,3,3,1], [1,1,1,1], padding='SAME')
# intIndeces = tf.cast(indexes, tf.int32)
# pooledTpGrads = [cc.ops.poolByIndex(tpGrad[0], intIndeces), cc.ops.poolByIndex(tpGrad[1], intIndeces)]
# avgrads = tf.concat(pooledTpGrads, axis=-1)
avgrads = tf.concat(tpGrad, axis=-1)
avgrads = tf.layers.average_pooling2d(avgrads*9, [5,5], [1,1], padding='SAME')

# grads = tf.concat(grad, axis=-1)
gradsRot = cc.transformations.rotateVectorField(avgrads, angle, irelevantAxisFirst=True)
splitGradsRot = tf.split(gradsRot, 2, -1)
gradStrength = tf.sqrt(splitGradsRot[0]**2 + splitGradsRot[1]**2)
noSmall = [tf.where(tf.abs(gradStrength)<10e-2, tf.zeros_like(splitGradsRot[0]), splitGradsRot[0]), tf.where(tf.abs(gradStrength)<10e-2, tf.zeros_like(splitGradsRot[1]), splitGradsRot[1])]
gamiesai = tf.concat(noSmall, axis=-1)
# gradsRot = tf.contrib.image.rotate(grads, angle, interpolation="BILINEAR")
ang = tf.atan2(noSmall[1],noSmall[0])
reMapped = tf.where(ang<-numpy.pi, ang+2*numpy.pi, ang)
reMapped = tf.where(reMapped>=numpy.pi, reMapped-2*numpy.pi, reMapped)
quantized = tf.round(reMapped/(numpy.pi/16))

rotInpt = tf.contrib.image.rotate(inpt, angle, interpolation="BILINEAR")
# avRotInpt = tf.layers.average_pooling2d(rotInpt, [5,5], [1,1], padding='SAME')
avRotInpt = rotInpt
rotGrad = tf.image.image_gradients(avRotInpt)
rottpGrad = []
rottpGrad.append(tf.nn.conv2d(rotGrad[0],wx,[1,1,1,1],"SAME")/2)
rottpGrad.append(tf.nn.conv2d(rotGrad[1],wy,[1,1,1,1],"SAME")/2)
# rotgradStrength = tf.sqrt(rottpGrad[0]**2 + rottpGrad[1]**2)
# pooledRotGrads, rIndexes = tf.nn.max_pool_with_argmax(rotgradStrength, [1,3,3,1], [1,1,1,1], padding='SAME')
# intRIndeces = tf.cast(rIndexes, tf.int32)
# pooledRotTpGrads = [cc.ops.poolByIndex(rottpGrad[0], intRIndeces), cc.ops.poolByIndex(rottpGrad[1], intRIndeces)]
# rotGrads = tf.concat(pooledRotTpGrads, axis=-1)
rotGrads = tf.concat(rottpGrad, axis=-1)
rotGrads = tf.layers.average_pooling2d(rotGrads*9, [5,5], [1,1], padding='SAME')
# noSmall2 = tf.where(tf.abs(rotGrads)<10e-2, tf.zeros_like(rotGrads), rotGrads)
splitGradsRot2 = tf.split(rotGrads, 2, -1)
rotgradStrength = tf.sqrt(splitGradsRot2[0]**2 + splitGradsRot2[1]**2)
noSmall2 = [tf.where(tf.abs(rotgradStrength)<10e-2, tf.zeros_like(splitGradsRot2[0]), splitGradsRot2[0]), tf.where(tf.abs(rotgradStrength)<10e-2, tf.zeros_like(splitGradsRot2[1]), splitGradsRot2[1])]
gamiesai2 = tf.concat(noSmall2, axis=-1)
# rotGrads = tf.concat(rotGrad, axis=-1)
# avRotGrad = tf.split(avRotGrads, 2, -1)
rotAng = tf.atan2(noSmall2[1], noSmall2[0])
rotReMapped = tf.where(rotAng<-numpy.pi, rotAng+2*numpy.pi, rotAng)
rotReMapped = tf.where(rotReMapped>=numpy.pi, rotReMapped-2*numpy.pi, rotReMapped)
rotquantized = tf.round(rotReMapped/(numpy.pi/16))

# batch_size = 100
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = numpy.expand_dims((x_train / 255.0).astype(numpy.float32), -1), numpy.expand_dims((x_test / 255.0).astype(numpy.float32), -1)
y_train, y_test = y_train.astype(numpy.int32), y_test.astype(numpy.int32)

sess = tf.Session()
g, rg, a, ra = sess.run([gamiesai, gamiesai2, quantized, rotquantized], feed_dict={inpt:x_train[0:1], angle: numpy.pi/3+numpy.pi/22})
# g, rg, a, ra = sess.run([gradsRot, rotGrads, reMapped, rotReMapped], feed_dict={inpt:x_train[0:1], angle: numpy.pi/3+numpy.pi/22})

numpy.sum(a!=ra)
d = numpy.where(a!=ra)
numpy.max(a[d]-ra[d])
numpy.min(a[d]-ra[d])

d = a-ra
d[d>numpy.pi] = d[d>numpy.pi] - 2*numpy.pi
d[d<-numpy.pi] = d[d<-numpy.pi] + 2*numpy.pi
numpy.max(d)
numpy.min(d)
numpy.sum(d>10e-3)
numpy.sum(d>10e-2)
numpy.sum(d>10e-1)
numpy.sum(d>numpy.pi/32)


gg = numpy.zeros([1,28,28,2])
gg[:,:,:,0:1] = a != ra
plot_field(g[0])
plot_field(rg[0])
plot_field(gg[0])
plt.show()

import sys
import numpy
from matplotlib import pyplot as plt
sys.path.append("/scratch/georgioutk/cliffordConvolution/")
import tensorflow as tf
import cliffordConvolution as cc

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


c_i = 50
c_o = 1
num_bins = 32
num_angles=4
offset = num_bins//(2*num_angles)
quantError = 2*numpy.pi/num_bins

v = tf.random.uniform([4, 4, 2*c_i, c_o], minval=0, maxval=1)
vt = tf.transpose(v, [3,0,1,2])
vn = cc.layers.normalizeVectorField(vt, 4, 4)
inpt = tf.transpose(vn, [1,2,3,0])

noise = tf.random.normal([4, 4, 2*c_i, c_o], 0, 0.1)
w = v+noise
wt = tf.transpose(w, [3,0,1,2])
wn = cc.layers.normalizeVectorField(wt, 4, 4)
w = tf.transpose(wn, [1,2,3,0])

# w = tf.contrib.layers.group_norm(v, c_i, 2, (0,1),  center=False, trainable=False)
# inpt = tf.contrib.layers.group_norm(v, c_i, 2, (0,1),  center=False, trainable=False)
# inpt = tf.contrib.layers.group_norm(v+noise, c_i, 2, (0,1),  center=False, scale=False, trainable=False)
# v = tf.ones([3, 3, 2*c_i, c_o])
phi = tf.random.uniform([], minval=-quantError, maxval=2*numpy.pi-quantError)
rotInpt = cc.transformations.rotateVectorField(inpt, phi)
rotInpt = tf.squeeze(rotInpt, [-1])
rotInpt = tf.expand_dims(rotInpt, 0)

rotInpt2 = cc.transformations.rotateVectorField(inpt, phi)
# rotInpt2 = cc.transformations.rotateVectorFieldMinus(inpt, phi)
rotInpt2 = tf.squeeze(rotInpt2, [-1])
rotInpt2 = tf.expand_dims(rotInpt2, 0)

rotatedWeights = []
thetas = []
angles = [tf.constant([(a-offset)*2*numpy.pi/num_bins]) for a in range(num_bins+1)]
for angle in angles:
	rotatedWeights.append(cc.transformations.rotateVectorField(w, angle))

rotatedWeights2 = []
thetas_2 = []
angles2 = [tf.constant([(a-offset)*2*numpy.pi/num_bins]) for a in range(num_bins+1)]
for angle in angles2:
	rotatedWeights2.append(cc.transformations.rotateVectorField(w, angle))
	# rotatedWeights2.append(cc.transformations.rotateVectorFieldMinus(w, angle))


rotatedWeights = tf.stack(rotatedWeights, axis=0)
weightShape = rotatedWeights.get_shape()
for angle in range(0, num_bins, num_bins//num_angles):
	weightSet = tf.gather(rotatedWeights, angle + offset)
	conv2 = cc.layers.conv_2tf(rotInpt, weightSet, c_i, c_o, 1, 1, "VALID")
	conv0 = tf.nn.conv2d(rotInpt, weightSet, [1,1,1,1], "VALID")
	thetas.append(tf.atan2(conv2, conv0))

rotatedWeights2 = tf.stack(rotatedWeights2, axis=0)
weightShape = rotatedWeights2.get_shape()
for angle in range(0, num_bins, num_bins//num_angles):
	weightSet2 = tf.gather(rotatedWeights2, angle + offset)
	conv2 = cc.layers.conv_2tf(rotInpt2, weightSet2, c_i, c_o, 1, 1, "VALID")
	conv0 = tf.nn.conv2d(rotInpt2, weightSet2, [1,1,1,1], "VALID")
	thetas_2.append(-tf.atan2(conv2, conv0))

angles = tf.concat(angles, axis=0)
thetas = tf.stack(thetas, axis=-1)
winner = tf.argmin(tf.abs(thetas), axis=-1, output_type=tf.int32)
thetas2 = cc.ops.reduceIndex(thetas, winner)
# thetas2, convMask = cc.ops.offsetCorrect(thetas2, [offset*2*numpy.pi/num_bins])
# quantized = tf.cast(tf.round(thetas2*num_bins/(2*numpy.pi)), tf.int32) + tf.cast(winner * (num_bins//num_angles), tf.int32) + offset
pred = thetas2 + (tf.cast(winner, tf.float32)*(num_bins//num_angles)) * 2 * numpy.pi / num_bins

angles2 = tf.concat(angles2, axis=0)
thetas_2 = tf.stack(thetas_2, axis=-1)
winner2 = tf.argmin(tf.abs(thetas_2), axis=-1, output_type=tf.int32)
thetas22 = cc.ops.reduceIndex(thetas_2, winner2)
# thetas22, convMask2 = cc.ops.offsetCorrect(thetas22, [offset*2*numpy.pi/num_bins])
# quantized = tf.cast(tf.round(thetas2*num_bins/(2*numpy.pi)), tf.int32) + tf.cast(winner * (num_bins//num_angles), tf.int32) + offset
pred2 = thetas22 + (tf.cast(winner2, tf.float32)*(num_bins//num_angles)) * 2 * numpy.pi / num_bins

diff = tf.squeeze(pred - phi)
diff2 = diff + 2*numpy.pi

diff_2 = tf.squeeze(pred2 - phi)
diff_22 = diff_2 + 2*numpy.pi
# fin = tf.minimum(tf.abs(diff), tf.abs(diff2))
res = []
res2 = []
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# p, k, t, w, t2,  a= sess.run([phi, inpt, thetas, winner, thetas2, pred])

for i in range(10000):
	if i % 1000==0:
		print(i/10000, end="\r", flush=True)
	r = numpy.array(sess.run([diff, diff2]))
	res.append(r[numpy.argmin(numpy.abs(r))])
	r2 = numpy.array(sess.run([diff_2, diff_22]))
	res2.append(r2[numpy.argmin(numpy.abs(r2))])


r = numpy.array(res)
print(r.mean())
print(r.std())
b = plt.hist(res2, bins=500, color="red")
a = plt.hist(res, bins=500, color="blue")
plt.show()

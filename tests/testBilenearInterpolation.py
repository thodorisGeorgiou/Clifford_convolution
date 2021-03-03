import numpy
import tensorflow as tf
import cliffordConvolution as cc

def rotate_vector_field(field, angle, order):
	angle = angle * numpy.pi
	cos_theta = numpy.cos(angle)
	sin_theta = numpy.sin(angle)
	r = numpy.array([[cos_theta, sin_theta],[-1*sin_theta, cos_theta]])
	rotated = numpy.matmul(field, r)
	return scipy.ndimage.interpolation.rotate(rotated, numpy.degrees(-angle), reshape=False, order=order)


def manual_rotate_vector_field(field, angle):
	angle = angle * numpy.pi
	cos_theta = numpy.cos(angle)
	sin_theta = numpy.sin(angle)
	r = numpy.array([[cos_theta, sin_theta],[-1*sin_theta, cos_theta]])
	rotated = numpy.matmul(field, r)
	center = numpy.array([1,1], dtype=numpy.float32)
	c = numpy.zeros(shape=field.shape, dtype=numpy.float32)
	cos_theta = numpy.cos(-angle)
	sin_theta = numpy.sin(-angle)
	r = numpy.array([[cos_theta, sin_theta],[-1*sin_theta, cos_theta]])
	for i in range(field.shape[0]):
		for j in range(field.shape[1]):
			_coor = numpy.array([i,j], dtype=numpy.float32)
			nCoor = numpy.matmul(_coor - center, r) + center
			coor = _coor.astype(numpy.int32)
			c[coor[0], coor[1],0] = bilinear_interpolation(rotated[:,:,0], nCoor)
			c[coor[0], coor[1],1] = bilinear_interpolation(rotated[:,:,1], nCoor)
	return rotated, c

def bilinear_interpolation(inpt, nCoor):
	# if numpy.any(numpy.logical_or(nCoor+1 > inpt.shape[:2], nCoor < 0)):
	# 	return 0
	a = numpy.zeros(shape=numpy.array(inpt.shape)+2, dtype=numpy.float32)
	a[1:inpt.shape[0]+1, 1:inpt.shape[1]+1] = inpt
	x1 = numpy.floor(nCoor[0]).astype(numpy.int32)
	y1 = numpy.floor(nCoor[1]).astype(numpy.int32)
	x = numpy.array([x1+1 - nCoor[0], nCoor[0] - x1])
	y = numpy.array([y1+1 - nCoor[1], nCoor[1] - y1])
	k = numpy.matmul(x,a[x1+1:x1+3, y1+1:y1+3])
	return numpy.matmul(k, numpy.transpose(y))

def conv_0(a,b):
	return numpy.sum(a*b)

def conv_2(a,b):
	part_sums = numpy.sum((a*(b[:,:,::-1])), axis=(0,1))
	return part_sums[0] - part_sums[1]

def plot_field(field):
	fig, axes = plt.subplots(nrows=field.shape[0], ncols=field.shape[1])
	for i in range(field.shape[0]):
		for j in range(field.shape[1]):
			axes[i,j].quiver(0, 0, field[i,j,0], field[i,j,1], angles='xy', scale_units='xy', scale=1)
			axes[i,j].set_xlim(-1.5, 1.5)
			axes[i,j].set_ylim(-1.5, 1.5)
	plt.show()


num_bins = 32
num_angles = 4
offset = num_bins//(2*num_angles)
numpyAngles = [(a-offset)*2*numpy.pi/num_bins for a in range(num_bins+1)]
# angles = [tf.Variable(a) for a in numpyAngles]
with tf.device('gpu:0'):
	an = tf.placeholder(tf.float32, shape=[])
	weights = tf.placeholder(tf.float32, shape=[3,3,200,1])
	inpt = tf.placeholder(tf.float32, shape=[1,3,3,200])
	# inpt2 = tf.transpose(inpt, [3,0,1,2])
	rotWeights = cc.tranformations.rotateVectorField(weights, an)
	conv2 = cc.layers.conv_2tf(inpt, rotWeights, 100, 1, 1, 1, "VALID")
	conv0 = tf.nn.conv2d(inpt, rotWeights, [1,1,1,1], "VALID")
	theta = tf.atan2(conv2, conv0)

# rotWeights = tf.contrib.image.rotate(weights, an, interpolation='BILINEAR')
numpyWeights = numpy.random.rand(3,3,2).astype(numpy.float32)
numpyWeights = numpy.zeros(shape=[3,3,2,1]).astype(numpy.float32)
numpyWeights[0] = 1
# numpyWeights[1,0] = 1


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# for a in numpyAngles:
# 	myRotWeights =  manual_rotate_vector_field(numpyWeights, a)
# 	tfRot = sess.run(rotWeights, feed_dict={weights: numpyWeights, an: a})
# 	print(numpy.sum(tfRot != myRotWeights))
wdot, wfi = manual_rotate_vector_field(numpyWeights[:,:,:,0], 0.5)

myRot = manual_rotate_vector_field(numpyWeights[:,:,:,0], 1.0/num_angles)
tfRot = sess.run(rotWeights, feed_dict={weights: numpyWeights, an: numpy.pi/num_angles})
spRot = rotate_vector_field(numpyWeights[:,:,:,0], 1.0/num_angles, 2)
myRotWeights[0,:,:,0]
tfRot[0,:,:,0]

print(numpy.sum(tfRot != myRotWeights))

import pickle
import numpy
from matplotlib import pyplot as plt
weights = pickle.load(open("weigthsAndRotations.pkl", "rb"))
for field in weights:
	plot_field(field)

# rotatedWeights = []
# for angle in angles:
# 	rotatedWeights.append(cc.tranformations.rotateVectorField(weights, angle))

# rotatedWeights = tf.stack(rotatedWeights, axis=0)
plot_field(numpyWeights[:,:,:,0])
plot_field(tfRot[:,:,:,0])
plot_field(myRot)
plot_field(spRot)


angle = numpy.zeros((3,1000), dtype=numpy.float32)
for rr in range(1000):
	print(rr, end="\r", flush=True)
	c2 = [[],[],[]]
	c0 = [[],[],[]]
	ee = numpy.zeros([1,3,3,200], dtype=numpy.float32)
	aa = numpy.zeros([3,3,200,1], dtype=numpy.float32)
	for k in range(100):
		a = numpy.ones(shape=[3,3,2,1])
		r = numpy.random.rand(3,3) * 2 * numpy.pi
		r2 = numpy.random.rand(3,3)
		a[:,:,0,0] = numpy.cos(r) * r2
		a[:,:,1,0] = numpy.sin(r) * r2
		e = a / numpy.linalg.norm(a)
		b = a + (numpy.random.rand(3,3,2,1) - 0.5)*0.01
		ee[0,:,:,2*k:2*k+2] = e[:,:,:,0]
		aa[:,:,2*k:2*k+2,0] = b[:,:,:,0]
		t = (1.0+numpy.random.rand()*0.1)/num_angles
		c = rotate_vector_field(b[:,:,:,0], t, 2)
		d = manual_rotate_vector_field(b[:,:,:,0], t)
		# g = sess.run(rotWeights, feed_dict={weights: b, an: numpy.pi/num_angles})
		# a = a / conv_0(a,a)
		# b = b / conv_0(b,b)0
		# c = c / conv_0(c,c)
		c2[0].append(conv_2(e[:,:,:,0],c))
		c0[0].append(conv_0(e[:,:,:,0],c))
		c2[1].append(conv_2(e[:,:,:,0],d))
		c0[1].append(conv_0(e[:,:,:,0],d))
		# conv22 = sess.run(conv2, feed_dict={weights: b, an: numpy.pi*t, inpt: e})
		# conv00 = sess.run(conv0, feed_dict={weights: b, an: numpy.pi*t, inpt: e})
		# c2[2].append(conv22)
		# c0[2].append(conv00)
		# c2[2].append(conv_2(e[:,:,:,0],g[:,:,:,0]))
		# c0[2].append(conv_0(e[:,:,:,0],g[:,:,:,0]))
	# tan1 = sum(c2[0])/sum(c0[0])
	# tan2 = sum(c2[1])/sum(c0[1])
	# tan3 = sum(c2[2])/sum(c0[2])
	angle[0,rr] = numpy.arctan2(sum(c2[0]),sum(c0[0]))
	angle[1,rr] = numpy.arctan2(sum(c2[1]),sum(c0[1]))
	angle[2,rr] = sess.run(theta, feed_dict={weights: aa, an: numpy.pi/num_angles, inpt: ee})
	# angle[2,rr] = numpy.arctan2(sum(c2[1]),sum(c0[1]))

# numpy.pi/3
numpy.mean(angle[0])
numpy.std(angle[0])
numpy.mean(angle[1])
numpy.std(angle[1])
numpy.mean(angle[2])
numpy.std(angle[2])

hist0 = numpy.zeros(num_bins)
hist1 = numpy.zeros(num_bins)
hist2 = numpy.zeros(num_bins)
quantized0 = numpy.round(angle[0]*num_bins/(2*numpy.pi))
quantized1 = numpy.round(angle[1]*num_bins/(2*numpy.pi))
quantized2 = numpy.round(angle[2]*num_bins/(2*numpy.pi))
q0 = quantized0.astype(numpy.int32)
q1 = quantized1.astype(numpy.int32)
q2 = quantized2.astype(numpy.int32)
numpy.add.at(hist0, q0, 1)
numpy.add.at(hist1, q1, 1)
numpy.add.at(hist2, q2, 1)

near, = plt.plot(hist0, label="sp")
ref, = plt.plot(hist1, label="me")
zero, = plt.plot(hist2, label="tf")
plt.legend(handles=[near, ref, zero])
plt.show()

a = numpy.random.rand(3,3,2).astype(numpy.float32)
b = numpy.random.rand(3,3,2).astype(numpy.float32)
c = manual_rotate_vector_field(a, 1.0/5)
d = manual_rotate_vector_field(b, -1.0/5)

s1 = numpy.sum(a[:,:,0]*d[:,:,1]-a[:,:,1]*d[:,:,0])
s2 = numpy.sum(c[:,:,0]*b[:,:,1]-c[:,:,1]*b[:,:,0])
print(str(s1)+" - "+str(s2))
print(s2/(s1-s2))
# e = rotate_vector_field(a, 1.0/5, 2)
# f = rotate_vector_field(b, -1.0/5, 2)
# print(str(s1-s2)+" - "+str(s1-s3))

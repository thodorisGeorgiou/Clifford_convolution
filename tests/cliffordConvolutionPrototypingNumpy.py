import numpy
import scipy.ndimage
import time
from PIL import Image
from matplotlib import pyplot as plt

def rotate_vector_field1(field, angle):
	angle = angle * numpy.pi
	cos_theta = numpy.cos(angle)
	sin_theta = numpy.sin(angle)
	r = numpy.array([[cos_theta, sin_theta],[-1*sin_theta, cos_theta]])
	rotated = numpy.matmul(field, r)
	return scipy.ndimage.interpolation.rotate(rotated, numpy.degrees(angle), reshape=False, mode='nearest')

def rotate_vector_field2(field, angle):
	angle = angle * numpy.pi
	cos_theta = numpy.cos(angle)
	sin_theta = numpy.sin(angle)
	r = numpy.array([[cos_theta, sin_theta],[-1*sin_theta, cos_theta]])
	rotated = numpy.matmul(field, r)
	return scipy.ndimage.interpolation.rotate(rotated, numpy.degrees(angle), reshape=False, mode="reflect")

def rotate_vector_field(field, angle, order):
	angle = angle * numpy.pi
	cos_theta = numpy.cos(angle)
	sin_theta = numpy.sin(angle)
	r = numpy.array([[cos_theta, sin_theta],[-1*sin_theta, cos_theta]])
	rotated = numpy.matmul(field, r)
	return scipy.ndimage.interpolation.rotate(rotated, numpy.degrees(angle), reshape=False, order=order)

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
	return c


def conv_0(a,b):
	return numpy.sum(a*b)

def conv_2(a,b):
	part_sums = numpy.sum((a*(b[:,:,::-1])), axis=(0,1))
	return part_sums[0] - part_sums[1]

def bilinear_interpolation(inpt, nCoor):
	if numpy.any(numpy.logical_or(nCoor+1 > inpt.shape[:2], nCoor < 0)):
		return 0
	a = numpy.zeros(shape=numpy.array(inpt.shape)+1, dtype=numpy.float32)
	a[0:inpt.shape[0], 0:inpt.shape[1]] = inpt
	x1 = numpy.floor(nCoor[0]).astype(numpy.int32)
	y1 = numpy.floor(nCoor[1]).astype(numpy.int32)
	x = numpy.array([x1+1 - nCoor[0], nCoor[0] - x1])
	y = numpy.array([y1+1 - nCoor[1], nCoor[1] - y1])
	k = numpy.matmul(x,a[x1:x1+2, y1:y1+2])
	return numpy.matmul(k, numpy.transpose(y))

def plot_field(field):
	fig, axes = plt.subplots(nrows=field.shape[0], ncols=field.shape[1])
	for i in range(field.shape[0]):
		for j in range(field.shape[1]):
			axes[i,j].quiver(0, 0, field[i,j,0], field[i,j,1], angles='xy', scale_units='xy', scale=1)
			axes[i,j].set_xlim(-1.5, 1.5)
			axes[i,j].set_ylim(-1.5, 1.5)
	plt.show()

angle = numpy.zeros((3,10000), dtype=numpy.float32)
for rr in range(10000):
	c2 = [[],[],[]]
	c0 = [[],[],[]]
	for k in range(100):
		a = numpy.ones(shape=[3,3,2])
		r = numpy.random.rand(3,3) * 2 * numpy.pi
		r2 = numpy.random.rand(3,3)
		a[:,:,0] = numpy.cos(r) * r2
		a[:,:,1] = numpy.sin(r) * r2
		e = a / numpy.linalg.norm(a)
		b = a + (numpy.random.rand(3,3,2) - 0.5)*0.1
		c = rotate_vector_field(a, 1.0/4, 2)
		d = rotate_vector_field(b, 1.0/4, 2)
		g = rotate_vector_field(f, 1.0/4, 2)
		# a = a / conv_0(a,a)
		# b = b / conv_0(b,b)0
		# c = c / conv_0(c,c)
		c2[0].append(conv_2(e,c))
		c0[0].append(conv_0(e,c))
		c2[1].append(conv_2(e,d))
		c0[1].append(conv_0(e,d))
		c2[2].append(conv_2(e,g))
		c0[2].append(conv_0(e,g))
	tan1 = sum(c2[0])/sum(c0[0])
	tan2 = sum(c2[1])/sum(c0[1])
	tan3 = sum(c2[2])/sum(c0[2])
	angle[0,rr] = numpy.arctan2(tan1)
	angle[1,rr] = numpy.arctan2(tan2)
	angle[2,rr] = numpy.arctan2(tan3)

# numpy.pi/3
numpy.mean(angle[0])
numpy.std(angle[0])
numpy.mean(angle[1])
numpy.std(angle[1])
numpy.mean(angle[2])
numpy.std(angle[2])

hist0 = numpy.zeros(64)
hist1 = numpy.zeros(64)
hist2 = numpy.zeros(64)
quantized0 = numpy.round(angle[0]*64/(2*numpy.pi))
quantized1 = numpy.round(angle[1]*64/(2*numpy.pi))
quantized2 = numpy.round(angle[2]*64/(2*numpy.pi))
q0 = quantized0.astype(numpy.int32)
q1 = quantized1.astype(numpy.int32)
q2 = quantized2.astype(numpy.int32)
numpy.add.at(hist0, q0, 1)
numpy.add.at(hist1, q1, 1)
numpy.add.at(hist2, q2, 1)

near, = plt.plot(hist0, label="1")
ref, = plt.plot(hist1, label="2")
zero, = plt.plot(hist2, label="3")
plt.legend(handles=[near, ref, zero])
plt.show()

s = 3*3
a = numpy.ones(shape=[3,3,2])
r = numpy.random.rand(3,3) * 2 * numpy.pi
a[:,:,0] = numpy.cos(r)
a[:,:,1] = numpy.sin(r)
c = numpy.ones(shape=[3,3,2])
r = numpy.random.rand(3,3) * 2 * numpy.pi
c[:,:,0] = numpy.cos(r)
c[:,:,1] = numpy.sin(r)
b = []
b2 = []
step = numpy.pi/16
for i in xrange(4):
	b.append(rotate_vector_field(a, i*2.0/4))

for i in xrange(32):
	b2.append(rotate_vector_field(a, i*2.0/32))

k = numpy.zeros(4, dtype=numpy.float32)
b = numpy.array(b)
b2 = numpy.array(b2)
t1 = time.time()
for rr in xrange(10000):
	k[0] = conv_0(b[0],c)/conv_2(b[0],c)
	k[1] = conv_0(b[1],c)/conv_2(b[1],c)
	k[2] = conv_0(b[2],c)/conv_2(b[2],c)
	k[3] = conv_0(b[3],c)/conv_2(b[3],c)
	l = numpy.arctan(k)
	fi = numpy.argmin(l)
	f = fi*8 + numpy.round(l[fi]/step).astype(numpy.int32)
	conv = conv_2(b2[f], c)/conv_0(b2[f], c)
	angle = f*step
	out = numpy.array([conv*numpy.sin(angle), conv*numpy.cos(angle)])

t2 = time.time()
t1 = time.time()
for rr in xrange(10000):
	out = []
	for j in b2:
		out.append(conv_0(j,c))

t2 = time.time()
max(c0)
min(c0)

conv_2(b,c)
conv_0(b,c)

angle = 1.0/3 * numpy.pi
cos_theta = numpy.cos(-angle)
sin_theta = numpy.sin(-angle)
r = numpy.array([[cos_theta, sin_theta],[-1*sin_theta, cos_theta]])
center = numpy.array([1,1], dtype=numpy.float32)
a = numpy.ones(shape=[3,3], dtype=numpy.float32)
c = numpy.zeros(shape=[3,3], dtype=numpy.float32)
d = numpy.zeros(shape=[3,3], dtype=numpy.float32)
a[0,1] = 2
for i in range(3):
	for j in range(3):
		_coor = numpy.array([i,j], dtype=numpy.float32)
		nCoor = numpy.matmul(_coor - center, r) + center
		coor = _coor.astype(numpy.int32)
		print("Point "+str(coor)+" - "+str(nCoor))
		c[coor[0], coor[1]] = bilinear_interpolation(a, nCoor)
		# print c

c = c/d
angle = 1.0/3 * numpy.pi
b = scipy.ndimage.interpolation.rotate(a, numpy.degrees(angle), reshape=False, order=1)
a1 = Image.fromarray(a)
c = a1.rotate(60, resample=Image.NEAREST)
d = a1.rotate(60, resample=Image.BILINEAR)
e = a1.rotate(60, resample=Image.BICUBIC)

numpy.reshape(numpy.array(c.getdata()), [3,3])

angle = numpy.zeros((2,10000), dtype=numpy.float32)
for rr in range(10000):
	print(rr)
	b = numpy.zeros([3,3,128], dtype=numpy.float32)
	c = numpy.zeros([3,3,128], dtype=numpy.float32)
	bb = numpy.zeros([3,3,128], dtype=numpy.float32)
	cc = numpy.zeros([3,3,128], dtype=numpy.float32)
	c0=[]
	c2=[]
	for k in range(128):
		a = numpy.ones(shape=[3,3,2], dtype=numpy.float32)
		r = numpy.random.rand(3,3).astype(numpy.float32) * 2 * numpy.pi
		r2 = numpy.random.rand(3,3).astype(numpy.float32)
		a[:,:,0] = numpy.cos(r) * r2
		a[:,:,1] = numpy.sin(r) * r2
		e = (a / numpy.linalg.norm(a)).astype(numpy.float32)
		f = a  + (numpy.random.rand(3,3,2) - 0.5)*0.01
		g = rotate_vector_field(f, 1.0/4, 2)
		b[:,:,k] = e[:,:,0]
		c[:,:,k] = e[:,:,1]
		bb[:,:,k] = g[:,:,0]
		cc[:,:,k] = g[:,:,1]
		c2.append(conv_2(e.astype(numpy.float32),g.astype(numpy.float32)))
		c0.append(conv_0(e.astype(numpy.float32),g.astype(numpy.float32)))
	tan1 = numpy.sum(c2)/numpy.sum(c0)
	mm = numpy.sum(b*bb)
	zz = numpy.sum(c*cc)
	mz = numpy.sum(b*cc)
	zm = numpy.sum(bb*c)
	cc0 = mm + zz
	cc2 = mz - zm
	tan2 = cc0/cc2
	angle[0,rr] = numpy.arctan(tan1)
	angle[1,rr] = numpy.arctan(tan2)

numpy.mean(angle[0])
numpy.std(angle[0])
numpy.mean(angle[1])
numpy.std(angle[1])
numpy.mean(angle[2])
numpy.std(angle[2])

hist0 = numpy.zeros(64)
hist1 = numpy.zeros(64)
quantized0 = numpy.round(angle[0]*64/(2*numpy.pi))
quantized1 = numpy.round(angle[1]*64/(2*numpy.pi))
q0 = quantized0.astype(numpy.int32)
q1 = quantized1.astype(numpy.int32)
numpy.add.at(hist0, q0, 1)
numpy.add.at(hist1, q1, 1)

fr, = plt.plot(hist0, label="1")
mz, = plt.plot(hist1, label="2")
plt.legend(handles=[fr, mz])
plt.show()

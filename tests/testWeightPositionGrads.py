import numpy
import time
import tensorflow as tf
test = tf.load_op_library("weight_to_angle_gradients.so").weight_to_angle_gradients
# import scipy.ndimage

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


batch = 64
rows=56
cols=56
in_depth=64
out_depth=64
filter_rows=3
filter_cols=3
num_angles=32
# a = numpy.ones([10,5,5,10], dtype=numpy.float32)				#input
# g = numpy.ones([10,5,5,10], dtype=numpy.float32)				#gradients
a = numpy.random.rand(batch,rows,cols,in_depth).astype(numpy.float32)		#input
g = numpy.random.rand(batch,rows,cols,out_depth).astype(numpy.float32)		#gradients
b = (numpy.random.rand(batch,rows,cols,out_depth)*(num_angles+1)).astype(numpy.int32)	#indexes
# b = numpy.zeros([10,5,5,10], dtype=numpy.int32)				#indexes
w = numpy.zeros([num_angles+1,filter_rows,filter_cols,in_depth,out_depth], dtype=numpy.float32)			#weights
# woa = numpy.zeros([num_angles+1,filter_rows,filter_cols,in_depth,out_depth], dtype=numpy.float32)		#woa
# normFactor = 3.14159265358979323846*4/num_angles


# init_weights = numpy.ones([3,3,10,10], dtype=numpy.float32)
init_weights = numpy.random.rand(filter_rows,filter_cols,in_depth,out_depth).astype(numpy.float32)
for dout in range(init_weights.shape[3]):
	for din in range(0,init_weights.shape[2],2):
		field = init_weights[:,:,din:din+2,dout]
		for angle in range(num_angles):
			an = angle*numpy.pi*2/num_angles
			w[angle,:,:,din:din+2,dout] = manual_rotate_vector_field(field, an)
			if angle==0: w[-1,:,:,din:din+2,dout] = w[angle,:,:,din:din+2,dout]

#weight over angle gradients
# woa[0] = (w[1]-w[num_angles-1])/normFactor
# woa[-1] = woa[0]
# for i in range(1,num_angles):
# 	woa[i] = (w[i+1] - w[i-1])/normFactor


#pad input
# padrows = filter_rows//2
# padcols = filter_cols//2
# pa = numpy.zeros([batch,a.shape[1]+2*padrows,a.shape[2]+2*padcols,in_depth])
# pa[:,padrows:-padrows,padcols:-padcols,:] = a

# #error over weight gradients (per point)
# eow = numpy.zeros([batch,rows,cols,out_depth,filter_rows,filter_cols,in_depth], dtype=numpy.float32)
# oef1 = numpy.zeros([batch,rows,cols,out_depth], dtype= numpy.float32)
# oef2 = numpy.zeros([batch,rows,cols,out_depth], dtype= numpy.float32)
# for ba in range(g.shape[0]):
# 	for r in range(g.shape[1]):
# 		for c in range(g.shape[2]):
# 			for dout in range(g.shape[3]):
# 				eow[ba,r,c,dout] = g[ba,r,c,dout] * pa[ba, r:r+w.shape[1], c:c+w.shape[2], :]
# 				oef1[ba,r,c,dout] = numpy.sum(woa[b[ba,r,c,dout],:,:,:,dout] * eow[ba,r,c,dout])
# 				for din in range(0,a.shape[3],2):
# 					oef2[ba,r,c,dout] -= numpy.sum(eow[ba,r,c,dout,:,:,din]*w[b[ba,r,c,dout],:,:,din+1,dout])
# 					oef2[ba,r,c,dout] += numpy.sum(eow[ba,r,c,dout,:,:,din+1]*w[b[ba,r,c,dout],:,:,din,dout])

# oef = oef1+oef2


aa = tf.placeholder(tf.float32, [batch,rows,cols,in_depth])
bb = tf.placeholder(tf.int32, [batch,rows,cols,out_depth])
gg = tf.placeholder(tf.float32, [batch,rows,cols,out_depth])
ww = tf.placeholder(tf.float32, [num_angles+1,filter_rows,filter_cols,in_depth,out_depth])

with tf.device('cpu:0'):
	c = test(aa,gg,ww,bb, [1,1,1,1], "SAME")

with tf.device('gpu:0'):
	d = test(aa,gg,ww,bb, [1,1,1,1], "SAME")

sess = tf.Session()

st1 = time.time()
res = sess.run([d], feed_dict={aa: a, bb: b, gg: g, ww: w})
t1 = time.time() - st1


res = sess.run([c, d], feed_dict={aa: a, bb: b, gg: g, ww: w})
#Check differences
# numpy.sum(res[0] != oef)
# numpy.sum(res[1] != oef)
numpy.sum(res[0] != res[1])
numpy.mean(numpy.abs(res[0]))
# numpy.mean(res[0] - oef)
# numpy.mean(res[1] - oef)
# numpy.max(res[0] - oef)
# numpy.max(res[1] - oef)
# numpy.min(res[0] - oef)
# numpy.min(res[1] - oef)
numpy.mean(res[0] - res[1])
numpy.mean(numpy.abs(res[0] - res[1]))
numpy.max(res[0] - res[1])
numpy.min(res[0] - res[1])

kapa = numpy.where(numpy.abs(res[0] - res[1])>0.00001)
numpy.sum(b[kapa]==27) + numpy.sum(b[kapa] == 28)
kapa[0].shape
lamda = numpy.where(b==27)
print(lamda[0].shape[0])
lamda = numpy.where(b==28)
print(lamda[0].shape[0])
for i in range(lamda[0].shape[0]):
    # print(str(lamda[0][i])+" "+str(lamda[1][i])+" "+str(lamda[2][i])+" "+str(lamda[3][i])+" - "+str(b[lamda[0][i],lamda[1][i],lamda[2][i],lamda[3][i]]))
    print(str(kapa[0][i])+" "+str(kapa[1][i])+" "+str(kapa[2][i])+" "+str(kapa[3][i])+" - "+str(b[kapa[0][i],kapa[1][i],kapa[2][i],kapa[3][i]]))

8.075713634490967
6.438404083251953
8.044379234313965


3.2938644886016846
2.1431124210357666
1.963057041168213
1.9582140445709229
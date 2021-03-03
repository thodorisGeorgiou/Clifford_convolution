import numpy
import time
import tensorflow as tf
import cliffordConvolution as cc
# test = tf.load_op_library("conv_by_index_input_grads.so")

batch = 64
rows=56
cols=56
in_depth=64
out_depth=64
filter_rows=3
filter_cols=3
num_angles=32
num_gamiesai=4
offset = num_angles//(2*num_gamiesai)
angles = [tf.Variable([(a-offset)*2*numpy.pi/num_angles]) for a in range(num_angles+1)]

a = numpy.random.rand(batch,rows,cols,in_depth).astype(numpy.float32)		#input
g = numpy.random.rand(batch,rows,cols,out_depth).astype(numpy.float32)		#gradients
b = (numpy.random.rand(batch,rows,cols,out_depth)*(num_angles+1)).astype(numpy.int32)	#indexes
# w = numpy.zeros([num_angles+1,filter_rows,filter_cols,in_depth,out_depth], dtype=numpy.float32)			#weights
init_weights = numpy.random.rand(filter_rows,filter_cols,in_depth,out_depth).astype(numpy.float32)	#weights


aa = tf.placeholder(tf.float32, [batch,rows,cols,in_depth])
bb = tf.placeholder(tf.int32, [batch,rows,cols,out_depth])
gg = tf.placeholder(tf.float32, [batch,rows,cols,out_depth])
w = tf.placeholder(tf.float32, [filter_rows,filter_cols,in_depth,out_depth])
ww = tf.placeholder(tf.float32, [num_angles+1,filter_rows,filter_cols,in_depth,out_depth])

with tf.device('gpu:0'):
	rotatedWeights = []
	for angle in angles:
		rotatedWeights.append(cc.tranformations.rotateVectorField(w, angle))
	rotatedWeights = tf.stack(rotatedWeights, axis=0)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
numpyWeights = sess.run(rotatedWeights, feed_dict={w: init_weights})

with tf.device('cpu:0'):
	cpuRes = cc.ops.convByIndexWeightGrads(aa, rotatedWeights, gg, bb,\
		dilations=[1,1,1,1], strides=[1,1,1,1],\
		padding="SAME")

with tf.device('gpu:0'):
	gpuRes = cc.ops.convByIndexWeightGrads(aa, rotatedWeights, gg, bb,\
		dilations=[1,1,1,1], strides=[1,1,1,1],\
		padding="SAME")

st1 = time.time()
res = sess.run([gpuRes], feed_dict={aa: a, rotatedWeights: numpyWeights, gg: g, bb: b})
t1 = time.time() - st1

t1

res = sess.run([gpuRes, cpuRes], feed_dict={aa: a, rotatedWeights: numpyWeights, gg: g, bb: b})

numpy.sum(res[0] != res[1])
numpy.mean(numpy.abs(res[0] - res[1]))
numpy.max(numpy.abs(res[0] - res[1]))


2.7339231967926025
2.7390339374542236
2.727132558822632

2.5907230377197266
2.5918288230895996
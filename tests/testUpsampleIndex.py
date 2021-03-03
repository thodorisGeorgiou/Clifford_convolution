import numpy
import sys
import tensorflow as tf
sys.path.append("/scratch/georgioutk/cliffordConvolutionMoreTest2/")
import cliffordConvolution as cc

batch=32
rows=28
cols=28
depth=16

# aa = numpy.random.rand(batch, rows, cols, depth).astype(numpy.float32)
aa = numpy.zeros([batch, rows, cols, depth], dtype=numpy.float32)
for ba in range(batch):
	for r in range(0,rows,2):
		for c in range(0,cols,2):
			for d in range(depth):
				i = (numpy.random.rand(2)*2).astype(numpy.int32)
				aa[ba,r+i[0],c+i[1],d] = 1


a = tf.placeholder(tf.float32, [batch, rows, cols, depth])
with tf.device('cpu:0'):
	pooledTf, inds = tf.nn.max_pool_with_argmax(a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	pooledMine = cc.ops.poolByIndex(a, tf.cast(inds, tf.int32))
	original = cc.ops.upsampleIndex(pooledMine, tf.cast(inds,tf.int32), a)

with tf.device('gpu:0'):
	pooledTfCuda, indsCuda = tf.nn.max_pool_with_argmax(a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
	pooledMineCuda = cc.ops.poolByIndex(a, tf.cast(indsCuda, tf.int32))
	originalCuda = cc.ops.upsampleIndex(pooledMineCuda, tf.cast(indsCuda,tf.int32), a)

# e = tf.nn.max_pool(a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
sess = tf.Session()
res = sess.run([pooledTf, pooledMine, original, pooledTfCuda, pooledMineCuda, originalCuda], feed_dict={a:aa})
res = sess.run([pooledMine, pooledTf], feed_dict={a:aa})

numpy.sum(res[0] != res[1])
numpy.sum(res[3] != res[4])

numpy.sum(res[2] != res[-1])
numpy.sum(aa != res[2])

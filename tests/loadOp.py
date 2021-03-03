import tensorflow as tf
import numpy

ga = tf.load_op_library("gather_angles.so")
convByIndex = tf.load_op_library("cliffordConvolution/objectFiles/gather_angles.so")
convByIndex = tf.load_op_library("cliffordConvolution/objectFiles/conv_by_index_2d.so")
# convByIndex2 = tf.load_op_library("./conv_by_index_2dOld.so")
inpt = numpy.ones([3,3,3,3], dtype=numpy.float32)
weights = numpy.ones([3,3,3,3,3], dtype=numpy.float32)
indexes = numpy.ones([3,3,3,3], dtype=numpy.int32)
mask = numpy.ones([3,3,3,3], dtype=numpy.int32)

mask[1] = 0
# weights[0] = 0
# weights[1] = 2
# weights[2] = 1

indexes[:,0,:,:] = 0
indexes[:,1,:,:] = 1
indexes[:,2,:,:] = 2


out = convByIndex.conv_by_index2d(inpt, weights, indexes, mask, [1,1,1,1], "SAME")
out2 = convByIndex2.conv_by_index2d_old(inpt, weights, indexes, [1,1,1,1], "SAME")

inpt = tf.placeholder(tf.float32, [28, 56, 56, 256])
for l in range(20):
	weights1 = tf.constant(numpy.random.rand(3,3,256,256))
	weights2 = tf.constant(numpy.random.rand(3,3,256,128))



sess = tf.Session()
o = sess.run(out)
o2 = sess.run(out2)



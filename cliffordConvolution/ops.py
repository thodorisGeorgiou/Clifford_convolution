import os
import inspect
import numpy
import tensorflow as tf

path = os.path.abspath(inspect.getfile(inspect.currentframe())).replace("ops.py", '')
gatherAngles = tf.load_op_library(path+"/objectFiles/gather_angles.so").gather_angles
convByIndex = tf.load_op_library(path+"/objectFiles/conv_by_index_2d.so").conv_by_index2d
convByIndexInputGrads = tf.load_op_library(path+"/objectFiles/conv_by_index_input_grads.so").conv_by_index_input_grads2d
convByIndexWeightGrads = tf.load_op_library(path+"/objectFiles/conv_by_index_weight_grads.so").conv_by_index_weight_grads2d
reduceIndex = tf.load_op_library(path+"/objectFiles/reduce_index.so").reduce_index
expandIndex = tf.load_op_library(path+"/objectFiles/expand_index.so").expand_index
offsetCorrect = tf.load_op_library(path+"/objectFiles/offset_correct.so").offset_correct
weightToAngleGradients = tf.load_op_library(path+"/objectFiles/weight_to_angle_gradients.so").weight_to_angle_gradients
poolByIndex = tf.load_op_library(path+"/objectFiles/pool_by_index.so").pool_by_index
upsampleIndex = tf.load_op_library(path+"/objectFiles/upsample_index.so").upsample_index

def batch_normalization_moving_averages(MOVING_AVERAGE_DECAY=0.9999):
	normalization_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, name='avg')
	normParams = tf.get_collection('normalizationParameters')
	norm_averages_op = normalization_averages.apply(normParams)
	return norm_averages_op

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

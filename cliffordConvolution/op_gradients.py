from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import tensorflow as tf
import numpy
from . import ops as ccOps
from . import misc
from . import transformations

@ops.RegisterGradient("PoolByIndex")
def _pool_index_grad(op, grad):
	upsampled_grads = ccOps.upsampleIndex(grad, op.inputs[1], op.inputs[0])
	return [upsampled_grads, None]

@ops.RegisterGradient("GatherAngles")
def _gather_angles_grad(op, grad):
	return [None, None, grad]

@ops.RegisterGradient("OffsetCorrect")
def _offset_correct_grad(op, grad0, grad1):
	return [grad0*tf.cast(op.outputs[1], tf.float32), None]

@ops.RegisterGradient("ReduceIndex")
def _reduce_index_grad(op, grad):
	expandedGrads = ccOps.expandIndex(grad, op.inputs[1], op.inputs[0])
	return [expandedGrads, None]

@ops.RegisterGradient("ConvByIndex2D")
def _conv_by_index_2d_grad(op, grad):
	thresholdedGrads = grad*tf.cast(op.inputs[3], tf.float32)
	inputGrads = ccOps.convByIndexInputGrads(op.inputs[0], op.inputs[1], thresholdedGrads, op.inputs[2],\
	dilations=op.get_attr("dilations"), strides=op.get_attr("strides"),\
	padding=op.get_attr("padding"), data_format=op.get_attr("data_format"))
	weightGrads = ccOps.convByIndexWeightGrads(op.inputs[0], op.inputs[1], thresholdedGrads, op.inputs[2],\
	dilations=op.get_attr("dilations"), strides=op.get_attr("strides"),\
	padding=op.get_attr("padding"), data_format=op.get_attr("data_format"))
	thetaGrads = ccOps.weightToAngleGradients(op.inputs[0], thresholdedGrads, op.inputs[1], op.inputs[2],\
	dilations=op.get_attr("dilations"), strides=op.get_attr("strides"),\
	padding=op.get_attr("padding"), data_format=op.get_attr("data_format"))
	return [inputGrads, weightGrads, None, None, thetaGrads]

@tf.custom_gradient
def weightToThetaGradsSteerable(inpt, output, mask, weights, qThetas, thetas, steerableCoefs, weightMask):
	def grad(dy):
		weightShape = weights.get_shape().as_list()
		numAngels = weightShape[0]
		num_bins = weightShape[0]-1
		offset = num_bins//(2*misc.num_angles)
		npAngles = [[(a-offset)*2*numpy.pi/num_bins] for a in range(num_bins+1)]
		tWeights = tf.reshape(tf.transpose(weights, [1,2,3,4,0]), [weightShape[1], weightShape[2], weightShape[3], weightShape[4]*weightShape[0]])
		verTWeights = transformations.rotateVectors(tWeights, [numpy.pi/2])
		verWeights = tf.transpose(tf.reshape(verTWeights, [weightShape[1], weightShape[2], weightShape[3], weightShape[4], weightShape[0]]), [4,0,1,2,3])
		# exInpt = tf.expand_dims(inpt, axis=1)
		# exInpt = tf.expand_dims(exInpt, axis=1)
		# exInpt = tf.expand_dims(exInpt, axis=1)
		gradientKernels = misc.getSteerableGradientKernel(weightShape[1], npAngles, weightShape[3]//2, weightShape[4], steerableCoefs, weightMask)
		woa = verWeights + gradientKernels
		thetaGrads = ccOps.convByIndex(inpt, woa, qThetas, mask, thetas, [1,1,1,1], "SAME")*dy*tf.cast(mask, tf.float32)
		# allWoa = verWeights + gradientKernels
		# allWoaL = tf.unstack(allWoa, axis=-1)
		# indL = tf.unstack(qThetas, axis=-1)
		# woa = []
		# for i, ind in enumerate(indL):
		# 	woa.append(tf.gather(allWoaL[i], ind))
		# woa = tf.stack(woa, axis=3)
		# woa = tf.gather(woa, qThetas)
		# woa = tf.reduce_sum(tf.multiply(woa, npReduction), axis=-1)
		# utGKernels = tf.gather(gradientKernels, qThetas)
		# utGKernels = tf.reduce_sum(tf.multiply(utGKernels, npReduction), axis=-1)
		# kerDefGrads = tf.reduce_sum(tf.multiply(exInpt, utGKernels), axis=[-3,-2,-1])
		# thetaGrads = tf.reduce_sum(tf.multiply(exInpt, woa), axis=[-3,-2,-1])*dy*mask
		return (None, dy, None, None, None, thetaGrads, None, None)
	return tf.identity(output), grad

@tf.custom_gradient
def weightToThetaGradsNoMaskSteerable(inpt, output, mask, weights, qThetas, thetas, steerableCoefs):
	def grad(dy):
		weightShape = weights.get_shape().as_list()
		numAngels = weightShape[0]
		num_bins = weightShape[0]-1
		offset = num_bins//(2*misc.num_angles)
		npAngles = [[(a-offset)*2*numpy.pi/num_bins] for a in range(num_bins+1)]
		tWeights = tf.reshape(tf.transpose(weights, [1,2,3,4,0]), [weightShape[1], weightShape[2], weightShape[3], weightShape[4]*weightShape[0]])
		verTWeights = transformations.rotateVectors(tWeights, [numpy.pi/2])
		verWeights = tf.transpose(tf.reshape(verTWeights, [weightShape[1], weightShape[2], weightShape[3], weightShape[4], weightShape[0]]), [4,0,1,2,3])
		# exInpt = tf.expand_dims(inpt, axis=1)
		# exInpt = tf.expand_dims(exInpt, axis=1)
		# exInpt = tf.expand_dims(exInpt, axis=1)
		gradientKernels = misc.getSteerableGradientKernel(weightShape[1], npAngles, weightShape[3]//2, weightShape[4], steerableCoefs, None)
		woa = verWeights + gradientKernels
		thetaGrads = ccOps.convByIndex(inpt, woa, qThetas, mask, thetas, [1,1,1,1], "SAME")*dy*tf.cast(mask, tf.float32)
		# allWoa = verWeights + gradientKernels
		# allWoaL = tf.unstack(allWoa, axis=-1)
		# indL = tf.unstack(qThetas, axis=-1)
		# woa = []
		# for i, ind in enumerate(indL):
		# 	woa.append(tf.gather(allWoaL[i], ind))
		# woa = tf.stack(woa, axis=3)
		# woa = tf.gather(woa, qThetas)
		# woa = tf.reduce_sum(tf.multiply(woa, npReduction), axis=-1)
		# utGKernels = tf.gather(gradientKernels, qThetas)
		# utGKernels = tf.reduce_sum(tf.multiply(utGKernels, npReduction), axis=-1)
		# kerDefGrads = tf.reduce_sum(tf.multiply(exInpt, utGKernels), axis=[-3,-2,-1])
		# thetaGrads = tf.reduce_sum(tf.multiply(exInpt, woa), axis=[-3,-2,-1])*dy*mask
		return (None, dy, None, None, None, thetaGrads, None)
	return tf.identity(output), grad

@tf.custom_gradient
def weightToThetaGrads(inpt, output, mask, weights, qThetas, thetas):
	def grad(dy):
		numAngels = weights.get_shape()[0].value
		w1Inds = tf.where(tf.equal(qThetas, 0), tf.ones_like(qThetas)*(numAngels-2), qThetas-1)
		w2Inds = tf.where(tf.equal(qThetas, numAngels-1), tf.ones_like(qThetas), qThetas+1)
		listWeights = tf.unstack(weights, axis=-1)
		lw1Inds = tf.unstack(w1Inds, axis=-1)
		lw2Inds = tf.unstack(w2Inds, axis=-1)
		w1 = []; w2 = []
		for i in range(qThetas.get_shape()[-1].value):
			w1.append(tf.gather(listWeights[i], lw1Inds[i]))
			w2.append(tf.gather(listWeights[i], lw2Inds[i]))
		w1 = tf.stack(w1, axis=3)
		w2 = tf.stack(w2, axis=3)
		normFactor = 4*numpy.pi/(numAngels-1)
		woa = (w2-w1)/normFactor
		exInpt = tf.expand_dims(inpt, axis=1)
		exInpt = tf.expand_dims(exInpt, axis=1)
		exInpt = tf.expand_dims(exInpt, axis=1)
		exInpt = tf.tile(exInpt, [1,1,1,output.get_shape()[-1].value,1,1,1])
		inptWoa = tf.reduce_sum(tf.multiply(exInpt, woa), axis=[-3,-2,-1])
		thetaGrads = tf.multiply(inptWoa, dy*mask)
		return (None, dy, None, None, None, thetaGrads)
	return tf.identity(output), grad

@ops.RegisterGradient("replaceNaNs")
def replaceGradientNans(op, grad):
	noNans = tf.where(tf.is_nan(grad), tf.zeros_like(grad), grad)
	noNans = tf.where(tf.is_inf(noNans), tf.zeros_like(noNans), noNans)
	return noNans

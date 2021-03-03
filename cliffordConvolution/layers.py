import numpy
import tensorflow as tf
from . import ops
from . import misc
from . import transformations
from . import op_gradients

default_bn_decay = 0.9
default_bn_epsilon = 1e-5

@tf.custom_gradient
def maskAngleGradients(angles, mask):
	def grad(dy):
		# return two gradients, one for 'x' and one for 'z'
		return (dy*mask, None)
	return tf.identity(angles), grad

@tf.custom_gradient
def maskGradients(tensor, mask):
	def grad(dy):
		# return two gradients, one for 'x' and one for 'z'
		return (dy*mask, None)
	return tf.identity(tensor), grad

@tf.custom_gradient
def monitorGrads(inpt):
	def grad(dy):
		# sepVals = tf.unstack(dy)
		# for sv in range(len(sepVals)):
		# tf.summary.scalar(str(sv), tf.reduce_mean(sepVals[sv]))
		tf.summary.scalar("gradAvg", tf.reduce_mean(tf.abs(dy)))
		tf.summary.scalar("gradMax", tf.reduce_max(tf.abs(dy)))
		return dy
	return tf.identity(inpt), grad

@tf.custom_gradient
def assertGrads(inpt, message):
	def grad(dy):
		assert_op0 = tf.Assert(tf.reduce_all(tf.logical_not(tf.is_nan(dy))), [message, "NaN"])
		assert_op1 = tf.Assert(tf.reduce_all(tf.logical_not(tf.is_inf(dy))), [message, "inf"])
		with tf.control_dependencies([assert_op0, assert_op1]):
			dy = tf.identity(dy)
		return (dy, None)
	return tf.identity(inpt), grad


def conv_2tf(inpt, weights, c_i, c_o, s_h, s_w, padding, monitor=False):
	if inpt.get_shape()[-1] % 2 != 0:
		print("conv_1 input must be a vector field")
		return None
	if inpt.get_shape()[-1] != 2*c_i:
		print("Number of inputs does not match input size")
		return None
	if weights.get_shape()[-1] != c_o:
		print("Number of outputs does not match weight tensor size")
		return None
	inptFeatMpas = tf.unstack(inpt, axis=-1)
	weightsSep = tf.unstack(weights, axis=-2)
	inptMona = tf.stack(inptFeatMpas[1::2], axis=-1)
	inptZiga = tf.stack(inptFeatMpas[::2], axis=-1)
	weightMona = tf.stack(weightsSep[1::2], axis=-2)
	weightZiga = tf.stack(weightsSep[::2], axis=-2)
	if monitor:
		with tf.device('/cpu:0'):
			tf.summary.histogram("weights_X", weightZiga)
			tf.summary.histogram("weights_Y", weightMona)
	first = tf.nn.conv2d(inptZiga, weightMona, [1, s_h, s_w, 1], padding=padding)
	second = tf.nn.conv2d(inptMona, weightZiga, [1, s_h, s_w, 1], padding=padding)
	# return first - second
	return second - first
	# return [first - second, first+second]

def getRotation(inpt, rotatedWeights, c_i, c_o, s_h, s_w, offset, padding, num_angles=4, num_bins=32):
	thetas = []
	conv0L = []
	for angle in range(0, num_bins, num_bins//num_angles):
		weightSet = tf.gather(rotatedWeights, angle + offset)
		conv2 = conv_2tf(inpt, weightSet, c_i, c_o, s_h, s_w, padding)
		conv0 = tf.nn.conv2d(inpt, weightSet, [1,s_h,s_w,1], padding)
		# angleMask = tf.abs(conv2)<1e-3
		# angleMask = tf.logical_and(tf.abs(conv0)<1e-2, tf.abs(conv2)<1e-2)
		angleMask = (tf.square(conv0) + tf.square(conv2)) < 1.5e-4
		conv0 = tf.where(angleMask, tf.ones_like(conv0)*(-1), conv0)
		conv2 = tf.where(angleMask, tf.zeros_like(conv2), conv2)
		conv0L.append(conv0)
		thetas.append(tf.atan2(conv2, conv0))
	thetas = tf.stack(thetas, axis=-1)
	conv0 = tf.stack(conv0L, axis=-1)
	# winner = tf.argmin(tf.abs(thetas), axis=-1, output_type=tf.int32)
	winner = tf.argmax(conv0, axis=-1, output_type=tf.int32)
	thetas2 = ops.reduceIndex(thetas, winner)
	thetas2, convMask = ops.offsetCorrect(thetas2, [2*numpy.pi/num_angles])
	tf.add_to_collection("conv_mask_by_angle", convMask)
	winnerInd = tf.cast(winner * (num_bins//num_angles), tf.int32) + offset
	theta2Ind = thetas2*num_bins/(2*numpy.pi)
	quantized = tf.cast(tf.round(theta2Ind), tf.int32) + winnerInd
	quantized = tf.where(quantized>num_bins, quantized-num_bins, quantized)
	quantized = tf.where(quantized<0, quantized+num_bins, quantized)
	return quantized, thetas2, convMask, winnerInd

def cliffordConv(inpt, weights, c_i, c_o, s_h, s_w, padding, num_angles=4, num_bins=32, mask=False, count=False, weightMask=None, steerable=False, first=False):
	offset = num_bins//(2*num_angles)
	npAngles = [[(a-offset)*2*numpy.pi/num_bins] for a in range(num_bins+1)]
	angles = [tf.constant([(a-offset)*2*numpy.pi/num_bins]) for a in range(num_bins+1)]
	# thetas = []
	rotatedWeights = []
	# weightSize = int(weights.get_shape().as_list()[0]/2)
	if not steerable:
		weightsShape = weights.get_shape().as_list()
		try:
			multipliers = tf.constant(misc.multipliers[num_bins][weightsShape[0]])
			# multipliers = tf.constant([1.0 for i in range(num_bins+1)])
		except KeyError:
			exit("multipliers for filter size "+str(weightsShape[0])+" do not exist. Change filter size or add them manually.")
		# angles = [tf.constant([a*2*numpy.pi/num_bins]) for a in range(num_bins+1)]
		for angle in angles:
			rWeight = transformations.rotateVectorField(weights, tf.stop_gradient(angle), returnIrelevantAxisFirst=True)
			# rw = transformations.rotateVectorField(weights, angle, returnIrelevantAxisFirst=True)
			# rWeight = tf.image.resize(rw, tf.constant([weightSize, weightSize], dtype=tf.int32))
			if weightMask != None:
				rotatedWeights.append(tf.transpose(rWeight, [1,2,3,0])* weightMask)
			else:
				# rotatedWeights.append(normalizeVectorField(transformations.rotateVectorField(weights, angle), weightsShape[0], weightsShape[1]))
				rotatedWeights.append(tf.transpose(rWeight, [1,2,3,0]))
	else:
		for angle_index, angle in enumerate(npAngles):
			reuse = ((angle_index!=0) or (not first))
			with tf.variable_scope("weights", reuse=reuse) as scope:
				if weightMask != None:
					rotatedWeights.append(misc.getSteerableKernel(weights, angle, c_i, c_o, first=(not reuse))*weightMask)
				else:
					rotatedWeights.append(misc.getSteerableKernel(weights, angle, c_i, c_o, first=(not reuse)))
	rotatedWeights = tf.stack(rotatedWeights, axis=0)
	# for angle in range(0, num_bins, num_bins//num_angles):
	# 	weightSet = tf.gather(rotatedWeights, angle + offset)
	# 	conv2 = conv_2tf(inpt, weightSet, c_i, c_o, s_h, s_w, padding)
	# 	conv0 = tf.nn.conv2d(inpt, weightSet, [1,s_h,s_w,1], padding)
	# 	# angleMask = tf.abs(conv2)<1e-3
	# 	# angleMask = tf.logical_and(tf.abs(conv0)<1e-2, tf.abs(conv2)<1e-2)
	# 	angleMask = (tf.square(conv0) + tf.square(conv2)) < 1.5e-4
	# 	if count:
	# 		with tf.device('/cpu:0'):
	# 			size = 1
	# 			for shape_value in angleMask.get_shape():
	# 				size *= shape_value
	# 			nonZeroMask = tf.count_nonzero(tf.math.logical_not(angleMask))
	# 			tf.summary.scalar("nonZero_mask", nonZeroMask/size.value)
	# 	conv0 = tf.where(angleMask, tf.ones_like(conv0)*(-1), conv0)
	# 	conv2 = tf.where(angleMask, tf.zeros_like(conv2), conv2)
	# 	thetas.append(tf.atan2(conv2, conv0))
	# thetas = tf.stack(thetas, axis=-1)
	# winner = tf.argmin(tf.abs(thetas), axis=-1, output_type=tf.int32)
	# thetas2 = ops.reduceIndex(thetas, winner)
	# thetas2, convMask = ops.offsetCorrect(thetas2, [2*numpy.pi/num_angles])
	# tf.add_to_collection("conv_mask_by_angle", convMask)
	# winnerInd = tf.cast(winner * (num_bins//num_angles), tf.int32) + offset
	# theta2Ind = thetas2*num_bins/(2*numpy.pi)
	# quantized = tf.cast(tf.round(theta2Ind), tf.int32) + winnerInd
	# quantized = tf.where(quantized>num_bins, quantized-num_bins, quantized)
	# quantized = tf.where(quantized<0, quantized+num_bins, quantized)
	quantized, thetas2, convMask, winnerInd = getRotation(inpt, rotatedWeights, c_i, c_o, s_h, s_w, offset, padding, num_angles, num_bins)
	angles = tf.concat(angles, axis=0)
	if steerable:
		res = ops.convByIndex(inpt, rotatedWeights, quantized, convMask, tf.stop_gradient(thetas2), [1,1,1,1], "SAME")
		steerableCoefs = tf.get_collection("weights")[-1]
		if weightMask != None:
			res = op_gradients.weightToThetaGradsSteerable(inpt, res, convMask, rotatedWeights, quantized, thetas2, steerableCoefs, weightMask)
		else:
			res = op_gradients.weightToThetaGradsNoMaskSteerable(inpt, res, convMask, rotatedWeights, quantized, thetas2, steerableCoefs)
	else:
		res = ops.convByIndex(inpt, rotatedWeights, quantized, convMask, thetas2, [1,1,1,1], "SAME")
	# resAngleQuantized = ops.gatherAngles(angles, quantized, thetas2)
	resAngleQuantized = tf.gather(angles, quantized)
	resAngle = thetas2 + tf.gather(angles, winnerInd)
	if not steerable:
		resMultiplier = tf.gather(multipliers, quantized)
	# resAngle = thetas2 + tf.cast(winner,tf.float32)*2*numpy.pi/num_angles
	# resAngle = resAngleQuantized
	# diff = tf.abs(resAngle - resAngleQuantized)
	# diff = resAngle - resAngleQuantized
	diff = resAngleQuantized - resAngle
	if steerable:
		res = res/tf.cos(diff)
	else:
		res = res*resMultiplier/tf.cos(diff)
	# res = res*resMultiplier/tf.stop_gradient(tf.cos(diff))
	if count:
		with tf.device('/cpu:0'):
			nonZeroRes = tf.count_nonzero(res)
			negativeRes = tf.count_nonzero(tf.less(res, 0.0))
			# nonZeroResAngles = tf.count_nonzero(resAngle)
			# tf.summary.scalar("angleCorrection", tf.reduce_sum(tf.cast(diff>numpy.pi/num_angles, tf.float32)))
			tf.summary.scalar("nonZero_res", nonZeroRes/size.value)
			tf.summary.scalar("negative_res", negativeRes/size.value)
			# tf.summary.scalar("nonZero_resAngles", nonZeroResAngles/size.value)
			# tf.summary.histogram("quantized", quantized)
			# nans = tf.reduce_sum(tf.cast(tf.is_nan(inpt), tf.float32))
			# infs = tf.reduce_sum(tf.cast(tf.is_inf(inpt), tf.float32))
			# zeros = tf.reduce_sum(tf.cast(tf.equal(inpt, 0), tf.float32))
			# tf.summary.scalar("nans", nans)
			# tf.summary.scalar("infs", infs)
			# tf.summary.scalar("zeros", zeros)
	if mask:
		return res, resAngle, convMask
	else:
		return res, resAngle

def cliffordConvPerChannelAngle(inpt, weights, c_i, c_o, s_h, s_w, padding, useType="test", num_angles=4, num_bins=32, mask=False, count=False, weightMask=None, steerable=False, first=False):
	offset = num_bins//(2*num_angles)
	npAngles = [[(a-offset)*2*numpy.pi/num_bins] for a in range(num_bins+1)]
	angles = [tf.constant([(a-offset)*2*numpy.pi/num_bins]) for a in range(num_bins+1)]
	rotatedWeights = []
	weightsShape = weights.get_shape().as_list()
	for angle in angles:
		rWeight = transformations.rotateVectorField(weights, angle, returnIrelevantAxisFirst=True)
		if weightMask != None:
			rotatedWeights.append(tf.transpose(rWeight, [1,2,3,0])* weightMask)
		else:
			rotatedWeights.append(tf.transpose(rWeight, [1,2,3,0]))
	rotatedWeights = tf.stack(rotatedWeights, axis=0)
	perInChannelWeights = tf.split(rotatedWeights, c_i, axis=-2)
	perInChannelInpt = tf.split(inpt, c_i, axis=-1)
	angles = tf.concat(angles, axis=0)
	perInChannelRes = []
	for i in range(c_i):
		with tf.variable_scope(str(i)) as scope:
			quantized, thetas2, convMask, winnerInd = getRotation(perInChannelInpt[i], perInChannelWeights[i], 1, c_o, s_h, s_w, offset, "SAME", num_angles, num_bins)
			partRes = ops.convByIndex(perInChannelInpt[i], perInChannelWeights[i], quantized, convMask, thetas2, [1,1,1,1], "SAME")
			resAngleQuantized = tf.gather(angles, quantized)
			resAngle = thetas2 + tf.gather(angles, winnerInd)
			diff = tf.abs(resAngle - resAngleQuantized)
			partRes = partRes/tf.cos(diff)
			if padding=="VALID":
				weightsShape = weights.get_shape().as_list()
				inptShape = inpt.get_shape().as_list()
				k_w = weightsShape[0]
				k_h = weightsShape[1]
				i_w = inptShape[1]
				i_h = inptShape[2]
				partRes = tf.image.crop_to_bounding_box(partRes, numpy.floor(k_w/2).astype(numpy.int32), numpy.floor(k_h/2).astype(numpy.int32), i_w-2*numpy.floor(k_w/2).astype(numpy.int32), i_h-2*numpy.floor(k_h/2).astype(numpy.int32))
				resAngle = tf.image.crop_to_bounding_box(resAngle, numpy.floor(k_w/2).astype(numpy.int32), numpy.floor(k_h/2).astype(numpy.int32), i_w-2*numpy.floor(k_w/2).astype(numpy.int32), i_h-2*numpy.floor(k_h/2).astype(numpy.int32))
			# partRes = batch_norm_only_rescale_learn_scale(partRes, useType, reuse=(not first))
			partRes = tf.nn.relu(partRes)
			reluMask = tf.where(partRes>0, tf.ones_like(partRes), tf.zeros_like(partRes))
			regulatedAngles = maskAngleGradients(resAngle, reluMask)
			perInChannelRes.append(transformations.changeToCartesian(partRes, regulatedAngles))
	res = tf.reduce_sum(tf.stack(perInChannelRes, axis=0), axis=0)
	return res


def fcCliffordLayer(inpt, c_i, c_o, s_h, s_w, first, useType, wd=misc.default_wd, num_bins=16, num_angles=4, normalize=True, resuse_batch_norm=False):
	fcW, fcb = misc.getWeightsNBiases(first, 1, 1, c_o, 2*c_i, s_h, s_w, wd=wd, lossType="nl")
	offset = num_bins//(2*num_angles)
	angles = [tf.constant([(a-offset)*2*numpy.pi/num_bins]) for a in range(num_bins+1)]
	rotatedWeightsL = []
	thetas = []
	conv0L = []
	activations0 = []
	activations2 = []
	for angle in angles:
		rotatedWeightsL.append(transformations.rotateVectors(fcW, tf.stop_gradient(angle)))

	for angle in range(0, num_bins, num_bins//num_angles):
		weightSet = rotatedWeightsL[angle + offset]
		if useType == "train" and first:
			with tf.variable_scope(str(angle)) as scope:
				fcConv2 = conv_2tf(inpt, weightSet, c_i, c_o, 1, 1, "VALID", monitor=False)
		else:
			fcConv2 = conv_2tf(inpt, weightSet, c_i, c_o, 1, 1, "VALID")
		fcConv0 = tf.nn.conv2d(inpt, weightSet, [1,1,1,1], "VALID")
		angleMask = (tf.square(fcConv0) + tf.square(fcConv2)) < 1.5e-4
		fcConv0 = tf.where(angleMask, tf.ones_like(fcConv0)*(-1), fcConv0)
		fcConv2 = tf.where(angleMask, tf.zeros_like(fcConv2), fcConv2)
		thetas.append(tf.atan2(fcConv2, fcConv0))
		conv0L.append(fcConv0)

	thetas = tf.stack(thetas, axis=-1)
	conv0 = tf.stack(conv0L, axis=-1)
	# winner = tf.argmin(tf.abs(thetas), axis=-1, output_type=tf.int32)
	winner = tf.argmax(conv0, axis=-1, output_type=tf.int32)
	thetas2 = ops.reduceIndex(thetas, winner)
	thetas2, convMask = ops.offsetCorrect(thetas2, [2*numpy.pi/num_angles])
	tf.add_to_collection("conv_mask_by_angle", convMask)
	winnerInd = tf.cast(winner * (num_bins//num_angles), tf.int32) + offset
	quantized = tf.cast(tf.round(thetas2*num_bins/(2*numpy.pi)), tf.int32) + winnerInd
	quantized = tf.where(quantized>num_bins , quantized-num_bins , quantized)
	quantized = tf.where(quantized<0, quantized+num_bins , quantized)
	floatMask = tf.cast(convMask, tf.float32)

	flatConvCudaResL = []
	for rotation in range(num_bins+1):
		rMask = tf.cast(tf.equal(quantized, rotation), tf.float32)
		flatConvCudaResL.append(tf.nn.conv2d(inpt, rotatedWeightsL[rotation], [1,1,1,1], "VALID")*rMask)
	rotatedWeights = tf.stack(rotatedWeightsL, axis=0)
	convRes = tf.reduce_sum(tf.stack(flatConvCudaResL, axis=0), axis=0)*floatMask
	convRes = op_gradients.weightToThetaGrads(inpt, convRes, floatMask, rotatedWeights, quantized, thetas2)
	angles = tf.concat(angles, axis=0)
	resAngleQuantized = tf.gather(angles, quantized)
	resAngle = thetas2 + tf.gather(angles, winnerInd)
	# diff = tf.abs(resAngle - resAngleQuantized)
	# diff = resAngle - resAngleQuantized
	diff = resAngleQuantized - resAngle
	convRes = convRes/tf.cos(diff)
	# convRes = convRes*resMultiplier/tf.cos(diff)
	# convRes = convRes/tf.stop_gradient(tf.cos(diff))
	if normalize:
		# fc = batch_norm(convRes, useType, resuse_batch_norm) * floatMask
		fc = batch_norm_only_rescale_learn_scale(convRes, useType, resuse_batch_norm)*floatMask
		fc = tf.nn.relu(fc)
		reluMask = tf.where(fc>0, tf.ones_like(fc), tf.zeros_like(fc))
		regulatedAngles = maskAngleGradients(resAngle, reluMask)
	else:
		fc = tf.nn.bias_add(convRes, fcb)
		regulatedAngles = resAngle
	tf.add_to_collection("activationMagnitudes", fc)
	tf.add_to_collection("activationAngles", resAngle)
	return fc, regulatedAngles

def whatShouldBeConvAllRotations(inpt, weights, biases, s_h, s_w, padding, num_bins=32, weightMask=None):
	angles = [tf.constant([a*2*numpy.pi/num_bins]) for a in range(num_bins)]
	convPerAngle = []
	weightsShape = weights.get_shape().as_list()
	for angle in angles:
		if weights.get_shape()[-2].value == 1:
			rotatedWeights = tf.transpose(tf.contrib.image.rotate(tf.transpose(weights, [3,0,1,2]), angle, interpolation="BILINEAR"), [1,2,3,0])
		else:
			rotatedWeights = transformations.rotateVectorField(weights, angle)
		if weightMask == None:
			partRes = tf.nn.conv2d(inpt, rotatedWeights, [1,s_h,s_w,1], padding)
		else:
			partRes = tf.nn.conv2d(inpt, rotatedWeights*weightMask, [1,s_h,s_w,1], padding)
		convPerAngle.append(tf.nn.bias_add(partRes, biases))
	# convPerAngle = tf.concat(convPerAngle, axis=-1)
	convPerAngle = tf.stack(convPerAngle, axis=-2)
	return convPerAngle

def rotInvarianceWithArgMax(inpt, weights, biases, s_h, s_w, padding, num_bins=32, weightMask=None, scalar=False):
	angles = [tf.constant([a*2*numpy.pi/num_bins]) for a in range(num_bins)]
	convPerAngle = []
	weightsShape = weights.get_shape().as_list()
	for angle in angles:
		if scalar:
			rotatedWeights = tf.transpose(tf.contrib.image.rotate(tf.transpose(weights, [3,0,1,2]), angle, interpolation="BILINEAR"), [1,2,3,0])
		else:
			rotatedWeights = transformations.rotateVectorField(weights, angle)
		if weightMask == None:
			convPerAngle.append(tf.nn.conv2d(inpt, rotatedWeights, [1,s_h,s_w,1], padding))
		else:
			convPerAngle.append(tf.nn.conv2d(inpt, rotatedWeights*weightMask, [1,s_h,s_w,1], padding))
	convPerAngle = tf.stack(convPerAngle, axis=0)
	convRes = tf.reduce_max(convPerAngle, axis=0)
	convRes = tf.nn.bias_add(convRes, biases)
	convBin = tf.argmax(convPerAngle, axis=0)
	convAngle = tf.gather(angles, convBin)
	return convRes, tf.squeeze(convAngle, axis=-1)

def rotInvarianceWithMax(inpt, weights, biases, s_h, s_w, padding, num_bins=32, weightMask=None):
	angles = [tf.constant([a*2*numpy.pi/num_bins]) for a in range(num_bins)]
	convPerAngle = []
	weightsShape = weights.get_shape().as_list()
	for angle in angles:
		rotatedWeights = tf.transpose(tf.contrib.image.rotate(tf.transpose(weights, [3,0,1,2]), angle, interpolation="BILINEAR"), [1,2,3,0])
		if weightMask == None:
			convPerAngle.append(tf.nn.conv2d(inpt, rotatedWeights, [1,s_h,s_w,1], padding))
		else:
			convPerAngle.append(tf.nn.conv2d(inpt, rotatedWeights*weightMask, [1,s_h,s_w,1], padding))
	convPerAngle = tf.stack(convPerAngle, axis=0)
	convRes = tf.reduce_max(convPerAngle, axis=0)
	convRes = tf.nn.bias_add(convRes, biases)
	return convRes

def rotInvarianceWithSum(inpt, weights, biases, s_h, s_w, padding, num_bins=32, weightMask=None):
	angles = [tf.constant([a*2*numpy.pi/num_bins]) for a in range(num_bins)]
	convPerAngle = []
	weightsShape = weights.get_shape().as_list()
	for angle in angles:
		rotatedWeights = tf.transpose(tf.contrib.image.rotate(tf.transpose(weights, [3,0,1,2]), angle, interpolation="BILINEAR"), [1,2,3,0])
		if weightMask == None:
			convPerAngle.append(tf.nn.conv2d(inpt, rotatedWeights, [1,s_h,s_w,1], padding))
		else:
			convPerAngle.append(tf.nn.conv2d(inpt, rotatedWeights*weightMask, [1,s_h,s_w,1], padding))
	convPerAngle = tf.stack(convPerAngle, axis=0)
	convRes = tf.reduce_sum(convPerAngle, axis=0)
	convRes = tf.nn.bias_add(convRes, biases)
	return convRes

def convAllRotations(inpt, weightsShape, k_h, s_h, s_w, padding, first, num_bins=32, weightMask=None, steerable=False):
	angles = [a*2*numpy.pi/num_bins for a in range(num_bins)]
	# angles = [tf.constant([a*2*numpy.pi/num_bins]) for a in range(num_bins)]
	convPerAngle = []
	c_i = inpt.get_shape()[-1].value
	if steerable:
		if c_i != weightsShape[-2]:
			basis = misc.getScalarBasis(weightsShape[0], num_bins, weightsShape[-2])
		else:
			num_bins=1
			basis = None
	else:
		if c_i == weightsShape[-2]:
			num_bins=1
	for abin, angle in enumerate(angles):
		firstDec = ((abin==0) and (first))
		if steerable:
			with tf.variable_scope("weights", reuse=(not firstDec)) as scope:
				rotatedWeights = misc.getScalarSteerableKernel(weightsShape[0], angle, abin, weightsShape[-2], weightsShape[-1], num_bins=num_bins, first=firstDec, basis=basis)
		else:
			with tf.variable_scope("weights", reuse=(not firstDec)) as scope:
				rWeights, biases = misc.getWeightsNBiases(first, weightsShape[0], weightsShape[1], weightsShape[-1], num_bins*weightsShape[-2], s_h, s_w, wd=misc.default_nl, lossType="nl")
				# rWeights = tf.transpose(tf.contrib.image.rotate(tf.transpose(weights, [3,0,1,2]), angle, interpolation="BILINEAR"), [1,2,3,0])
			if num_bins == 1:
				rotatedWeights = rWeights
			else:
				rWeights = tf.reshape(rWeights, [weightsShape[0], weightsShape[1], num_bins, weightsShape[-2], weightsShape[-1]])
				inds = [i for i in range(abin,-1,-1)]+[i for i in range(num_bins-1,abin,-1)]
				# inds = [i for i in range(num_bins-abin,num_bins)]+[i for i in range(num_bins-abin)]
				inds = tf.constant(inds)
				angles = tf.cast(inds, tf.float32)*2*numpy.pi/num_bins
				angles = tf.reshape(angles, [1,1,1,num_bins, 1])
				rWeights = tf.gather(rWeights, inds, axis=-3)
				rotatedWeights = tf.reshape(rWeights, weightsShape[:2]+[weightsShape[-2]*num_bins, weightsShape[-1]])
		if weightMask == None:
			convPerAngle.append(tf.nn.conv2d(inpt, rotatedWeights, [1,s_h,s_w,1], padding))
		else:
			convPerAngle.append(tf.nn.conv2d(inpt, rotatedWeights*weightMask, [1,s_h,s_w,1], padding))
	# convPerAngle = tf.concat(convPerAngle, axis=-1)
	convPerAngle = tf.stack(convPerAngle, axis=-2)
	return convPerAngle

def convAllRotationsCartesianOut(inpt, weightsShape, k_h, s_h, s_w, padding, first, num_bins=32, weightMask=None, steerable=False, cartesian=False, cartesianIn=False):
	if cartesianIn and not cartesian:
		exit("Not supported cartesian ouput with vectorised input.")
	outAngles = [a*2*numpy.pi/num_bins for a in range(num_bins)]
	# outAngles = [tf.constant([a*2*numpy.pi/num_bins]) for a in range(num_bins)]
	convPerAngle = []
	mask = []
	c_i = inpt.get_shape()[-1].value
	if steerable:
		if c_i != weightsShape[-2]:
			basis = misc.getScalarBasis(weightsShape[0], num_bins, weightsShape[-2])
		else:
			num_bins=1
			basis = None
	else:
		if c_i == weightsShape[-2]:
			num_bins=1
	for abin, angle in enumerate(outAngles):
		firstDec = ((abin==0) and (first))
		if steerable:
			with tf.variable_scope("weights", reuse=(not firstDec)) as scope:
				rotatedWeights, angles = misc.getScalarSteerableKernel(weightsShape[0], angle, abin, weightsShape[-2], weightsShape[-1], num_bins=num_bins, first=firstDec, basis=basis, returnAngles=True)
		else:
			with tf.variable_scope("weights", reuse=(not firstDec)) as scope:
				if cartesianIn:
					weights, biases = misc.getWeightsNBiases(first, weightsShape[0], weightsShape[1], weightsShape[-1], 2*num_bins*weightsShape[-2], s_h, s_w, wd=misc.default_nl, lossType="nl")
					rWeights = tf.transpose(transformations.rotateVectorField(tf.transpose(weights, [3,0,1,2]), angle, interpolation="BILINEAR"), [1,2,3,0])
				else:
					rWeights, biases = misc.getWeightsNBiases(first, weightsShape[0], weightsShape[1], weightsShape[-1], num_bins*weightsShape[-2], s_h, s_w, wd=misc.default_nl, lossType="nl")
					# rWeights = tf.transpose(tf.contrib.image.rotate(tf.transpose(weights, [3,0,1,2]), angle, interpolation="BILINEAR"), [1,2,3,0])
			if num_bins == 1:
				rotatedWeights = rWeights
			else:
				rWeights = tf.reshape(rWeights, [weightsShape[0], weightsShape[1], num_bins, weightsShape[-2], weightsShape[-1]])
				# inds = [i for i in range(num_bins-abin,num_bins)]+[i for i in range(num_bins-abin)]
				inds = [i for i in range(abin,-1,-1)]+[i for i in range(num_bins-1,abin,-1)]
				inds = tf.constant(inds)
				angles = tf.cast(inds, tf.float32)*2*numpy.pi/num_bins
				# angles = tf.constant(outAngles, dtype=tf.float32)
				rWeights = tf.gather(rWeights, inds, axis=-3)
				rotatedWeights = tf.unstack(rWeights, axis=-3)
		if weightMask == None:
			exit("Cartesian output without weight mask not implemented yet.")
		inptL = tf.split(inpt, num_bins, axis=-1)
		angles = tf.reshape(angles, [1,1,1,num_bins, 1])
		angleConv = tf.stack([tf.nn.conv2d(inptL[inA], rotatedWeights[inA]*weightMask, [1,s_h,s_w,1], padding) for inA in range(num_bins)], axis=-2)
		# angleConv = tf.stack([tf.nn.bias_add(tf.nn.conv2d(inptL[inA], rotatedWeights[inA]*weightMask, [1,s_h,s_w,1], padding), biases/num_bins) for inA in range(num_bins)], axis=-2)
		inShape = inpt.get_shape().as_list()
		outShape = angleConv.get_shape().as_list()
		angles = tf.tile(angles, outShape[:-2]+[1,weightsShape[-1]])
		# angles=tf.reshape(angles, inShape[:-1]+[weightsShape[-1]*num_bins])
		magnSum = tf.expand_dims(tf.reduce_sum(angleConv, axis=-2), axis=-1)
		# tf.add_to_collection("magnSum", magnSum)
		magnSum = tf.reshape(tf.tile(magnSum, [1, 1, 1, 1, 2]), outShape[:-2]+[2*weightsShape[-1]])
		# cart = transformations.changeToCartesian(tf.nn.relu(angleConv), angles)
		cart = transformations.changeToCartesian(angleConv, angles)
		cart = tf.reduce_sum(cart, axis=-2)
		cart = tf.where(magnSum>0, cart, tf.zeros_like(cart))
		cart = transformations.rotateVectors(tf.transpose(cart, [1,2,3,0]), angle)
		cart = tf.transpose(cart, [3,0,1,2])
		convPerAngle.append(cart)
		# magnSum = tf.reduce_sum(angleConv, axis=-2)
		# mask.append(tf.where(magnSum>0, tf.ones_like(magnSum), tf.zeros_like(magnSum)))
	# convPerAngle = tf.concat(convPerAngle, axis=-1)
	convPerAngle = tf.stack(convPerAngle, axis=-2)
	# mask = tf.stack(mask, axis=-2)
	mask = None
	return convPerAngle, mask

def cliffordConvAllRotations(inpt, weightsShape, k_h, s_h, s_w, padding, first, num_bins=32, num_angles=4, weightMask=None, steerable=False, cartesian=False, cartesianIn=False):
	outAngles = [[a*2*numpy.pi/num_bins] for a in range(num_bins+1)]
	rotatedWeightsL = []
	# firstDec = ((abin==0) and (first))
	with tf.variable_scope("weights", reuse=(not first)) as scope:
		weights, biases = misc.getWeightsNBiases(first, weightsShape[0], weightsShape[1], weightsShape[-1], 2*num_bins*weightsShape[-2], s_h, s_w, wd=misc.default_nl, lossType="nl")
		weightsL = tf.split(weights, num_bins, axis=-2)
	for abin, angle in enumerate(outAngles[:-1]):
		# inds = [i for i in range(num_bins-abin,num_bins)]+[i for i in range(num_bins-abin)]
		inds = [i for i in range(abin,-1,-1)]+[i for i in range(num_bins-1,abin,-1)]
		rWeightsL = []
		for ind, inFi in enumerate(inds):
			inAngle = ind*2*numpy.pi/num_bins
			rWeightsL.append(transformations.rotateVectorField(weightsL[inFi], inAngle))
		rWeights = tf.concat(rWeightsL, axis=-2)
		rotatedWeightsL.append(rWeights)

	rotatedWeightsL.append(rotatedWeightsL[0])

	thetas = []
	conv0L = []
	for angle in range(0, num_bins, num_bins//num_angles):
		weightSet = rotatedWeightsL[angle]
		fcConv2 = conv_2tf(inpt, weightSet, num_bins*weightsShape[-2], weightsShape[-1], 1, 1, "VALID")
		fcConv0 = tf.nn.conv2d(inpt, weightSet, [1,1,1,1], "VALID")
		angleMask = (tf.square(fcConv0) + tf.square(fcConv2)) < 1.5e-4
		fcConv0 = tf.where(angleMask, tf.ones_like(fcConv0)*(-1), fcConv0)
		fcConv2 = tf.where(angleMask, tf.zeros_like(fcConv2), fcConv2)
		thetas.append(tf.atan2(fcConv2, fcConv0))
		conv0L.append(fcConv0)

	thetas = tf.stack(thetas, axis=-1)
	conv0 = tf.stack(conv0L, axis=-1)
	# winner = tf.argmin(tf.abs(thetas), axis=-1, output_type=tf.int32)
	winner = tf.argmax(conv0, axis=-1, output_type=tf.int32)
	thetas2 = ops.reduceIndex(thetas, winner)
	thetas2, convMask = ops.offsetCorrect(thetas2, [2*numpy.pi/num_angles])
	winnerInd = tf.cast(winner * (num_bins//num_angles), tf.int32)
	quantized = tf.cast(tf.round(thetas2*num_bins/(2*numpy.pi)), tf.int32) + winnerInd
	quantized = tf.where(quantized>num_bins , quantized-num_bins , quantized)
	quantized = tf.where(quantized<0, quantized+num_bins , quantized)
	# wrong = tf.reduce_any(tf.greater(quantized, 0))
	# wrong2 = tf.reduce_any(tf.less(quantized, num_bins))
	# asserts = [tf.Assert(wrong, [-1]), tf.Assert(wrong2, [+1])]
	floatMask = tf.cast(convMask, tf.float32)
	# woaPlus = [rotatedWeightsL[i+1] for i in range(num_bins)] + [rotatedWeightsL[1]]
	# woaMinus = [rotatedWeightsL[num_bins-1]] + [rotatedWeightsL[i-1] for i in range(1,num_bins+1)]
	# woaL = []
	rMask = []
	for rotation in range(num_bins+1):
		rMask.append(tf.cast(tf.equal(quantized, rotation), tf.float32))
		# woaL.append((woaPlus[rotation] - woaMinus[rotation])*num_bins/(4*numpy.pi))
		# flatConvCudaResL.append(tf.nn.conv2d(inpt, rotatedWeightsL[rotation], [1,1,1,1], "VALID")*rMask)
	rMask = tf.concat(rMask, axis=-1)
	rotatedWeights = tf.concat(rotatedWeightsL, axis=-1)
	# woa = tf.concat(woaL, axis=-1)
	# with tf.control_dependencies(asserts):
	flatConvCudaRes = tf.nn.conv2d(inpt, rotatedWeights, [1,1,1,1], "VALID")*rMask
	# flatCorrection = tf.nn.conv2d(inpt, woa, [1,1,1,1], "VALID")*rMask
	inShape = inpt.get_shape().as_list()
	flatConvCudaRes = tf.reshape(flatConvCudaRes, [inShape[0], 1, 1, num_bins+1, weightsShape[-1]])
	# flatCorrection = tf.reshape(flatCorrection, [inShape[0], 1, 1, num_bins+1, weightsShape[-1]])
	convRes = tf.reduce_sum(flatConvCudaRes, axis=-2)*floatMask
	# correction = tf.reduce_sum(flatCorrection, axis=-2)*floatMask
	rotatedWeights = tf.stack(rotatedWeightsL, axis=0)
	convRes = op_gradients.weightToThetaGrads(inpt, convRes, floatMask, rotatedWeights, quantized, thetas2)
	outAngles = tf.concat(outAngles, axis=0)
	resAngleQuantized = tf.gather(outAngles, quantized)
	resAngle = thetas2 + tf.gather(outAngles, winnerInd)
	# resAngle = resAngleQuantized
	# diff = tf.abs(resAngle - resAngleQuantized)
	diff = resAngleQuantized - resAngle
	# diff = resAngle - resAngleQuantized

	convRes = convRes/tf.cos(diff)
	# convRes = convRes + correction*diff
	# convRes = convRes/tf.stop_gradient(tf.cos(diff))
	# fc = convRes
	fc = tf.nn.bias_add(convRes, biases)
	regulatedAngles = resAngle
	if first:
		tf.add_to_collection("activationMagnitudes", fc)
		tf.add_to_collection("activationAngles", resAngle)
	return fc, regulatedAngles

# def cliffordConvAllRotations(inpt, weightsShape, k_h, s_h, s_w, padding, first, num_bins=32, num_angles=4, weightMask=None, steerable=False, cartesian=False, cartesianIn=False):
# 	outAngles = [[a*2*numpy.pi/num_bins] for a in range(num_bins+1)]
# 	rotatedWeightsL = []
# 	for abin, angle in enumerate(outAngles[:-1]):
# 		firstDec = ((abin==0) and (first))
# 		with tf.variable_scope("weights", reuse=(not firstDec)) as scope:
# 			weights, biases = misc.getWeightsNBiases(firstDec, weightsShape[0], weightsShape[1], weightsShape[-1], 2*num_bins*weightsShape[-2], s_h, s_w, wd=misc.default_nl, lossType="nl")
# 			if abin == 0:
# 				fcb = biases
# 			rWeights = transformations.rotateVectorField(weights, tf.stop_gradient(angle))
# 			rWeights = tf.reshape(rWeights, [weightsShape[0], weightsShape[1], num_bins, weightsShape[-2]*2, weightsShape[-1]])
# 			# inds = [i for i in range(abin,-1,-1)]+[i for i in range(num_bins-1,abin,-1)]
# 			inds = [i for i in range(num_bins-abin,num_bins)]+[i for i in range(num_bins-abin)]
# 			inds = tf.constant(inds)
# 			rWeights = tf.gather(rWeights, inds, axis=-3)
# 			rotatedWeightsL.append(tf.reshape(rWeights, [weightsShape[0], weightsShape[1], num_bins*weightsShape[-2]*2, weightsShape[-1]]))
# 	rotatedWeightsL.append(rotatedWeightsL[0])

# 	thetas = []
# 	conv0L = []
# 	for angle in range(0, num_bins, num_bins//num_angles):
# 		weightSet = rotatedWeightsL[angle]
# 		fcConv2 = conv_2tf(inpt, weightSet, num_bins*weightsShape[-2], weightsShape[-1], 1, 1, "VALID")
# 		fcConv0 = tf.nn.conv2d(inpt, weightSet, [1,1,1,1], "VALID")
# 		angleMask = (tf.square(fcConv0) + tf.square(fcConv2)) < 1.5e-4
# 		fcConv0 = tf.where(angleMask, tf.ones_like(fcConv0)*(-1), fcConv0)
# 		fcConv2 = tf.where(angleMask, tf.zeros_like(fcConv2), fcConv2)
# 		thetas.append(tf.atan2(fcConv2, fcConv0))
# 		conv0L.append(fcConv0)

# 	thetas = tf.stack(thetas, axis=-1)
# 	conv0 = tf.stack(conv0L, axis=-1)
# 	# winner = tf.argmin(tf.abs(thetas), axis=-1, output_type=tf.int32)
# 	winner = tf.argmax(conv0, axis=-1, output_type=tf.int32)
# 	thetas2 = ops.reduceIndex(thetas, winner)
# 	thetas2, convMask = ops.offsetCorrect(thetas2, [2*numpy.pi/num_angles])
# 	winnerInd = tf.cast(winner * (num_bins//num_angles), tf.int32)
# 	quantized = tf.cast(tf.round(thetas2*num_bins/(2*numpy.pi)), tf.int32) + winnerInd
# 	quantized = tf.where(quantized>num_bins , quantized-num_bins , quantized)
# 	quantized = tf.where(quantized<0, quantized+num_bins , quantized)
# 	# wrong = tf.reduce_any(tf.greater(quantized, 0))
# 	# wrong2 = tf.reduce_any(tf.less(quantized, num_bins))
# 	# asserts = [tf.Assert(wrong, [-1]), tf.Assert(wrong2, [+1])]
# 	floatMask = tf.cast(convMask, tf.float32)
# 	# woaPlus = [rotatedWeightsL[i+1] for i in range(num_bins)] + [rotatedWeightsL[1]]
# 	# woaMinus = [rotatedWeightsL[num_bins-1]] + [rotatedWeightsL[i-1] for i in range(1,num_bins+1)]
# 	# woaL = []
# 	rMask = []
# 	for rotation in range(num_bins+1):
# 		rMask.append(tf.cast(tf.equal(quantized, rotation), tf.float32))
# 		# woaL.append((woaPlus[rotation] - woaMinus[rotation])*num_bins/(4*numpy.pi))
# 		# flatConvCudaResL.append(tf.nn.conv2d(inpt, rotatedWeightsL[rotation], [1,1,1,1], "VALID")*rMask)
# 	rMask = tf.concat(rMask, axis=-1)
# 	rotatedWeights = tf.concat(rotatedWeightsL, axis=-1)
# 	# woa = tf.concat(woaL, axis=-1)
# 	# with tf.control_dependencies(asserts):
# 	flatConvCudaRes = tf.nn.conv2d(inpt, rotatedWeights, [1,1,1,1], "VALID")*rMask
# 	# flatCorrection = tf.nn.conv2d(inpt, woa, [1,1,1,1], "VALID")*rMask
# 	inShape = inpt.get_shape().as_list()
# 	flatConvCudaRes = tf.reshape(flatConvCudaRes, [inShape[0], 1, 1, num_bins+1, weightsShape[-1]])
# 	# flatCorrection = tf.reshape(flatCorrection, [inShape[0], 1, 1, num_bins+1, weightsShape[-1]])
# 	convRes = tf.reduce_sum(flatConvCudaRes, axis=-2)*floatMask
# 	# correction = tf.reduce_sum(flatCorrection, axis=-2)*floatMask
# 	rotatedWeights = tf.stack(rotatedWeightsL, axis=0)
# 	convRes = op_gradients.weightToThetaGrads(inpt, convRes, floatMask, rotatedWeights, quantized, thetas2)
# 	outAngles = tf.concat(outAngles, axis=0)
# 	resAngleQuantized = tf.gather(outAngles, quantized)
# 	resAngle = thetas2 + tf.gather(outAngles, winnerInd)
# 	# resAngle = resAngleQuantized
# 	# diff = tf.abs(resAngle - resAngleQuantized)
# 	diff = resAngleQuantized - resAngle
# 	# diff = resAngle - resAngleQuantized

# 	convRes = convRes/tf.cos(diff)
# 	# convRes = convRes + correction*diff
# 	# convRes = convRes/tf.stop_gradient(tf.cos(diff))
# 	# fc = convRes
# 	fc = tf.nn.bias_add(convRes, fcb)
# 	regulatedAngles = resAngle
# 	if first:
# 		tf.add_to_collection("activationMagnitudes", fc)
# 		tf.add_to_collection("activationAngles", resAngle)
# 	return fc, regulatedAngles


def cliffordConvAllRotationsK(inpt, weightsShape, k_h, s_h, s_w, padding, first, num_bins=32, num_angles=4, weightMask=None, steerable=False, cartesian=False, cartesianIn=False):
	outAngles = [[a*2*numpy.pi/num_bins] for a in range(num_bins+1)]
	rotatedWeightsL = []
	# firstDec = ((abin==0) and (first))
	with tf.variable_scope("weights", reuse=(not first)) as scope:
		weights, biases = misc.getWeightsNBiases(first, weightsShape[0], weightsShape[1], weightsShape[-1], 2*num_bins*weightsShape[-2], s_h, s_w, wd=misc.default_nl, lossType="nl")
		weightsL = tf.split(weights, num_bins, axis=-2)
	for abin, angle in enumerate(outAngles[:-1]):
		# inds = [i for i in range(num_bins-abin,num_bins)]+[i for i in range(num_bins-abin)]
		inds = [i for i in range(abin,-1,-1)]+[i for i in range(num_bins-1,abin,-1)]
		rWeightsL = []
		for ind, inFi in enumerate(inds):
			inAngle = ind*2*numpy.pi/num_bins
			rWeightsL.append(transformations.rotateVectorField(weightsL[inFi], inAngle))
		rWeights = tf.concat(rWeightsL, axis=-2)
		rotatedWeightsL.append(rWeights)

	rotatedWeightsL.append(rotatedWeightsL[0])


	thetas = []
	conv0L = []
	for angle in range(0, num_bins, num_bins//num_angles):
		weightSet = rotatedWeightsL[angle]
		fcConv2 = conv_2tf(inpt, weightSet, num_bins*weightsShape[-2], weightsShape[-1], 1, 1, "SAME")
		fcConv0 = tf.nn.conv2d(inpt, weightSet, [1,1,1,1], "SAME")
		angleMask = (tf.square(fcConv0) + tf.square(fcConv2)) < 1.5e-4
		fcConv0 = tf.where(angleMask, tf.ones_like(fcConv0)*(-1), fcConv0)
		fcConv2 = tf.where(angleMask, tf.zeros_like(fcConv2), fcConv2)
		thetas.append(tf.atan2(fcConv2, fcConv0))
		conv0L.append(fcConv0)

	thetas = tf.stack(thetas, axis=-1)
	conv0 = tf.stack(conv0L, axis=-1)
	# winner = tf.argmin(tf.abs(thetas), axis=-1, output_type=tf.int32)
	winner = tf.argmax(conv0, axis=-1, output_type=tf.int32)
	thetas2 = ops.reduceIndex(thetas, winner)
	thetas2, convMask = ops.offsetCorrect(thetas2, [2*numpy.pi/num_angles])
	winnerInd = tf.cast(winner * (num_bins//num_angles), tf.int32)
	quantized = tf.cast(tf.round(thetas2*num_bins/(2*numpy.pi)), tf.int32) + winnerInd
	quantized = tf.where(quantized>num_bins , quantized-num_bins , quantized)
	quantized = tf.where(quantized<0, quantized+num_bins , quantized)

	floatMask = tf.cast(convMask, tf.float32)
	rotatedWeights = tf.stack(rotatedWeightsL, axis=0)
	convRes = ops.convByIndex(inpt, rotatedWeights, quantized, convMask, thetas2, [1,1,1,1], "SAME")

	# correction = tf.reduce_sum(flatCorrection, axis=-2)*floatMask
	outAngles = tf.concat(outAngles, axis=0)
	resAngleQuantized = tf.gather(outAngles, quantized)
	resAngle = thetas2 + tf.gather(outAngles, winnerInd)
	# resAngle = resAngleQuantized
	# diff = tf.abs(resAngle - resAngleQuantized)
	diff = resAngleQuantized - resAngle
	# diff = resAngle - resAngleQuantized

	convRes = convRes/tf.cos(diff)
	if padding == "VALID":
		convRes = tf.image.crop_to_bounding_box(convRes, numpy.floor(k_h/2).astype(numpy.int32), numpy.floor(k_h/2).astype(numpy.int32), 1, 1)
		resAngle = tf.image.crop_to_bounding_box(resAngle, numpy.floor(k_h/2).astype(numpy.int32), numpy.floor(k_h/2).astype(numpy.int32), 1, 1)

	# convRes = convRes + correction*diff
	# convRes = convRes/tf.stop_gradient(tf.cos(diff))
	# fc = convRes
	fc = tf.nn.bias_add(convRes, biases)
	regulatedAngles = resAngle
	if first:
		tf.add_to_collection("activationMagnitudes", fc)
		tf.add_to_collection("activationAngles", resAngle)
	return fc, regulatedAngles


def batch_norm(inpt, useType='test', reuse=False, bn_decay=default_bn_decay, bn_epsilon=default_bn_epsilon):
	dims = inpt.get_shape().as_list()[-1]
	if reuse:
		scale = tf.get_variable("scale")
		beta = tf.get_variable("beta")
		pop_mean = tf.get_variable("popMean")
		pop_var = tf.get_variable("popVariance")
	else:
		# scale = misc._variable('scale', dims, tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
		# beta = misc._variable('beta', dims, tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
		scale = misc._variable('scale', dims, tf.constant_initializer(1.0), trainable=True)
		beta = misc._variable('beta', dims, tf.constant_initializer(0.0), trainable=True)
		pop_mean = misc._variable('popMean', dims, tf.constant_initializer(0.0), trainable=False)
		pop_var = misc._variable('popVariance', dims, tf.constant_initializer(1.0), trainable=False)
	if useType=="train":
		batch_mean, batch_var = tf.nn.moments(inpt, axes=[i for i in range(len(inpt.get_shape().as_list())-1)])
		if not reuse:
			train_mean = tf.assign(pop_mean, pop_mean * bn_decay + batch_mean * (1 - bn_decay))
			train_var = tf.assign(pop_var, pop_var * bn_decay + batch_var * (1 - bn_decay))
			with tf.control_dependencies([train_mean, train_var]):
				return tf.nn.batch_normalization(inpt, batch_mean, batch_var, beta, scale, bn_epsilon)
		else:
			return tf.nn.batch_normalization(inpt, batch_mean, batch_var, beta, scale, bn_epsilon)
	else:
		return tf.nn.batch_normalization(inpt, pop_mean, pop_var, beta, scale, bn_epsilon)

def batch_norm_only_rescale(inpt, useType='test', reuse=False, bn_decay=default_bn_decay, bn_epsilon=default_bn_epsilon):
	dims = inpt.get_shape().as_list()[-1]
	if reuse:
		pop_var = tf.get_variable("popVariance")
	else:
		pop_var = misc._variable('popVariance', dims, tf.constant_initializer(1.0), trainable=False)
	if useType=="train":
		batch_mean, batch_var = tf.nn.moments(inpt, axes=[i for i in range(len(inpt.get_shape().as_list())-1)])
		if not reuse:
			train_var = tf.assign(pop_var, pop_var * bn_decay + batch_var * (1 - bn_decay))
			with tf.control_dependencies([train_var]):
				return tf.nn.batch_normalization(inpt, 0, batch_var, 0, 1, bn_epsilon)
		else:
			return tf.nn.batch_normalization(inpt, 0, batch_var, 0, 1, bn_epsilon)
	else:
		return tf.nn.batch_normalization(inpt, 0, pop_var, 0, 1, bn_epsilon)

def batch_norm_only_rescale_learn_scale(inpt, useType='test', reuse=False, bn_decay=default_bn_decay, bn_epsilon=default_bn_epsilon):
	dims = inpt.get_shape().as_list()[-1]
	if reuse:
		scale = tf.get_variable("scale")
		pop_var = tf.get_variable("popVariance")
	else:
		# scale = misc._variable('scale', dims, tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
		scale = misc._variable('scale', dims, tf.constant_initializer(1.0), trainable=True)
		pop_var = misc._variable('popVariance', dims, tf.constant_initializer(1.0), trainable=False)
	if useType=="train":
		batch_mean, batch_var = tf.nn.moments(inpt, axes=[i for i in range(len(inpt.get_shape().as_list())-1)])
		if not reuse:
			train_var = tf.assign(pop_var, pop_var * bn_decay + batch_var * (1 - bn_decay))
			with tf.control_dependencies([train_var]):
				return tf.nn.batch_normalization(inpt, 0, batch_var, 0, scale, bn_epsilon)
		else:
			return tf.nn.batch_normalization(inpt, 0, batch_var, 0, scale, bn_epsilon)
	else:
		return tf.nn.batch_normalization(inpt, 0, pop_var, 0, scale, bn_epsilon)

def batch_norm_only_rescale_learn_scale_cartesian(inpt, c_o, useType='test', reuse=False, bn_decay=default_bn_decay, bn_epsilon=default_bn_epsilon, mask=None):
	odd = [2*i for i in range(c_o)]
	even = [2*i+1 for i in range(c_o)]
	x = tf.gather(inpt, odd, axis=-1)
	y = tf.gather(inpt, even, axis=-1)
	magnitude = tf.sqrt(tf.square(x)+tf.square(y))
	# print(magnitude.get_shape())
	if mask != None:
		magnitude = tf.where(mask==1, mask, tf.ones_like(mask)*(-1))*magnitude
		# print(magnitude.get_shape())
	dims = magnitude.get_shape().as_list()[-1]
	if reuse:
		scale = tf.get_variable("scale")
		pop_var = tf.get_variable("popVariance")
	else:
		# scale = misc._variable('scale', dims, tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
		scale = misc._variable('scale', dims, tf.constant_initializer(1.0), trainable=True)
		pop_var = misc._variable('popVariance', dims, tf.constant_initializer(1.0), trainable=False)
	if useType=="train":
		batch_mean, batch_var = tf.nn.moments(magnitude, axes=[i for i in range(len(magnitude.get_shape().as_list())-1)])
		if not reuse:
			train_var = tf.assign(pop_var, pop_var * bn_decay + batch_var * (1 - bn_decay))
			with tf.control_dependencies([train_var]):
				x = tf.nn.batch_normalization(x, 0, batch_var, 0, scale, bn_epsilon)
				y = tf.nn.batch_normalization(y, 0, batch_var, 0, scale, bn_epsilon)
		else:
			x = tf.nn.batch_normalization(x, 0, batch_var, 0, scale, bn_epsilon)
			y = tf.nn.batch_normalization(y, 0, batch_var, 0, scale, bn_epsilon)
	else:
		x = tf.nn.batch_normalization(x, 0, pop_var, 0, scale, bn_epsilon)
		y = tf.nn.batch_normalization(y, 0, pop_var, 0, scale, bn_epsilon)
	if mask != None:
		x = x*mask
		y = y*mask
	xs = tf.unstack(x, axis=-1)
	ys = tf.unstack(y, axis=-1)
	res = [None]*(len(xs)+len(ys))
	res[::2] = xs
	res[1::2] = ys
	return tf.stack(res, axis=-1)

def batch_norm_cartesian(inpt, c_o, useType='test', reuse=False, bn_decay=default_bn_decay, bn_epsilon=default_bn_epsilon, mask=None):
	odd = [2*i for i in range(c_o)]
	even = [2*i+1 for i in range(c_o)]
	x = tf.gather(inpt, odd, axis=-1)
	y = tf.gather(inpt, even, axis=-1)
	magnitude = tf.sqrt(tf.square(x)+tf.square(y))
	x_mon, y_mon = x/magnitude, y/magnitude
	# print(magnitude.get_shape())
	if mask != None:
		magnitude = tf.where(mask==1, mask, tf.ones_like(mask)*(-1))*magnitude
		# print(magnitude.get_shape())
	dims = magnitude.get_shape().as_list()[-1]
	if reuse:
		scale = tf.get_variable("scale")
		beta = tf.get_variable("beta")
		pop_mean = tf.get_variable("popMean")
		pop_var = tf.get_variable("popVariance")
	else:
		# scale = misc._variable('scale', dims, tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
		scale = misc._variable('scale', dims, tf.constant_initializer(1.0), trainable=True)
		beta = misc._variable('beta', dims, tf.constant_initializer(0.0), trainable=True)
		pop_mean = misc._variable('popMean', dims, tf.constant_initializer(0.0), trainable=False)
		pop_var = misc._variable('popVariance', dims, tf.constant_initializer(1.0), trainable=False)
	if useType=="train":
		batch_mean, batch_var = tf.nn.moments(magnitude, axes=[i for i in range(len(magnitude.get_shape().as_list())-1)])
		if not reuse:
			train_var = tf.assign(pop_var, pop_var * bn_decay + batch_var * (1 - bn_decay))
			with tf.control_dependencies([train_var]):
				magnitude = tf.nn.batch_normalization(magnitude, batch_mean, batch_var, beta, scale, bn_epsilon)
		else:
			magnitude = tf.nn.batch_normalization(magnitude, batch_mean, batch_var, beta, scale, bn_epsilon)
	else:
		magnitude = tf.nn.batch_normalization(magnitude, pop_mean, pop_var, beta, scale, bn_epsilon)

	magnitude = tf.nn.relu(magnitude)
	x = x_mon*magnitude
	y = y_mon*magnitude
	xs = tf.unstack(x, axis=-1)
	ys = tf.unstack(y, axis=-1)
	res = [None]*(len(xs)+len(ys))
	res[::2] = xs
	res[1::2] = ys
	return tf.stack(res, axis=-1)

def conv(inpt, weights, biases, c_i, c_o, s_h, s_w, first=True, useType="test", padding="SAME", num_angles=4, num_bins=32, normalize="vn", count=False, weightMask=None, steerable=False):
	# if count:
	# 	with tf.device('/cpu:0'):
	# 		weights = monitorGrads(weights)
	# 	weightst = tf.transpose(weights, [3,0,1,2])
	# 	weightst = tf.reshape(weightst, [c_o,weights.get_shape()[0].value, c_i*2*weights.get_shape()[1].value])
	# 	norm = tf.norm(weightst, axis=[-2,-1], keepdims=True)
	# 	normE = tf.reshape(norm, [c_o,1,1,1])
	# 	normSep = tf.unstack(normE, axis=0)
	# 	with tf.device('/cpu:0'):
	# 		for ns in normSep:
	# 			tf.summary.scalar("weightNorm", tf.squeeze(ns))
	convRes, angle, convMask = cliffordConv(inpt, weights, c_i, c_o, s_h, s_w, padding="SAME", num_angles=num_angles, num_bins=num_bins, mask=True, count=False, weightMask=weightMask, steerable=steerable, first=first)
	floatMask = tf.cast(convMask, tf.float32)
	if padding == "VALID":
		print("In here")
		weightsShape = weights.get_shape().as_list()
		inptShape = inpt.get_shape().as_list()
		k_w = weightsShape[0]
		k_h = weightsShape[1]
		i_w = inptShape[1]
		i_h = inptShape[2]
		withBias = tf.image.crop_to_bounding_box(convRes, numpy.floor(k_w/2).astype(numpy.int32), numpy.floor(k_h/2).astype(numpy.int32), i_w-2*numpy.floor(k_w/2).astype(numpy.int32), i_h-2*numpy.floor(k_h/2).astype(numpy.int32))
		angle = tf.image.crop_to_bounding_box(angle, numpy.floor(k_w/2).astype(numpy.int32), numpy.floor(k_h/2).astype(numpy.int32), i_w-2*numpy.floor(k_w/2).astype(numpy.int32), i_h-2*numpy.floor(k_h/2).astype(numpy.int32))
	else:
		withBias = convRes

	# withBias = tf.multiply(tf.nn.bias_add(convRes, biases), floatMask)
	# conv_norm = tf.contrib.layers.group_norm(withBias, c_o, -1, (-3,-2)) * floatMask
	# conv_relu = tf.nn.leaky_relu(withBias, alpha=0.1)
	if normalize == "bn":
		# out = batch_norm(withBias, useType, reuse=(not first))
		out = batch_norm_only_rescale_learn_scale(withBias, useType, reuse=(not first))
	elif normalize == "vn":
		out = normalizeVectorField(withBias, weights.get_shape()[-4].value, weights.get_shape()[-3].value)
	else:
		# out = conv_norm
		out = withBias
	out = tf.nn.relu(out)
	reluMask = tf.where(out>0, tf.ones_like(out), tf.zeros_like(out))
	regulatedAngles = maskAngleGradients(angle, reluMask)
	# if count:
	# 	with tf.device('/cpu:0'):
	# 		size = 1
	# 		for shape_value in withBias.get_shape():
	# 			size *= shape_value
	# 		nonZeroOut = tf.count_nonzero(out)
	# 		negative = tf.count_nonzero(tf.less(withBias, 0.))
	# 		tf.summary.scalar("negative_withBias", negative/size.value)
	# 		tf.summary.scalar("nonZeroOut", nonZeroOut/size.value)
	return out, regulatedAngles

def normalizeVectorField(inpt, width, height):
	inptShape = inpt.get_shape().as_list()
	# inptR = tf.reshape(inpt, [inptShape[0].value, inptShape[-3].value, inptShape[-2].value*inptShape[-1].value])
	inptR = tf.reshape(inpt, inptShape[:2]+[numpy.prod(inptShape[2:])])
	# norm = tf.norm(inptR, axis=[-2,-1], keepdims=True)*(width*height/(inptShape[-3].value*inptShape[-2].value))**0.5
	# norm  = tf.reshape(norm, [inptShape[0].value, 1, 1, 1])
	norm = tf.norm(inptR, axis=[-2,-1], keepdims=True)
	# norm  = tf.reshape(norm, [inptShape[0], 1, 1, 1])
	norm  = tf.reshape(norm, [inptShape[0]]+[1 for i in range(len(inptShape)-1)])
	norm = tf.where(tf.equal(norm, 0), tf.ones_like(norm), norm)
	out = inpt*tf.sqrt(inptShape[1]*inptShape[2]/(width*height))/norm
	return out

def localyNormalizeVectorField(inpt, width, height):
	norm = tf.nn.avg_pool(inpt, [1,3,3,1], [1,1,1,1], "SAME")
	# inptShape = inpt.get_shape()
	# inptR = tf.reshape(inpt, [inptShape[0].value, inptShape[-3].value, inptShape[-2].value*inptShape[-1].value])
	# norm = tf.norm(inptR, axis=[-2,-1], keepdims=True)*(width*height/(inptShape[-3].value*inptShape[-2].value))**0.5
	# norm  = tf.reshape(norm, [inptShape[0].value, 1, 1, 1])
	# norm = tf.norm(inptR, axis=[-2,-1], keepdims=True)
	# norm  = tf.reshape(norm, [inptShape[0].value, 1, 1, 1])
	norm = tf.where(tf.equal(norm, 0), tf.ones_like(norm), norm)
	# out = inpt*tf.sqrt(inptShape[-3].value*inptShape[-2].value/(width*height))/norm
	out = inpt/norm
	return out

def calculateImageGradients(inpt):
	numpyWx = numpy.zeros([3,3], dtype=numpy.float32)
	numpyWx[0,0] = -1/12
	numpyWx[0,2] = 1/12
	numpyWx[2,2] = 1/12
	numpyWx[2,0] = -1/12
	numpyWx[1,0] = -1/6
	numpyWx[1,2] = 1/6
	numpyWy = numpy.zeros([3,3], dtype=numpy.float32)
	numpyWy[0,0] = -1/12
	numpyWy[0,2] = -1/12
	numpyWy[2,2] = 1/12
	numpyWy[2,0] = 1/12
	numpyWy[0,1] = -1/6
	numpyWy[2,1] = 1/6
	wx = tf.constant(numpyWx, dtype=tf.float32)
	wy = tf.constant(numpyWy, dtype=tf.float32)
	wz = tf.zeros_like(wx)
	if inpt.get_shape()[-1].value == 3:
		wx0 = tf.stack([wx, wz, wz], axis=-1)
		wx1 = tf.stack([wz, wx, wz], axis=-1)
		wx2 = tf.stack([wz, wz, wx], axis=-1)
		wy0 = tf.stack([wy, wz, wz], axis=-1)
		wy1 = tf.stack([wz, wy, wz], axis=-1)
		wy2 = tf.stack([wz, wz, wy], axis=-1)
		w = tf.stack([wy0, wx0, wy1, wx1, wy2, wx2], axis=-1)
	elif inpt.get_shape()[-1].value == 1:
		wx0 = tf.stack([wx], axis=-1)
		wy0 = tf.stack([wy], axis=-1)
		w = tf.stack([wy0, wx0], axis=-1)
	grads = tf.nn.conv2d(inpt, w, [1,1,1,1], "SAME")
	return grads

def l2_pooling(inpt, ks, ss):
	 return tf.sqrt(tf.nn.avg_pool(tf.square(inpt), [1,ks,ks,1],[1,ss,ss,1],"VALID"))

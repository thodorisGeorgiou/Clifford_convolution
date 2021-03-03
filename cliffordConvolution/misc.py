import numpy
import tensorflow as tf
from . import transformations

default_wd = 5e-4
default_nl = 1e-6
default_regularization = "nl"
default_regularization_mode = "op"
num_angles = 4

def getBasisArrays(s):
	c = s//2
	w = numpy.zeros([s,s,2], dtype=numpy.float32)
	for i in range(s):
		for j in range(s):
			x = i - c
			y = j - c
			r = numpy.sqrt(numpy.square(x) + numpy.square(y))
			th = numpy.arctan2(y,x)
			w[i,j,0] = r
			w[i,j,1] = th
	return numpy.split(w, 2, axis=-1)

# basisParameters = {5:[[0,1,2], {0:[0], 1:[0,1,2,3,4], 2:[0,1,2,3,4,5,6]}, {0:0.6, 1:0.6, 2:0.4}, 13], \
# 					7:[[0,1,2,3], {0:[0], 1:[0,1,2,3,4], 2:[0,1,2,3,4,5,6], 3:[0,1,2,3,4]}, {0:0.6, 1:0.6, 2:0.6, 3:0.4}, 18]}
basisParameters = {5:[[0,1,2], {0:[0], 1:[0,1,2], 2:[0,1,2]}, {0:0.6, 1:0.6, 2:0.4}, 7], \
					7:[[0,1,2,3], {0:[0], 1:[0,1,2], 2:[0,1,2,3], 3:[0,1,2]}, {0:0.6, 1:0.6, 2:0.6, 3:0.4}, 11], \
					9:[[0,1,2,3,4], {0:[0], 1:[0,1,2], 2:[0,1,2,3], 3:[0,1,2,3,4], 4:[0,1,2]}, {0:0.6, 1:0.6, 2:0.6, 3:0.6, 4:0.4}, 16]}

multipliers = { \
			16:{5:[1.0464623, 1.0412558, 1.0, 1.0412558, 1.0464623, 1.0412558, 1.0000001, 1.0412558, \
				  1.0464623, 1.0412558, 1.0000001, 1.0412558, 1.0464623, 1.0412558, 1.0, 1.0412558, 1.0464623],\
				4:[1.0000001, 1.1280547, 1.0, 1.1280547, 1.0000001, 1.1280547, 1.0000001, 1.1280547, 1.0000001, \
				1.1280547, 1.0000001, 1.1280547, 1.0000001, 1.1280547, 1.0, 1.1280547, 1.0000001],\
				7:[1.0000001, 1.0141906, 1.0, 1.0141907, 1.0000001, 1.0141906, 1.0000001, 1.0141906, 1.0000001, \
				 1.0141906, 1.0000001, 1.0141906, 1.0000001, 1.0141906, 1.0, 1.0141906, 1.0000001], \
				9:[1.0392774, 1.0398802, 1.0, 1.0398802, 1.0392774, 1.0398802, 1.0000001, 1.0398802, 1.0392774, \
				 1.0398802, 1.0000001, 1.0398802, 1.0392774, 1.0398802, 1.0, 1.0398802, 1.0392774]}, \
			32:{9:[1.0392774, 1.0296552, 1.0398802, 1.0212318, 1.0, 1.0212318, 1.0398802, 1.0296552, 1.0392774, \
				 1.0296552, 1.0398802, 1.0212317, 1.0000001, 1.0212318, 1.0398802, 1.0296552, 1.0392774, \
				 1.0296552, 1.0398802, 1.0212317, 1.0000001, 1.0212318, 1.0398802, 1.0296553, 1.0392774, \
				 1.0296553, 1.0398802, 1.0212318, 1.0, 1.0212318, 1.0398802, 1.0296553, 1.0392774], \
				7:[1.0000001, 1.0331969, 1.0141906, 1.038093, 1.0, 1.038093, 1.0141907, 1.0331969, 1.0000001, \
				 1.0331969, 1.0141906, 1.038093, 1.0000001, 1.038093, 1.0141906, 1.0331969, 1.0000001, \
				 1.0331969, 1.0141906, 1.038093, 1.0000001, 1.038093, 1.0141906, 1.0331969, 1.0000001, \
				 1.033197, 1.0141906, 1.038093, 1.0, 1.038093, 1.0141906, 1.033197, 1.0000001], \
				5:[1.0464623, 1.0392715, 1.0412558, 1.0776817, 1.0, 1.0776817, 1.0412558, 1.0392715, \
				  1.0464623, 1.0392715, 1.0412558, 1.0776815, 1.0000001, 1.0776817, 1.0412558, 1.0392715, \
				  1.0464623, 1.0392715, 1.0412558, 1.0776814, 1.0000001, 1.0776815, 1.0412558, 1.0392715, \
				  1.0464623, 1.0392715, 1.0412558, 1.0776815, 1.0, 1.0776817, 1.0412558, 1.0392715, 1.0464623],\
				4:[1.0000001, 1.0083287, 1.1280547, 1.0831803, 1.0, 1.0831802, 1.1280547, 1.0083287, \
				   1.0000001, 1.0083287, 1.1280547, 1.0831802, 1.0000001, 1.0831802, 1.1280547, 1.0083287, \
				   1.0000001, 1.0083287, 1.1280547, 1.0831802, 1.0000001, 1.0831803, 1.1280547, 1.0083287, \
				   1.0000001, 1.0083288, 1.1280547, 1.0831803, 1.0, 1.0831803, 1.1280547, 1.0083288, 1.0000001]},\
			64:{5:[1.0464623, 1.041508, 1.0392715, 1.042048, 1.0412558, 1.1028172, 1.0776817, 1.0432093, 1.0, \
				1.0432092, 1.0776817, 1.1028172, 1.0412558, 1.042048, 1.0392715, 1.0415081, 1.0464623, \
				1.041508, 1.0392715, 1.042048, 1.0412558, 1.1028172, 1.0776815, 1.0432093, 1.0000001, \
				1.0432092, 1.0776817, 1.1028172, 1.0412558, 1.042048, 1.0392715, 1.0415081, 1.0464623, \
				1.0415081, 1.0392715, 1.042048, 1.0412558, 1.1028172, 1.0776814, 1.0432093, 1.0000001, \
				1.0432092, 1.0776815, 1.1028172, 1.0412558, 1.0420481, 1.0392715, 1.041508, 1.0464623, \
				1.0415081, 1.0392715, 1.042048, 1.0412558, 1.102817, 1.0776815, 1.0432092, 1.0, 1.0432092, \
				1.0776817, 1.102817, 1.0412558, 1.0420481, 1.0392715, 1.0415081, 1.0464623],\
				4:[1.0000001, 1.0, 1.0083287, 1.0195115, 1.1280547, 1.1109005, 1.0831803, 1.0456862, 1.0, \
				1.0456862, 1.0831802, 1.1109005, 1.1280547, 1.0195115, 1.0083287, 1.0, 1.0000001, 1.0, \
				1.0083287, 1.0195115, 1.1280547, 1.1109005, 1.0831802, 1.0456862, 1.0000001, 1.0456862, \
				1.0831802, 1.1109005, 1.1280547, 1.0195115, 1.0083287, 1.0, 1.0000001, 1.0, 1.0083287, \
				1.0195113, 1.1280547, 1.1109005, 1.0831802, 1.0456862, 1.0000001, 1.0456862, 1.0831803, \
				1.1109005, 1.1280547, 1.0195115, 1.0083287, 1.0, 1.0000001, 1.0, 1.0083288, 1.0195115, \
				1.1280547, 1.1109005, 1.0831803, 1.0456861, 1.0, 1.0456861, 1.0831803, 1.1109005, 1.1280547, \
				1.0195115, 1.0083288, 1.0, 1.0000001]}\
			}

def _variable(name, shape, initializer, trainable=True):
	"""Helper to create a Variable stored on CPU memory.
	Args:
		name: name of the variable
		shape: list of ints
		initializer: initializer for Variable
	Returns:
		Variable Tensor
	"""
	if trainable:
		var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
	else:
		var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32, trainable=trainable)
	return var

def _variable_with_weight_decay(name, shape, stddev, wd=default_wd, mode=default_regularization_mode, cpu=False):
	"""Helper to create an initialized Variable with weight decay.

	Note that the Variable is initialized with a truncated normal distribution.
	A weight decay is added only if one is specified.

	Args:
		name: name of the variable
		shape: list of ints
		stddev: standard deviation of a truncated Gaussian
		wd: add L2Loss weight decay multiplied by this float. If None, weight
			decay is not added for this Variable.

	Returns:
		Variable Tensor
	"""
	if cpu:
		var = _variable_on_cpu(name, shape, \
			tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
	else:
		var = tf.get_variable(name, shape, \
			initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32), dtype=tf.float32)
	lr = tf.get_collection("learning_rate")
	if len(lr) != 1:
		exit("Something wrong with learning rate collection")
	if wd is not None:
		if mode=="loss":
			weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		elif mode=="op":
			regOp = tf.assign(var, var*(1-2*lr[0]*wd))
			tf.add_to_collection('regularizationOps', regOp)
			norm = tf.norm(var)
			tf.add_to_collection("norms", norm)
		else:
			exit("Invalid regularization mode!")
	return var

# def _variable_with_weight_reguralization(name, shape, stddev, wd, cpu=False):
# 	if cpu:
# 		var = _variable_on_cpu(name, shape, \
# 			tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
# 	else:
# 		var = tf.get_variable(name, shape, \
# 			initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32), dtype=tf.float32)
# 	if wd is not None:
# 		# weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
# 		tf.add_to_collection('weight_decay_ops', tf.assign(var, (1-wd)*var))
# 	return var

def _variable_with_norm_loss(name, shape, stddev, nl=default_nl, mode=default_regularization_mode, cpu=False):
	"""Helper to create an initialized Variable with norm loss.

	Note that the Variable is initialized with a truncated normal distribution.

	Args:
		name: name of the variable
		shape: list of ints
		stddev: standard deviation of a truncated Gaussian
		nl: add L2Loss weight decay multiplied by this float. If None, weight
			decay is not added for this Variable.

	Returns:
		Variable Tensor
	"""
	if cpu:
		var = _variable_on_cpu(name, shape, \
			tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
	else:
		var = tf.get_variable(name, shape, \
			initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32), dtype=tf.float32)
	lr = tf.get_collection("learning_rate")
	if len(lr) != 1:
		exit("Something wrong with learning rate collection")
	if nl is not None:
		weightst = tf.transpose(var, [3,0,1,2])
		weightst = tf.reshape(weightst, [shape[3], shape[0], shape[2]*shape[1]])
		norm = tf.norm(weightst, axis=[-2,-1])
		tf.add_to_collection("norms", norm)
		if mode=="loss":
			loss = tf.square(1-norm, name="weight_norm_loss")
			tf.add_to_collection('losses', tf.multiply(tf.reduce_sum(loss), nl))
		elif mode=="op":
			norm = tf.reshape(norm, [1,1,1,shape[-1]])
			correction = 2*nl*(1-tf.divide(1,norm))
			# correction = 4*nl*tf.divide(tf.pow(norm-1, 3), norm)
			regOp = tf.assign(var, var*(1-lr*correction))
			tf.add_to_collection('regularizationOps', regOp)
		else:
			exit("Invalid regularization mode!")
	return var

def getWeightsNBiases(first, k_h, k_w, c_o, c_i, s_h, s_w, wd=None, lossType=default_regularization, mode=default_regularization_mode, normalize=0):
	if wd == None:
		if lossType == "wd":
			wd = default_wd
		else:
			wd = default_nl
	if first:
		stddev=numpy.sqrt(2.0 / (c_i*k_h*k_w))
		if lossType=="wd":
			convW = _variable_with_weight_decay('weights', shape=[k_h, k_w, c_i, c_o], stddev=stddev, wd=wd, mode=mode)
		elif lossType=="nl":
			convW = _variable_with_norm_loss('weights', shape=[k_h, k_w, c_i, c_o], stddev=stddev, nl=wd, mode=mode)
		else:
			convW = _variable('weights', [k_h, k_w, c_i, c_o], tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
			normalizedWeights = normalizeWeights(convW)
			tf.add_to_collection("weight_normalization_ops", normalizedWeights)
		convb = _variable('biases', [c_o], tf.constant_initializer(0.0))
	else:
		convW = tf.get_variable("weights")
		convb = tf.get_variable("biases")
	if normalize==0:
		return convW, convb
	elif normalize==1:
		return normalizeWeights(convW, full=False), convb
	elif normalize==2:
		return normalizeWeights(convW, full=True), convb

# def getSteerableBasis(r, th, k, m, sigma, fi):
# 	t = tf.exp(-tf.square(r-m)/(2*tf.square(sigma)))
# 	x = tf.cos(th-fi)*t
# 	y = tf.sin(th-fi)*t
# 	# basis = tf.concat([x,y], axis=-1)
# 	return x, y



def getScalarSteerableBasis(r, th, k, m, sigma, fi):
	fis = numpy.ones(th.shape, dtype=numpy.float32)*fi
	fis[numpy.where(r==0)] = 0
	t = numpy.exp(-numpy.square(r-m)/(2*numpy.square(sigma)))
	x = numpy.cos((th-fis)*k)*t
	y = -numpy.sin((th-fis)*k)*t
	return x, y

def getSteerableBasis(r, th, k, m, sigma, fi):
	t = numpy.exp(-numpy.square(r-m)/(2*numpy.square(sigma)))
	x = numpy.cos((th-fi)*k)*t
	y = numpy.sin((th-fi)*k)*t
	basis = numpy.concatenate([x,y], axis=-1)
	return basis

def getSteerableGradientBasis(r, th, k, m, sigma, fi):
	t = numpy.exp(-numpy.square(r-m)/(2*numpy.square(sigma)))
	x = numpy.sin((th-fi)*k)*t*k
	y = numpy.cos((th-fi)*k)*t*(-k)
	basis = numpy.concatenate([x,y], axis=-1)
	return basis

def getSteerableKernel(s, fi, c_i, c_o, first=False):
	npr, npth = getBasisArrays(s)
	# r = tf.constant(npr, dtype=tf.float32)
	# th = tf.constant(npth, dtype=tf.float32)
	# basisX = []
	# basisY = []
	params = basisParameters[s]
	basisXY = []
	for j in params[0]:
		for k in params[1][j]:
			basisXY.append(getSteerableBasis(npr, npth, k, j, params[2][j], fi))
			# x, y = getSteerableBasis(npr, npth, k, j, params[2][j], fi)
			# basisX.append(x)
			# basisY.append(y)
	npBasisXY = numpy.stack(basisXY, axis=-1)
	npBasisXY = numpy.expand_dims(numpy.tile(npBasisXY, [1,1,c_i,1]), axis=-1).astype(numpy.float32)
	basisXY = tf.constant(npBasisXY, name="basis_"+str(fi[0]))
	weights = tf.get_variable("weights", [1,1, 2*c_i, basisXY.get_shape().as_list()[-2], c_o], initializer=tf.truncated_normal_initializer(stddev=numpy.sqrt(2/(c_i*basisParameters[s][-1])), dtype=tf.float32), dtype=tf.float32)
	fltr = tf.reduce_sum(tf.multiply(basisXY, weights), axis=-2)
	steerableFilters = transformations.rotateVectors(fltr, fi)
	if first:
		tf.add_to_collection("weights", weights)
	# if fi == 0.0:
	# 	with tf.device('/cpu:0'):
	# 		norm = tf.norm(steerableFilters)/c_o
	# 		tf.summary.scalar("norm_"+str(fi), norm)
	return steerableFilters


def getScalarBasis(s, num_bins, c_i):
	npr, npth = getBasisArrays(s)
	params = basisParameters[s]
	fis = [a*2*numpy.pi/num_bins for a in range(num_bins)]
	perAngleBasis = []
	for fi in fis:
		basis = []
		for j in params[0]:
			for k in params[1][j]:
				x, y = getScalarSteerableBasis(npr, npth, k, j, params[2][j], fi)
				basis.append(x)
				basis.append(y)
		npBasis = numpy.stack(basis, axis=-1)
		perAngleBasis.append(numpy.expand_dims(numpy.tile(npBasis, [1,1,c_i,1]), axis=-1))
	npBasis = numpy.concatenate(perAngleBasis, axis=2)
	# npBasis = numpy.stack(perAngleBasis, axis=-4)
	basis = tf.constant(npBasis, name="basis")
	return basis

def getScalarSteerableKernel(s, fi, abin, c_i, c_o, num_bins=16, first=False, basis=None, returnAngles=False):
	if num_bins == 1:
		npr, npth = getBasisArrays(s)
		params = basisParameters[s]
		basis = []
		for j in params[0]:
			for k in params[1][j]:
				x, y = getScalarSteerableBasis(npr, npth, k, j, params[2][j], fi)
				basis.append(x)
				basis.append(y)
		npBasis = numpy.stack(basis, axis=-1)
		npBasis = numpy.expand_dims(numpy.tile(npBasis, [1,1,c_i,1]), axis=-1)
		basis = tf.constant(npBasis, name="basis_"+str(fi))
		weights = _variable_with_weight_decay('weights', shape=[1, 1, c_i, basis.get_shape().as_list()[-2], c_o], stddev=numpy.sqrt(2/(c_i*basisParameters[s][-1])), wd=1e-7, mode="loss")
		# weights = tf.get_variable("weights", [1, 1, c_i, basis.get_shape().as_list()[-2], c_o], initializer=tf.truncated_normal_initializer(stddev=numpy.sqrt(2/(c_i*basisParameters[s][-1])), dtype=tf.float32), dtype=tf.float32)
		fltr = tf.reduce_sum(tf.multiply(basis, weights), axis=-2)
		# steerableFilters = tf.contrib.image.rotate(fltr, fi, interpolation="BILINEAR")
	else:
		weights = _variable_with_weight_decay('weights', shape=[1, 1, basis.get_shape().as_list()[-2], c_o, num_bins, c_i], stddev=numpy.sqrt(2/(c_i*basisParameters[s][-1])), wd=1e-7, mode="loss")
		# weights = tf.get_variable("weights", [1, 1, basis.get_shape().as_list()[-2], c_o, num_bins, c_i], initializer=tf.truncated_normal_initializer(stddev=numpy.sqrt(2/(c_i*basisParameters[s][-1])), dtype=tf.float32), dtype=tf.float32)
		inds = [i for i in range(abin,-1,-1)]+[i for i in range(num_bins-1,abin,-1)]
		# inds = [i for i in range(num_bins-abin,num_bins)]+[i for i in range(num_bins-abin)]
		# inds = [i for i in range(abin,num_bins)]+[i for i in range(abin)]
		inds = tf.constant(inds)
		rWeights = tf.gather(weights, inds, axis=-2)
		rWeights = tf.reshape(rWeights, [1, 1, basis.get_shape().as_list()[-2], c_o, c_i * num_bins])
		rWeights = tf.transpose(rWeights, [0, 1, 4, 2, 3])
		fltr = tf.reduce_sum(tf.multiply(basis, rWeights), axis=-2)
		if returnAngles:
			angles = tf.cast(inds, tf.float32)*2*numpy.pi/num_bins
			fltr = tf.reshape(fltr, [s, s, num_bins, c_i, c_o])
			fltr = tf.unstack(fltr, axis=-3)
			fltr = [fltr, angles]
		else:
			rWeights = tf.gather(weights, inds, axis=-2)
			rWeights = tf.reshape(rWeights, [1, 1, basis.get_shape().as_list()[-2], c_o, c_i * num_bins])
			rWeights = tf.transpose(rWeights, [0, 1, 4, 2, 3])
			fltr = tf.reduce_sum(tf.multiply(basis, rWeights), axis=-2)
	if first:
		tf.add_to_collection("weights", weights)
	# if fi == 0.0:
	# 	with tf.device('/cpu:0'):
	# 		norm = tf.norm(steerableFilters)/c_o
	# 		tf.summary.scalar("norm_"+str(fi), norm)
	# basisX = tf.constant(npBasisX)
	# basisY = tf.constant(npBasisY)
	# npBasisX = numpy.concatenate(basisX, axis=-1)
	# npBasisY = numpy.concatenate(basisY, axis=-1)
	# npBasisX = numpy.tile(numpy.expand_dims(npBasisX, axis=-1), [1,1,1,c_i])
	# npBasisY = numpy.tile(numpy.expand_dims(npBasisY, axis=-1), [1,1,1,c_i])
	# cosFi = numpy.cos(fi)
	# sinFi = numpy.sin(fi)
	# steerableFilters = []
	# for f in range(c_o):
	# 	weights_x = tf.get_variable("weights_x_"+str(f), [1,1, basisX.get_shape().as_list()[-2], c_i], initializer=tf.truncated_normal_initializer(stddev=numpy.sqrt(2/(c_i*basisParameters[s][-1])), dtype=tf.float32), dtype=tf.float32)
	# 	weights_y = tf.get_variable("weights_y_"+str(f), [1,1, basisY.get_shape().as_list()[-2], c_i], initializer=tf.truncated_normal_initializer(stddev=numpy.sqrt(2/(c_i*basisParameters[s][-1])), dtype=tf.float32), dtype=tf.float32)
	# 	filter_x = tf.reduce_sum(tf.multiply(basisX, weights_x), axis=-2)
	# 	filter_y = tf.reduce_sum(tf.multiply(basisY, weights_y), axis=-2)
	# 	turned_x = filter_x*cosFi-filter_y*sinFi
	# 	turned_y = filter_x*sinFi+filter_y*cosFi
	# 	xs = tf.unstack(turned_x, axis=-1)
	# 	ys = tf.unstack(turned_y, axis=-1)
	# 	res = [None]*(len(xs)+len(ys))
	# 	res[::2] = xs
	# 	res[1::2] = ys
	# 	steerableFilters.append(tf.stack(res, axis=-1))
	# steerableFilters = tf.stack(steerableFilters, axis=-1)
	return fltr

def getSteerableGradientKernel(s, fis, c_i, c_o, weights, weightMask):
	kernels = []
	for fi in fis:
		npr, npth = getBasisArrays(s)
		params = basisParameters[s]
		basisXY = []
		for j in params[0]:
			for k in params[1][j]:
				basisXY.append(getSteerableGradientBasis(npr, npth, k, j, params[2][j], fi))
				# x, y = getSteerableBasis(npr, npth, k, j, params[2][j], fi)
				# basisX.append(x)
				# basisY.append(y)
		npBasisXY = numpy.stack(basisXY, axis=-1)
		npBasisXY = numpy.expand_dims(numpy.tile(npBasisXY, [1,1,c_i,1]), axis=-1).astype(numpy.float32)
		basisXY = tf.constant(npBasisXY, name="basis_"+str(fi[0]))
		fltr = tf.reduce_sum(tf.multiply(basisXY, weights), axis=-2)
		steerableGradientFilters = transformations.rotateVectors(fltr, fi)
		if weightMask != None:
			kernels.append(steerableGradientFilters*weightMask)
		else:
			kernels.append(steerableGradientFilters)
	return tf.stack(kernels, axis=0)

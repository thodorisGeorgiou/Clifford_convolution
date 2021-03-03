import tensorflow as tf

def rotateVectors(vectors, theta):
	vectors = tf.transpose(vectors, [0,3,1,2])
	shape = vectors.get_shape()
	vectors = tf.reshape(vectors, [shape[0].value*shape[3].value//2*shape[1].value, shape[2].value, 2])
	# vectors = tf.reshape(vectors, [shape[0].value, shape[1].value, shape[2].value, shape[3].value//2, 2])
	rotation_matrix = tf.stack([tf.cos(theta), tf.sin(theta), -tf.sin(theta), tf.cos(theta)])
	rotation_matrix = tf.reshape(rotation_matrix, (2,2))
	rotation_matrix = tf.tile(tf.expand_dims(rotation_matrix, axis=0), [vectors.get_shape()[0],1,1])
	# rotation_matrix = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(rotation_matrix, axis=0), axis=0), axis=0), [vectors.get_shape()[0],vectors.get_shape()[1],vectors.get_shape()[2],1,1])
	rotated = tf.reshape(tf.matmul(vectors, rotation_matrix), shape)
	return tf.transpose(rotated, [0,2,3,1])

# def rotateVectors(vectors, theta):
# 	vectors = tf.transpose(vectors, [0,1,3,2])
# 	shape = vectors.get_shape()
# 	vectors = tf.reshape(vectors, [shape[0].value, shape[1].value, shape[2].value, shape[3].value//2, 2])
# 	rotation_matrix_x = tf.stack([tf.cos(theta), -tf.sin(theta)])
# 	rotation_matrix_y = tf.stack([tf.sin(theta), tf.cos(theta)])
# 	rotation_matrix = tf.concat([rotation_matrix_x, rotation_matrix_y], axis=1)
# 	rotation_matrix = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(rotation_matrix, axis=0), axis=0), axis=0), [shape[0].value, shape[1].value, shape[2].value, 1, 1])
# 	rotated = tf.matmul(vectors, rotation_matrix)
# 	rotated = tf.reshape(rotated, shape)
# 	return tf.transpose(rotated, [0,1,3,2])

def rotateVectorFieldMinus(field, angle, irelevantAxisFirst=False):
	if irelevantAxisFirst:
		field = tf.transpose(field, [1,2,3,0])
	inPlaceRotated = rotateVectors(field, angle)
	rotated = tf.contrib.image.rotate(tf.transpose(inPlaceRotated, [3,0,1,2]), -angle, interpolation="BILINEAR")
	if irelevantAxisFirst:
		return rotated
	return tf.transpose(rotated, [1,2,3,0])

def rotateVectorField(field, angle, irelevantAxisFirst=False, returnIrelevantAxisFirst=False):
	if irelevantAxisFirst:
		field = tf.transpose(field, [1,2,3,0])
	inPlaceRotated = rotateVectors(field, angle)
	rotated = tf.contrib.image.rotate(tf.transpose(inPlaceRotated, [3,0,1,2]), angle, interpolation="BILINEAR")
	if irelevantAxisFirst or returnIrelevantAxisFirst:
		return rotated
	return tf.transpose(rotated, [1,2,3,0])

def changeToCartesian(magnitude, angles, count=False):
	x = tf.multiply(magnitude, tf.cos(angles))
	y = tf.multiply(magnitude, tf.sin(angles))
	if count:
		with tf.device('/cpu:0'):
			size = 1
			for shape_value in x.get_shape():
				size *= shape_value
			nonZero_x = tf.count_nonzero(x)
			nonZero_y = tf.count_nonzero(y)
			nonZero_magnitude = tf.count_nonzero(magnitude)
			tf.summary.scalar("nonZero_x", nonZero_x/size.value)		
			tf.summary.scalar("nonZero_y", nonZero_y/size.value)		
			tf.summary.scalar("nonZero_magnitude", nonZero_magnitude/size.value)		
	xs = tf.unstack(x, axis=-1)
	ys = tf.unstack(y, axis=-1)
	res = [None]*(len(xs)+len(ys))
	res[::2] = xs
	res[1::2] = ys
	return tf.stack(res, axis=-1)

def changeToPolar(vectors, c_o, count=False):
	odd = [2*i for i in range(c_o)]
	even = [2*i+1 for i in range(c_o)]
	x = tf.gather(vectors, odd, axis=-1)
	y = tf.gather(vectors, even, axis=-1)
	with tf.get_default_graph().gradient_override_map({"Identity": "replaceNaNs"}):
		x = tf.identity(x)
		y = tf.identity(y)
	magnitude = tf.sqrt(x**2 + y**2)
	angle = tf.atan2(y,x)
	return magnitude, angle

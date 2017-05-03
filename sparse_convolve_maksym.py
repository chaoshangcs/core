import tensorflow as tf 
import numpy as np

class Net():
	def __init__(self, kernel, bias, strides):
		self.kernel = kernel
		self.bias = bias
		self.strides = strides
	def sparse_convolve(self, x_image):
		"""
		x_image: (SparseTensor) represents image 
		"""
		#Generate distance matrix
		x_image_coords = x_image.indices 
		number_of_atoms = tf.shape(x_image_coords)[0] #converts to python int of number of atoms
		x_image_coords_copy = x_image.indices
		#Create distance matrix
		transpose_coords = tf.transpose(tf.expand_dims(x_image_coords_copy, 0), perm=[1, 0, 2])
		distance_matrix = tf.sqrt(tf.cast(tf.reduce_sum(tf.square(x_image_coords-transpose_coords), reduction_indices=[2]), tf.float32)) #[tf.reduce_sum(tf.square(x_image_coords-transpose_coords), reduction_indices=[2])**0.5]
		rounded_distance_matrix = tf.ceil(distance_matrix) #take ceiling of all values
		#Mask distance matrix based on a threshold distance
		threshold_distances = tf.fill(tf.shape(distance_matrix), 10.0) #creates matrix of same shape as distance matrix
		masked_distance_matrix = tf.less_equal(rounded_distance_matrix, threshold_distances)
		flattened_masked_distance_matrix = tf.reshape(masked_distance_matrix, [-1])
		#Add atom index and tile indices list
		atm_labels = tf.expand_dims(tf.range(1, number_of_atoms+1), -1)
		copy_atm_labels = tf.reshape(tf.tile(atm_labels, [1, number_of_atoms]), [number_of_atoms*number_of_atoms, 1])
		tile_image = tf.tile(tf.cast(x_image_coords, tf.int32), [number_of_atoms, 1])
		tiled_image_with_atm = tf.concat([tile_image, copy_atm_labels], 1)
		#Apply binary mask
		image_after_binary_mask = tf.boolean_mask(tiled_image_with_atm, flattened_masked_distance_matrix)
		#FIGURE OUT HOW TO CONVERT INTO DENSE REPRESENTATION

		#Convolve
		return tf.nn.conv3d(full_image, self.kernel, self.strides, "VALID")


#Test
testOneInput = tf.SparseTensor([[1, 1, 1], [2, 2, 2], [10, 10, 10], [11, 11, 11]], [1, 1, 1, 1], [20, 20, 20])
testOneKernel = tf.Variable(tf.constant([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
testOneBias = tf.Variable(tf.constant([0]))
testOneStrides = [1, 1, 1, 1, 1]
testNet = Net(testOneKernel, testOneBias, testOneStrides)
with tf.Session() as sess:
	print(sess.run(testNet.sparse_convolve(testOneInput)))
	sparse = tf.SparseTensor(idx, tf.gather_nd(a_t, idx), a_t.get_shape())
	dense = tf.sparse_tensor_to_dense(sparse)
	b = sess.run(dense)
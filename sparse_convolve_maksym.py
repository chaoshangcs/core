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
		x_image_values = x_image.values
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
		#Tile the coordinate values
		#tiled_image_vals = tf.reshape(tf.tile(tf.expand_dims(x_image_values, -1), [1, number_of_atoms]), [-1]) 
		tiled_image_vals = tf.tile(x_image_values, [number_of_atoms])
		print(sess.run(tiled_image_vals))
		#Tile the coord_copy for centerings
		tiled_image_copy = tf.reshape(tf.tile(x_image_coords_copy, [1, number_of_atoms]), [number_of_atoms*number_of_atoms, 3]) 
		#Tile coordinates and add their set number
		atm_labels = tf.expand_dims(tf.range(1, number_of_atoms+1), -1)
		copy_atm_labels = tf.cast(tf.reshape(tf.tile(atm_labels, [1, number_of_atoms]), [number_of_atoms*number_of_atoms, 1]), tf.int64)
		tile_image = tf.tile(tf.cast(x_image_coords, tf.int32), [number_of_atoms, 1])
		#Center all of the items in their respective set
		tile_image = tf.subtract(tf.cast(tile_image, tf.int64), tf.cast(tiled_image_copy, tf.int64))
		center_coords = tf.cast(tf.tile(tf.constant([[10, 10, 10]]), [number_of_atoms*number_of_atoms, 1]), tf.int64)
		tile_image = tf.add(tile_image, center_coords)		
		#Concatenate atom set labels onto image coords
		tiled_image_with_atm = tf.concat([tile_image, copy_atm_labels], 1)
		#Apply binary mask to coord and vals
		image_after_binary_mask = tf.boolean_mask(tiled_image_with_atm, flattened_masked_distance_matrix)
		image_after_binary_mask = tf.cast(image_after_binary_mask, tf.int64)
		#print(sess.run(image_after_binary_mask))
		image_vals_after_binary_mask = tf.cast(tf.boolean_mask(tiled_image_vals, flattened_masked_distance_matrix), tf.int64)
		#print(sess.run(image_vals_after_binary_mask))
		new_sparse_tensor = tf.SparseTensor(image_after_binary_mask, image_vals_after_binary_mask, [20, 20, 20, tf.cast(number_of_atoms+1, tf.int64)]) #20x20x20 is size of kernel
		new_dense = tf.sparse_tensor_to_dense(tf.sparse_reorder(new_sparse_tensor))
		#Reshape 4D into 3D		
		new_dense_3d = tf.expand_dims(tf.expand_dims(tf.reshape(new_dense, [20, 20, 100]), 0), -1)
		return tf.nn.conv3d(tf.cast(new_dense_3d, tf.float32), self.kernel, self.strides, "VALID")


#Test
testOneInput = tf.SparseTensor([[1, 1, 1], [2, 2, 2], [10, 10, 10], [11, 11, 11]], [1, 3, 5, 10], [20, 20, 20])
k = tf.ones([20, 20, 20, 1, 10], tf.float32)
testOneKernel = tf.Variable(k)
#testOneKernel = tf.Variable(tf.constant([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
testOneBias = tf.Variable(tf.constant([0]))
testOneStrides = [1, 1, 1, 1, 1]
testNet = Net(testOneKernel, testOneBias, testOneStrides)
with tf.Session() as sess:
	sess.run(testOneKernel.initializer)
	print(sess.run(testNet.sparse_convolve(testOneInput)))
	
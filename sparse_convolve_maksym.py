import tensorflow as tf 
import numpy as np

class Net():
	def __init__(self):
		self.kernel = 
		self.bias = 
		self.strides = 
	def sparse_convolve(self, x_image):
		"""
		x_image: (SparseTensor) represents image 
		"""
		#Generate distance matrix
		x_image_coords = x_image.indices 
		number_of_atoms = x_image_coords.get_shape().as_list()[0] #converts to python int of number of atoms
		x_image_coords_copy = x_image.indices
		#Create distance matrix
		transpose_coords = tf.transpose(tf.expand_dims(x_image_coords_copy, 0), perm=[1, 0, 2])
		distance_matrix = [tf.reduce_sum(tf.square(x_image_coords-transpose_coords), reduction_indices=[2])**0.5]
		rounded_distance_matrix = tf.ceil(distance_matrix) #take ceiling of all values
		#Mask distance matrix based on a threshold distance
		threshold_distances = tf.fill(distance_matrix.get_shape(), 10) #creates matrix of same shape as distance matrix
		masked_distance_matrix = tf.less_equal(rounded_distance_matrix, threshold_distances)
		#Find neighbors for each atom 	
		neighbors_list = [row for row in tf.split(masked_distance_matrix, num_or_size_splits=number_of_atoms, axis=0)]
		#Generate dense representation of neighbors for each atom
		dense_tensor_representations = []
		for atom_neighbors in neighbors_list: 		
			neighbors_indices = tf.where(atom_neighbors) #then pull coordinates of elements that are within threshold (tf.where)
			dense_representation = tf.sparse_to_dense(neighbors_indices, KERNEL_SIZE, tf.fill(neighbors_indices.get_shape(), 1)) #generate cube based on kernel size 
			dense_tensor_representations.append(dense_representation)
		full_image = tf.stack(dense_tensor_representations)
		#Convolve
		return tf.nn.conv3d(full_image, self.kernel, self.strides, "VALID")



# default_sparse_values = tf.fill(neighbors_indices.get_shape(), 1) #initialize default values 			
# 			sparse_representation = tf.SparseTensor(atom_neighbors, default_sparse_values, KERNEL_SIZE) #put into indices of sparse tensor. say default value of 1 for all--but this can be changed


#1. Take neighbors
#2. Round distances(ceil)
#3. Generate images where n atoms on x axis, sparse neighbors
split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)
tf.shape(split0) ==> [5, 10]
#4. Sparse to dense for each atoms 
#5. n atoms (for convolutions)
#6. conv3d 

a = tf.constant([[1.0,2,3],[4,5,6]])
b = tf.constant([[1.0,2,2],[3,5,7]])
sess = tf.Session()




c = tf.transpose(tf.expand_dims(b,0),perm=[1,0,2])
print sess.run([tf.shape(a),tf.shape(c)])

print sess.run([tf.reduce_sum(tf.square(a-c), reduction_indices=[2])**0.5])


a = np.reshape(np.arange(24), (3, 4, 2))
with tf.Session() as sess:
    a_t = tf.constant(a)
    idx = tf.where(tf.not_equal(a_t, 0))
    # Use tf.shape(a_t, out_type=tf.int64) instead of a_t.get_shape() if tensor shape is dynamic
    sparse = tf.SparseTensor(idx, tf.gather_nd(a_t, idx), a_t.get_shape())
    dense = tf.sparse_tensor_to_dense(sparse)
    b = sess.run(dense)
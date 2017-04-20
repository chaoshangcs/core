import tensorflow as tf
import numpy as np

#---------------------------------HYPERPARAMETERS---------------------------------#

#atom type embedding size
d_atm = 200
#amino acid embedding size
d_amino = 200
#charge embedding size
d_chrg = 200
#distance embedding size
d_dist = 200
#number convolutional filters
cf = 400
#number hidden units
h = 50
#learning rate
l = 0.075
#number of neighbor atoms from ligand
k_c = 6
#number of neighbor atoms from protein
k_p = 0
#number of atom types
ATOM_TYPES = 7
#number of distance bins
DIST_BINS = 18
DIST_INTERVAL = 0.3

#-------------------------------LAYER CONSTRUCTION--------------------------------#

# telling tensorflow how we want to randomly initialize weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.005)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var, name):
    """attaches a lot of summaries to a tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)

def embed_layer(layer_name, input_tensor):
	"""performs feature embedding on the input tensor, 
	transforming numbers to 200D vectors.
	Input: tensor of shape k_c * 2 * m
	Output: tensor of shape [k_c, d_dist + d_atm, m, 1]
	"""
	with tf.name_scope(layer_name):
		with tf.name_scope('atom_weights'):
			W_atom = weight_variable([ATOM_TYPES, d_atm])
		with tf.name_scope('dist_weights'):
			W_dist = weight_variable([DIST_BINS, d_dist])

		h_embed = tf.nn.embedding_lookup([W_atom, W_dist], input_tensor, name='embed_layer')
		h_embed = tf.transpose(h_embed, perm=[0, 1, 3, 2])
		h_embed = tf.reshape(h_embed, [1, k_c, d_atm+d_dist, -1, 1])

	print layer_name, "output dimensions:", h_embed.get_shape()
	return h_embed


def conv_layer(layer_name, input_tensor, filter_size, strides=[1,1,1,1,1], padding='SAME'):
	"""makes a simple face convolutional layer"""
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			W_conv = weight_variable(filter_size)
			variable_summaries(W_conv, layer_name + '/weights')
		with tf.name_scope('biases'):
			b_conv = bias_variable([filter_size[3]])
			variable_summaries(b_conv, layer_name + '/biases')
		h_conv = tf.nn.conv3d(input_tensor, W_conv, strides=strides, padding=padding) + b_conv
		tf.summary.histogram(layer_name + '/pooling_output', h_conv)
	print layer_name,"output dimensions:", h_conv.get_shape()
	return h_conv
	

def fc_layer(layer_name,input_tensor,output_dim):
	"""makes a simple fully connected layer"""
	input_dim = int((input_tensor.get_shape())[1])

	with tf.name_scope(layer_name):
		weights = weight_variable([input_dim, output_dim])
		variable_summaries(weights, layer_name + '/weights')
	with tf.name_scope('biases'):
		biases = bias_variable([output_dim])
		variable_summaries(biases, layer_name + '/biases')
	with tf.name_scope('Wx_plus_b'):
		h_fc = tf.matmul(input_tensor, weights) + biases
		tf.summary.histogram(layer_name + '/fc_output', h_fc)
	print layer_name, "output dimensions:", h_fc.get_shape()
	return h_fc

#-----------------------------------NETWORK----------------------------------------#

class Z(object):

	def __init__(self, ligand_atoms, ligand_coords):
		""" Takes input ligand_atoms, ligand_coords in the form of lists. Transforms ligand_coords into 
		a dictionary and constructs z, a [m * kc * 2] matrix. """
		self.ligand_atoms = ligand_atoms
		ligand_coords_dict = {}
		for i in range(len(ligand_atoms)):
			ligand_coords_dict[i] = ligand_coords[i]
		self.ligand_coords = ligand_coords_dict
		self.z = self.build_z()

	def build_z(self):
		#returns matrix of dimensions [m * kc * 2]
		raw_z = self.get_raw_z()
		for atom_index in range(len(raw_z)): #iterate over atoms dimension (m)
			for neighbor_index in range(len(raw_z[0])): #iterate over neighbors (kc)
				atom, distance = raw_z[atom_index][neighbor_index]
				raw_z[atom_index][neighbor_index][0] = self.ligand_atoms[raw_z[atom_index][neighbor_index][0]]
				raw_z[atom_index][neighbor_index][1] = int(distance//DIST_INTERVAL + 1)
		return raw_z

	def get_raw_z(self):
		#returns matrix of dimensions [m * kc * 2]
		raw_z = []
		for atom_index in range(len(self.ligand_atoms)):
			kc_neighbors_dict = self.get_closest_atoms_and_distances(atom_index)
			kc_neighbors_list = [[neighbor, distance] for neighbor, distance in kc_neighbors_dict.items()]
			kc_neighbors_list.sort(key=lambda x: x[1]) #sort by distance
			raw_z.append(kc_neighbors_list) 
		return raw_z

	def get_closest_atoms_and_distances(self, atom_index):
		distances = self.convert_coords_to_distances(atom_index)
		closest = {}
		for _ in range(k_c):
			closest_atom = min(distances, key=distances.get)
			closest[closest_atom] = distances[closest_atom]
			del distances[closest_atom]
		return closest

	def convert_coords_to_distances(self, atom_index):
		ligand_distances = {}
		atom_coord = self.ligand_coords[atom_index]
		for neighbor in self.ligand_coords:
			ligand_distances[neighbor] = self.distance(self.ligand_coords[neighbor], atom_coord)
		return ligand_distances

	def distance(self, coord1, coord2):
		x1, y1, z1 = coord1
		x2, y2, z2 = coord2
		return ((x1-x2) ** 2 + (y1-y2) ** 2 + (z1-z2) ** 2) ** 0.5


def construct_z(ligand_atoms, ligand_coords):
	""" Returns the z, that has not done feature embedding. Returns array. """
	return Z(ligand_atoms, ligand_coords).z


def deepVS_net(input_tensor, keep_prob):
	"""input_tensor is a tensor of shape [k_c * 2 * m]"""
	#do the feature embedding to get a tensor with shape [k_c, d_dist+d_atm, m, 1]
	z_embed = embed_layer('embedding', input_tensor)
	#convolutional layer - padding = 'VALID' prevents 0 padding
	z_conv = conv_layer('face_conv', input_tensor=z_embed, filter_size=[k_c, d_atm+d_dist, 1, 1, cf], padding='VALID')
	#max pool along the columns (corresponding to each convolutional filter)
	z_pool = tf.reduce_max(z_conv, axis=[3], keep_dims=True)
	#pool gives us batch*1*1*1*cf tensor; flatten it to get a tensor of length cf
	#NOTE: THIS IS ACTUALLY 2D in batch!
	z_flattened = tf.reshape(z_pool, [-1, cf])
	#fully connected layer
	z_fc1 = fc_layer(layer_name='fc1', input_tensor=z_flattened, output_dim=h)
	#dropout
	#output layer
	z_output = fc_layer(layer_name='out_neuron', input_tensor=z_fc1, output_dim=2)
	return z_output
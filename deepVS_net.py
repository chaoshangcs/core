import tensorflow as tf

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

def preprocess_layer(layer_name, atoms, coords):
	#get a matrix of distances between atoms (tensor with shape [m, m])
	coords_copy = tf.transpose(tf.expand_dims(coords, 0), perm=[1,0,2])
	distances = tf.reduce_sum(tf.square(coords - coords_copy), reduction_indices=[2]) ** 0.5
	
	#get the closest neighbors using tf.nn.top_k, and the corresponding atoms
	neg_distances = tf.multiply(distances, tf.constant(-1.0))
	neighbor_neg_distances, neighbor_atom_indices = tf.nn.top_k(neg_distances, k_c)
	#binning the distances and associating atom indices with actual atom types
	neighbor_distances = tf.multiply(neighbor_neg_distances, tf.constant(-1.0/DIST_INTERVAL))
	neighbor_distances = tf.add(neighbor_distances, tf.constant(1.0))
	neighbor_atoms = tf.gather(atoms, neighbor_atom_indices)
	#convert to int, since we'll need that for tf.nn.embedding_lookup
	neighbor_distances = tf.to_int32(neighbor_distances)

	#combine the neighbor atoms and neighbor distances
	z = tf.stack([neighbor_atoms, neighbor_distances], axis=1)
	#reshape our tensor to get shape [k_c, 2, m]
	z_transposed = tf.transpose(z, perm=[2, 1, 0])

	print layer_name, "output dimensions:", z_transposed.get_shape()
	return z_transposed

def embed_layer(layer_name, input_tensor):
	"""performs feature embedding on the input tensor, 
	transforming numbers to 200D vectors.
	Input: tensor of shape [k_c, 2, m]
	Output: tensor of shape [k_c, d_dist + d_atm, m, 1]
	"""
	with tf.name_scope(layer_name):
		with tf.name_scope('atom_weights'):
			W_atom = weight_variable([ATOM_TYPES, d_atm])
			variable_summaries(W_atom, layer_name + '/atom_weights')
		with tf.name_scope('dist_weights'):
			W_dist = weight_variable([DIST_BINS, d_dist])
			variable_summaries(W_dist, layer_name + '/dist_weights')

		h_embed = tf.nn.embedding_lookup([W_atom, W_dist], input_tensor, name='embed_layer')
		h_embed = tf.transpose(h_embed, perm=[0, 1, 3, 2])
		h_embed = tf.reshape(h_embed, [1, k_c, d_atm+d_dist, -1, 1])
		tf.summary.histogram(layer_name + '/embed_output', h_embed)

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
		tf.summary.histogram(layer_name + 'conv_output', h_conv)		
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

def deepVS_net(ligand_atoms, ligand_coords, keep_prob):
	"""ligand_atoms has dimensions [m], ligand_coords has dimensions [m, 3]"""
	#take ligand_atoms, ligand_coords, preprocess it into our z vector
	z_processed = preprocess_layer('preprocess', ligand_atoms, ligand_coords)
	#do the feature embedding to get a tensor with shape [k_c, d_dist+d_atm, m, 1]
	z_embed = embed_layer('embedding', z_processed)
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
	#get rid of the batch dimension 
	z_labels = tf.reshape(z_output, [2])
	return z_labels
import tensorflow as tf

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

def conv_layer(layer_name, input_tensor, filter_size, strides=[1, 1, 1, 1, 1], padding='SAME'):
    """makes a simple convolutional layer"""
    input_depth = filter_size[3]
    output_depth = filter_size[4]
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            W_conv = weight_variable(filter_size)
            variable_summaries(W_conv, layer_name + '/weights')
        with tf.name_scope('biases'):
            b_conv = bias_variable([output_depth])
            variable_summaries(b_conv, layer_name + '/biases')
            h_conv = tf.nn.conv3d(input_tensor, W_conv, strides=strides, padding=padding) + b_conv
            tf.summary.histogram(layer_name + '/pooling_output', h_conv)
    print layer_name,"output dimensions:", h_conv.get_shape()
    return h_conv

def relu_layer(layer_name,input_tensor,act=tf.nn.relu):
    """makes a simple relu layer"""
    with tf.name_scope(layer_name):
        h_relu = act(input_tensor, name='activation')
        tf.summary.histogram(layer_name + '/relu_output', h_relu)
        tf.summary.scalar(layer_name + '/sparsity', tf.nn.zero_fraction(h_relu))

    print layer_name, "output dimensions:", h_relu.get_shape()
    return h_relu

def pool_layer(layer_name,input_tensor,ksize,strides=[1, 1, 1, 1, 1],padding='SAME'):
    """makes a simple max pooling layer"""
    with tf.name_scope(layer_name):
        h_pool = tf.nn.max_pool3d(input_tensor,ksize=ksize,strides=strides,padding=padding)
        tf.summary.histogram(layer_name + '/max_pooling_output', h_pool)
    print layer_name, "output dimensions:", h_pool.get_shape()
    return h_pool


def avg_pool_layer(layer_name,input_tensor,ksize,strides=[1, 1, 1, 1, 1],padding='SAME'):
    """makes a average pooling layer"""
    with tf.name_scope(layer_name):
        h_pool = tf.nn.avg_pool3d(input_tensor,ksize=ksize,strides=strides,padding=padding)
        tf.summary.histogram(layer_name + '/average_pooling_output', h_pool)
    print layer_name, "output dimensions:", h_pool.get_shape()
    return h_pool



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



def max_net(x_image_batch,keep_prob,batch_size):
    "makes a simple network that can receive 20x20x20 input images. And output 2 classes"
    with tf.name_scope('input'):
        pass
    with tf.name_scope("input_reshape"):
        print "image batch dimensions", x_image_batch.get_shape()
        # formally adding one depth dimension to the input
        x_image_batch = tf.reshape(x_image_batch, [batch_size, 20, 20, 20, 1])
        print "input to the first layer dimensions", x_image_batch.get_shape()

    h_conv1 = conv_layer(layer_name='conv1_5x5x5', input_tensor=x_image_batch, filter_size=[5, 5, 5, 1, 20])
    h_relu1 = relu_layer(layer_name='relu1', input_tensor=h_conv1)
    h_pool1 = pool_layer(layer_name='pool1_2x2x2', input_tensor=h_relu1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1])

    h_conv2 = conv_layer(layer_name="conv2_3x3x3", input_tensor=h_pool1, filter_size=[3, 3, 3, 20, 30])
    h_relu2 = relu_layer(layer_name="relu2", input_tensor=h_conv2)
    h_pool2 = pool_layer(layer_name="pool2_2x2x2", input_tensor=h_relu2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1])

    h_conv3 = conv_layer(layer_name="conv3_2x2x2", input_tensor=h_pool2, filter_size=[2, 2, 2, 30, 40])
    h_relu3 = relu_layer(layer_name="relu3", input_tensor=h_conv3)
    h_pool3 = pool_layer(layer_name="pool3_2x2x2", input_tensor=h_relu3, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1])

    h_conv4 = conv_layer(layer_name="conv4_2x2x2", input_tensor=h_pool3, filter_size=[2, 2, 2, 40, 50])
    h_relu4 = relu_layer(layer_name="relu4", input_tensor=h_conv4)
    h_pool4 = pool_layer(layer_name="pool4_2x2x2", input_tensor=h_relu4, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1])

    h_conv5 = conv_layer(layer_name="conv5_2x2x2", input_tensor=h_pool4, filter_size=[2, 2, 2, 50, 60])
    h_relu5 = relu_layer(layer_name="relu5", input_tensor=h_conv5)
    h_pool5 = pool_layer(layer_name="pool5_2x2x2", input_tensor=h_relu5, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1])

    with tf.name_scope("flatten_layer"):
        h_pool2_flat = tf.reshape(h_pool5, [-1, 5 * 5 * 5 * 60])

    h_fc1 = fc_layer(layer_name="fc1", input_tensor=h_pool2_flat, output_dim=1024)
    h_fc1_relu = relu_layer(layer_name="fc1_relu", input_tensor=h_fc1)

    with tf.name_scope("dropout"):
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        h_fc1_drop = tf.nn.dropout(h_fc1_relu, keep_prob)

    h_fc2 = fc_layer(layer_name="fc2", input_tensor=h_fc1_drop, output_dim=256)
    h_fc2_relu = relu_layer(layer_name="fc2_relu", input_tensor=h_fc2)

    y_conv = fc_layer(layer_name="out_neuron", input_tensor=h_fc2_relu, output_dim=2)

    return y_conv




class Network:
    def __init__(self):
        self.w1 = weight_variable([5, 5, 5, 1, 20])
        self.w2 = weight_variable([3, 3, 3, 20, 30])
        self.w3 = weight_variable([2, 2, 2, 30, 40])
        self.w4 = weight_variable([2, 2, 2, 40, 50])
        self.w5 = weight_variable([2, 2, 2, 50, 60])

        self.b1 = bias_variable([20])
        self.b2 = bias_variable([30])
        self.b3 = bias_variable([40])
        self.b4 = bias_variable([50])
        self.b5 = bias_variable([60])

        self.fc1w = weight_variable([7500,1024])
        self.fc1b = bias_variable([1024])
        self.fc2w = weight_variable([1024,256])
        self.fc2b = bias_variable([256])
        self.fc3w = weight_variable([256,2])
        self.fc3b = bias_variable([2])

    def compute_output(self,image_batch,keep_prob):
        x_image_batch = tf.reshape(image_batch, [100, 20, 20, 20, 1])

        h_conv1 = tf.nn.conv3d(x_image_batch,self.w1, strides=[1,1,1,1,1], padding='SAME') + self.b1
        h_relu1 = tf.nn.relu(h_conv1)
        h_pool1 = tf.nn.max_pool3d(h_relu1,ksize=[1, 2, 2, 2, 1],strides=[1, 2, 2, 2, 1],padding='SAME')

        h_conv2 = tf.nn.conv3d(h_pool1,self.w2, strides=[1,1,1,1,1], padding='SAME') + self.b2
        h_relu2 = tf.nn.relu(h_conv2)
        h_pool2 = tf.nn.max_pool3d(h_relu2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

        h_conv3 = tf.nn.conv3d(h_pool2, self.w3, strides=[1, 1, 1, 1, 1], padding='SAME') + self.b3
        h_relu3 = tf.nn.relu(h_conv3)
        h_pool3 = tf.nn.max_pool3d(h_relu3, ksize=[1, 1, 1, 1, 1], strides=[1, 1, 1, 1, 1], padding='SAME')

        h_conv4 = tf.nn.conv3d(h_pool3, self.w4, strides=[1, 1, 1, 1, 1], padding='SAME') + self.b4
        h_relu4 = tf.nn.relu(h_conv4)
        h_pool4 = tf.nn.max_pool3d(h_relu4, ksize=[1, 1, 1, 1, 1], strides=[1, 1, 1, 1, 1], padding='SAME')

        h_conv5 = tf.nn.conv3d(h_pool4, self.w5, strides=[1, 1, 1, 1, 1], padding='SAME') + self.b5
        h_relu5 = tf.nn.relu(h_conv5)
        h_pool5 = tf.nn.max_pool3d(h_relu5, ksize=[1, 1, 1, 1, 1], strides=[1, 1, 1, 1, 1], padding='SAME')

        h_pool2_flat = tf.reshape(h_pool5, [-1, 5 * 5 * 5 * 60])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,self.fc1w) + self.fc1b)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, self.fc2w) + self.fc2b)
        y_conv = tf.matmul(h_fc2, self.fc3w) + self.fc3b

        return y_conv



my_net = Network()



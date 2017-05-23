import tensorflow as tf
import numpy as np
import time

# generate one very large tensor
# take slices from that tensor
# assuming our tensor is big,
# random slices from it should represent affine transform

def generate_deep_affine_transform(num_frames):
    """Generates a very big batch of affine transform matrices in 3D. The first dimension is batch, the other two
    describe typical affine transform matrices. Deep affine transform can be generated once in the beginning
    of training, and later slices can be taken from it randomly to speed up the computation."""

    # shift range is hard coded to 10A because that's how the proteins look like
    # rotation range is hardcoded to 360 degrees

    shift_range = tf.constant(10, dtype=tf.float32)  # FIXME
    rotation_range = tf.cast(tf.convert_to_tensor(np.pi * 2), dtype=tf.float32)

    #shift_range = tf.constant(0, dtype=tf.float32)  # FIXME
    #rotation_range = tf.cast(tf.convert_to_tensor(0), dtype=tf.float32)

    # randomly shift along X,Y,Z
    x_shift = tf.random_uniform([num_frames], minval=-1, maxval=1, dtype=tf.float32) * shift_range
    y_shift = tf.random_uniform([num_frames], minval=-1, maxval=1, dtype=tf.float32) * shift_range
    z_shift = tf.random_uniform([num_frames], minval=-1, maxval=1, dtype=tf.float32) * shift_range

    # [1, 0, 0, random_x_shift],
    # [0, 1, 0, random_y_shift],
    # [0, 0, 1, random_z_shift],
    # [0, 0, 0, 1]])

    # try to do the following:
    # generate nine tensors for each of them
    # concatenate and reshape sixteen tensors

    afn0_0 = tf.ones([num_frames])
    afn0_1 = tf.zeros([num_frames])
    afn0_2 = tf.zeros([num_frames])
    afn0_3 = x_shift

    afn1_0 = tf.zeros([num_frames])
    afn1_1 = tf.ones([num_frames])
    afn1_2 = tf.zeros([num_frames])
    afn1_3 = y_shift

    afn2_0 = tf.zeros([num_frames])
    afn2_1 = tf.zeros([num_frames])
    afn2_2 = tf.ones([num_frames])
    afn2_3 = z_shift

    afn3_0 = tf.zeros([num_frames])
    afn3_1 = tf.zeros([num_frames])
    afn3_2 = tf.zeros([num_frames])
    afn3_3 = tf.ones([num_frames])

    xyz_shift_stick = tf.stack(
        [afn0_0, afn0_1, afn0_2, afn0_3, afn1_0, afn1_1, afn1_2, afn1_3, afn2_0, afn2_1, afn2_2, afn2_3, afn3_0,
         afn3_1, afn3_2, afn3_3])
    xyz_shift_matrix = tf.transpose(tf.reshape(xyz_shift_stick, [4, 4, num_frames]), perm=[2, 0, 1])

    # randomly rotate along X
    x_rot = tf.random_uniform([num_frames], minval=-1, maxval=1, dtype=tf.float32, seed=None,
                              name=None) * rotation_range

    # [[1, 0, 0, 0],
    # [0, cos(x_rot),-sin(x_rot),0],
    # [0, sin(x_rot),cos(x_rot),0],
    # [0, 0, 0, 1]],dtype=tf.float32)

    afn0_0 = tf.ones([num_frames])
    afn0_1 = tf.zeros([num_frames])
    afn0_2 = tf.zeros([num_frames])
    afn0_3 = tf.zeros([num_frames])

    afn1_0 = tf.zeros([num_frames])
    afn1_1 = tf.cos(x_rot)
    afn1_2 = -tf.sin(x_rot)
    afn1_3 = tf.zeros([num_frames])

    afn2_0 = tf.zeros([num_frames])
    afn2_1 = tf.sin(x_rot)
    afn2_2 = tf.cos(x_rot)
    afn2_3 = tf.zeros([num_frames])

    afn3_0 = tf.zeros([num_frames])
    afn3_1 = tf.zeros([num_frames])
    afn3_2 = tf.zeros([num_frames])
    afn3_3 = tf.ones([num_frames])

    x_rot_stick = tf.stack(
        [afn0_0, afn0_1, afn0_2, afn0_3, afn1_0, afn1_1, afn1_2, afn1_3, afn2_0, afn2_1, afn2_2, afn2_3, afn3_0,
         afn3_1, afn3_2, afn3_3])
    x_rot_matrix = tf.transpose(tf.reshape(x_rot_stick, [4, 4, num_frames]), perm=[2, 0, 1])

    # randomly rotate along Y
    y_rot = tf.random_uniform([num_frames], minval=-1, maxval=1, dtype=tf.float32, seed=None,
                              name=None) * rotation_range

    # [cos(y_rot), 0,sin(y_rot), 0],
    # [0, 1, 0, 0],
    # [-sin(y_rot), 0,cos(y_rot), 0],
    # [0, 0 ,0 ,1]])

    afn0_0 = tf.cos(y_rot)
    afn0_1 = tf.zeros([num_frames])
    afn0_2 = tf.sin(y_rot)
    afn0_3 = tf.zeros([num_frames])

    afn1_0 = tf.zeros([num_frames])
    afn1_1 = tf.ones([num_frames])
    afn1_2 = tf.zeros([num_frames])
    afn1_3 = tf.zeros([num_frames])

    afn2_0 = -tf.sin(y_rot)
    afn2_1 = tf.zeros([num_frames])
    afn2_2 = tf.cos(y_rot)
    afn2_3 = tf.zeros([num_frames])

    afn3_0 = tf.zeros([num_frames])
    afn3_1 = tf.zeros([num_frames])
    afn3_2 = tf.zeros([num_frames])
    afn3_3 = tf.ones([num_frames])

    y_rot_stick = tf.stack(
        [afn0_0, afn0_1, afn0_2, afn0_3, afn1_0, afn1_1, afn1_2, afn1_3, afn2_0, afn2_1, afn2_2, afn2_3, afn3_0,
         afn3_1, afn3_2, afn3_3])
    y_rot_matrix = tf.transpose(tf.reshape(y_rot_stick, [4, 4, num_frames]), perm=[2, 0, 1])

    # randomly rotate along Z
    z_rot = tf.random_uniform([num_frames], minval=-1, maxval=1, dtype=tf.float32, seed=None,
                              name=None) * rotation_range

    # [[cos(z_rot), -sin(z_rot), 0, 0],
    # [sin(z_rot), cos(z_rot), 0, 0],
    # [0, 0, 1, 0],
    # [0, 0, 0, 1]])

    afn0_0 = tf.cos(z_rot)
    afn0_1 = -tf.sin(z_rot)
    afn0_2 = tf.zeros([num_frames])
    afn0_3 = tf.zeros([num_frames])

    afn1_0 = tf.sin(z_rot)
    afn1_1 = tf.cos(z_rot)
    afn1_2 = tf.zeros([num_frames])
    afn1_3 = tf.zeros([num_frames])

    afn2_0 = tf.zeros([num_frames])
    afn2_1 = tf.zeros([num_frames])
    afn2_2 = tf.ones([num_frames])
    afn2_3 = tf.zeros([num_frames])

    afn3_0 = tf.zeros([num_frames])
    afn3_1 = tf.zeros([num_frames])
    afn3_2 = tf.zeros([num_frames])
    afn3_3 = tf.ones([num_frames])

    z_rot_stick = tf.stack(
        [afn0_0, afn0_1, afn0_2, afn0_3, afn1_0, afn1_1, afn1_2, afn1_3, afn2_0, afn2_1, afn2_2, afn2_3, afn3_0,
         afn3_1, afn3_2, afn3_3])
    z_rot_matrix = tf.transpose(tf.reshape(z_rot_stick, [4, 4, num_frames]), perm=[2, 0, 1])

    xyz_shift_xyz_rot = tf.matmul(tf.matmul(tf.matmul(xyz_shift_matrix, x_rot_matrix), y_rot_matrix), z_rot_matrix)

    return xyz_shift_xyz_rot



def gen_deep_affine_tform_with_labels(num_frames, x_shift_range=0,
                                      y_shift_range=0,
                                      z_shift_range=0,
                                      x_rot_range=tf.cast(tf.convert_to_tensor(0*np.pi), tf.float32),
                                      y_rot_range=tf.cast(tf.convert_to_tensor(0*np.pi), tf.float32),
                                      z_rot_range=tf.cast(tf.convert_to_tensor(0*np.pi), tf.float32),
                                      shift_partitions = None,
                                      rot_partitions = None,
                                      abs=False):
    """

    :param num_frames:
    :param x_shift_range:
    :param y_shift_range:
    :param z_shift_range:
    :param x_rot_range:
    :param y_rot_range:
    :param z_rot_range:
    :param shift_partitions:
    :param rot_partitions:
    :param abs:
    :return:
    """

    if shift_partitions is None and not abs:
        x_shift = tf.random_uniform([num_frames], minval=-x_shift_range, maxval=x_shift_range, dtype=tf.float32)
        y_shift = tf.random_uniform([num_frames], minval=-y_shift_range, maxval=y_shift_range, dtype=tf.float32)
        z_shift = tf.random_uniform([num_frames], minval=-z_shift_range, maxval=z_shift_range, dtype=tf.float32)
    elif shift_partitions is None and abs:
        x_shift = tf.random_uniform([num_frames], minval=0, maxval=x_shift_range, dtype=tf.float32)
        y_shift = tf.random_uniform([num_frames], minval=0, maxval=y_shift_range, dtype=tf.float32)
        z_shift = tf.random_uniform([num_frames], minval=0, maxval=z_shift_range, dtype=tf.float32)
    elif shift_partitions is not None and not abs:
        x_shift = tf.to_float(tf.random_uniform([num_frames], minval=-shift_partitions, maxval=shift_partitions+1, dtype=tf.int32)) * (x_shift_range / shift_partitions)
        y_shift = tf.to_float(tf.random_uniform([num_frames], minval=-shift_partitions, maxval=shift_partitions+1, dtype=tf.int32)) * (y_shift_range / shift_partitions)
        z_shift = tf.to_float(tf.random_uniform([num_frames], minval=-shift_partitions, maxval=shift_partitions+1, dtype=tf.int32)) * (z_shift_range / shift_partitions)
    elif shift_partitions is not None and abs:
        x_shift = tf.to_float(tf.random_uniform([num_frames], minval=0, maxval=shift_partitions+1, dtype=tf.int32)) * (x_shift_range / shift_partitions)
        y_shift = tf.to_float(tf.random_uniform([num_frames], minval=0, maxval=shift_partitions+1, dtype=tf.int32)) * (y_shift_range / shift_partitions)
        z_shift = tf.to_float(tf.random_uniform([num_frames], minval=0, maxval=shift_partitions+1, dtype=tf.int32)) * (z_shift_range / shift_partitions)
    else:
        raise ValueError("combination of shift partitions and abs is not in a list of allowed values")


    if rot_partitions is None and not abs:
        x_rot = tf.random_uniform([num_frames], minval=-x_rot_range, maxval=x_rot_range, dtype=tf.float32)
        y_rot = tf.random_uniform([num_frames], minval=-y_rot_range, maxval=y_rot_range, dtype=tf.float32)
        z_rot = tf.random_uniform([num_frames], minval=-z_rot_range, maxval=z_rot_range, dtype=tf.float32)
    elif rot_partitions is None and abs:
        x_rot = tf.random_uniform([num_frames], minval=0, maxval=x_rot_range, dtype=tf.float32)
        y_rot = tf.random_uniform([num_frames], minval=0, maxval=y_rot_range, dtype=tf.float32)
        z_rot = tf.random_uniform([num_frames], minval=0, maxval=z_rot_range, dtype=tf.float32)
    elif rot_partitions is not None and not abs:
        x_rot = tf.to_float(tf.random_uniform([num_frames], minval=-rot_partitions, maxval=rot_partitions+1, dtype=tf.int32)) * (x_rot_range / rot_partitions)
        y_rot = tf.to_float(tf.random_uniform([num_frames], minval=-rot_partitions, maxval=rot_partitions+1, dtype=tf.int32)) * (y_rot_range / rot_partitions)
        z_rot = tf.to_float(tf.random_uniform([num_frames], minval=-rot_partitions, maxval=rot_partitions+1, dtype=tf.float32)) * (z_rot_range / rot_partitions)
    elif rot_partitions is not None and abs:
        x_rot = tf.to_float(tf.random_uniform([num_frames], minval=0, maxval=rot_partitions+1, dtype=tf.int32)) * (x_rot_range / rot_partitions)
        y_rot = tf.to_float(tf.random_uniform([num_frames], minval=0, maxval=rot_partitions+1, dtype=tf.int32)) * (y_rot_range / rot_partitions)
        z_rot = tf.to_float(tf.random_uniform([num_frames], minval=0, maxval=rot_partitions+1, dtype=tf.int32)) * (z_rot_range / rot_partitions)
    else:
        raise ValueError("combination of rot partitions and abs is not in a list of allowed values")


    # [1, 0, 0, random_x_shift],
    # [0, 1, 0, random_y_shift],
    # [0, 0, 1, random_z_shift],
    # [0, 0, 0, 1]])

    # try to do the following:
    # generate nine tensors for each of them
    # concatenate and reshape sixteen tensors

    afn0_0 = tf.ones([num_frames])
    afn0_1 = tf.zeros([num_frames])
    afn0_2 = tf.zeros([num_frames])
    afn0_3 = x_shift

    afn1_0 = tf.zeros([num_frames])
    afn1_1 = tf.ones([num_frames])
    afn1_2 = tf.zeros([num_frames])
    afn1_3 = y_shift

    afn2_0 = tf.zeros([num_frames])
    afn2_1 = tf.zeros([num_frames])
    afn2_2 = tf.ones([num_frames])
    afn2_3 = z_shift

    afn3_0 = tf.zeros([num_frames])
    afn3_1 = tf.zeros([num_frames])
    afn3_2 = tf.zeros([num_frames])
    afn3_3 = tf.ones([num_frames])

    xyz_shift_stick = tf.stack(
        [afn0_0, afn0_1, afn0_2, afn0_3, afn1_0, afn1_1, afn1_2, afn1_3, afn2_0, afn2_1, afn2_2, afn2_3, afn3_0,
         afn3_1, afn3_2, afn3_3])
    xyz_shift_matrix = tf.transpose(tf.reshape(xyz_shift_stick, [4, 4, num_frames]), perm=[2, 0, 1])

    # randomly rotate along X
    # [[1, 0, 0, 0],
    # [0, cos(x_rot),-sin(x_rot),0],
    # [0, sin(x_rot),cos(x_rot),0],
    # [0, 0, 0, 1]],dtype=tf.float32)

    afn0_0 = tf.ones([num_frames])
    afn0_1 = tf.zeros([num_frames])
    afn0_2 = tf.zeros([num_frames])
    afn0_3 = tf.zeros([num_frames])

    afn1_0 = tf.zeros([num_frames])
    afn1_1 = tf.cos(x_rot)
    afn1_2 = -tf.sin(x_rot)
    afn1_3 = tf.zeros([num_frames])

    afn2_0 = tf.zeros([num_frames])
    afn2_1 = tf.sin(x_rot)
    afn2_2 = tf.cos(x_rot)
    afn2_3 = tf.zeros([num_frames])

    afn3_0 = tf.zeros([num_frames])
    afn3_1 = tf.zeros([num_frames])
    afn3_2 = tf.zeros([num_frames])
    afn3_3 = tf.ones([num_frames])

    x_rot_stick = tf.stack(
        [afn0_0, afn0_1, afn0_2, afn0_3, afn1_0, afn1_1, afn1_2, afn1_3, afn2_0, afn2_1, afn2_2, afn2_3, afn3_0,
         afn3_1, afn3_2, afn3_3])
    x_rot_matrix = tf.transpose(tf.reshape(x_rot_stick, [4, 4, num_frames]), perm=[2, 0, 1])

    # randomly rotate along Y
    # [cos(y_rot), 0,sin(y_rot), 0],
    # [0, 1, 0, 0],
    # [-sin(y_rot), 0,cos(y_rot), 0],
    # [0, 0 ,0 ,1]])

    afn0_0 = tf.cos(y_rot)
    afn0_1 = tf.zeros([num_frames])
    afn0_2 = tf.sin(y_rot)
    afn0_3 = tf.zeros([num_frames])

    afn1_0 = tf.zeros([num_frames])
    afn1_1 = tf.ones([num_frames])
    afn1_2 = tf.zeros([num_frames])
    afn1_3 = tf.zeros([num_frames])

    afn2_0 = -tf.sin(y_rot)
    afn2_1 = tf.zeros([num_frames])
    afn2_2 = tf.cos(y_rot)
    afn2_3 = tf.zeros([num_frames])

    afn3_0 = tf.zeros([num_frames])
    afn3_1 = tf.zeros([num_frames])
    afn3_2 = tf.zeros([num_frames])
    afn3_3 = tf.ones([num_frames])

    y_rot_stick = tf.stack(
        [afn0_0, afn0_1, afn0_2, afn0_3, afn1_0, afn1_1, afn1_2, afn1_3, afn2_0, afn2_1, afn2_2, afn2_3, afn3_0,
         afn3_1, afn3_2, afn3_3])
    y_rot_matrix = tf.transpose(tf.reshape(y_rot_stick, [4, 4, num_frames]), perm=[2, 0, 1])

    # randomly rotate along Z
    # [[cos(z_rot), -sin(z_rot), 0, 0],
    # [sin(z_rot), cos(z_rot), 0, 0],
    # [0, 0, 1, 0],
    # [0, 0, 0, 1]])

    afn0_0 = tf.cos(z_rot)
    afn0_1 = -tf.sin(z_rot)
    afn0_2 = tf.zeros([num_frames])
    afn0_3 = tf.zeros([num_frames])

    afn1_0 = tf.sin(z_rot)
    afn1_1 = tf.cos(z_rot)
    afn1_2 = tf.zeros([num_frames])
    afn1_3 = tf.zeros([num_frames])

    afn2_0 = tf.zeros([num_frames])
    afn2_1 = tf.zeros([num_frames])
    afn2_2 = tf.ones([num_frames])
    afn2_3 = tf.zeros([num_frames])

    afn3_0 = tf.zeros([num_frames])
    afn3_1 = tf.zeros([num_frames])
    afn3_2 = tf.zeros([num_frames])
    afn3_3 = tf.ones([num_frames])

    z_rot_stick = tf.stack(
        [afn0_0, afn0_1, afn0_2, afn0_3, afn1_0, afn1_1, afn1_2, afn1_3, afn2_0, afn2_1, afn2_2, afn2_3, afn3_0,
         afn3_1, afn3_2, afn3_3])
    z_rot_matrix = tf.transpose(tf.reshape(z_rot_stick, [4, 4, num_frames]), perm=[2, 0, 1])

    xyz_shift_xyz_rot = tf.matmul(tf.matmul(tf.matmul(xyz_shift_matrix, x_rot_matrix), y_rot_matrix), z_rot_matrix)
    xyz_labels = tf.transpose(tf.reshape(tf.stack([x_shift,y_shift,z_shift,x_rot,y_rot,z_rot]),[6, num_frames]), perm=[1,0])

    return tf.Variable(xyz_shift_xyz_rot), tf.Variable(xyz_labels)




def affine_transform(coordinates,transition_matrix):
    """applies affine transform to the array of coordinates. By default generates a random affine transform matrix."""
    coordinates_with_ones = tf.concat([coordinates, tf.cast(tf.ones([tf.shape(coordinates)[0],1]),tf.float32)],1)
    transformed_coords = tf.matmul(coordinates_with_ones,tf.transpose(transition_matrix))[0:,:-1]

    return transformed_coords,transition_matrix


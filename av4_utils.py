from __future__ import absolute_import
from __future__ import division

import threading
import weakref

from tensorflow.core.protobuf import queue_runner_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging

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
    rotation_range = tf.cast(tf.convert_to_tensor(0), dtype=tf.float32)

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

def affine_transform(coordinates,transition_matrix):
    """Applies affine transform to the array of X,Y,Z coordinates. By default generates a random affine transform matrix."""
    coordinates_with_ones = tf.concat([coordinates, tf.cast(tf.ones([tf.shape(coordinates)[0],1]),tf.float32)],1)
    transformed_coords = tf.matmul(coordinates_with_ones,tf.transpose(transition_matrix))[0:,:-1]

    return transformed_coords,transition_matrix


def deep_affine_transform(coords,deep_transition_matrix):
    """Applies multiple affine transformations to the array of X,Y,Z coordinates."""
    # TODO generate random affine transform matrix by default
    depth_dimensions = tf.shape(deep_transition_matrix)[0]
    coords_with_ones = tf.concat([coords, tf.cast(tf.ones([tf.shape(coords)[0],1]),tf.float32)],1)
    broadcast_coords_with_ones = tf.reshape(tf.tile(coords_with_ones,[depth_dimensions,1]),[depth_dimensions,tf.shape(coords)[0],4])
    transformed_coords = tf.batch_matmul(broadcast_coords_with_ones,tf.transpose(deep_transition_matrix,[0,2,1]))[:,:,:3]

    return transformed_coords,deep_transition_matrix



def generate_exhaustive_affine_transform(shift_ranges=[5,5,5],shift_deltas=[1,1,1],rot_ranges=[360,360,360]):
    """By default,makes shifts by 1, in X,Y,Z directions"""

    # shift along X,Y,Z
    x_shift = tf.range(start=-shift_ranges[0],limit=shift_ranges[0],delta=shift_deltas[0],dtype=tf.float32)
    y_shift = tf.range(start=-shift_ranges[1],limit=shift_ranges[1],delta=shift_deltas[1],dtype=tf.float32)
    z_shift = tf.range(start=-shift_ranges[2],limit=shift_ranges[2],delta=shift_deltas[2],dtype=tf.float32)

    # [1, 0, 0, random_x_shift],
    # [0, 1, 0, random_y_shift],
    # [0, 0, 1, random_z_shift],
    # [0, 0, 0, 1]])


    # get affine transformation shifts along X
    num_frames = tf.shape(x_shift)[0]

    afn0_0 = tf.ones([num_frames])
    afn0_1 = tf.zeros([num_frames])
    afn0_2 = tf.zeros([num_frames])
    afn0_3 = x_shift

    afn1_0 = tf.zeros([num_frames])
    afn1_1 = tf.ones([num_frames])
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

    x_shift_stick = tf.stack(
        [afn0_0, afn0_1, afn0_2, afn0_3, afn1_0, afn1_1, afn1_2, afn1_3, afn2_0, afn2_1, afn2_2, afn2_3, afn3_0,
         afn3_1, afn3_2, afn3_3])

    x_shift_matrix = tf.transpose(tf.reshape(x_shift_stick, [4, 4, num_frames]), perm=[2, 0, 1])

    # get affine transformations along Y
    afn0_0 = tf.ones([num_frames])
    afn0_1 = tf.zeros([num_frames])
    afn0_2 = tf.zeros([num_frames])
    afn0_3 = tf.zeros([num_frames])

    afn1_0 = tf.zeros([num_frames])
    afn1_1 = tf.ones([num_frames])
    afn1_2 = tf.zeros([num_frames])
    afn1_3 = y_shift

    afn2_0 = tf.zeros([num_frames])
    afn2_1 = tf.zeros([num_frames])
    afn2_2 = tf.ones([num_frames])
    afn2_3 = tf.zeros([num_frames])

    afn3_0 = tf.zeros([num_frames])
    afn3_1 = tf.zeros([num_frames])
    afn3_2 = tf.zeros([num_frames])
    afn3_3 = tf.ones([num_frames])

    y_shift_stick = tf.stack(
        [afn0_0, afn0_1, afn0_2, afn0_3, afn1_0, afn1_1, afn1_2, afn1_3, afn2_0, afn2_1, afn2_2, afn2_3, afn3_0,
         afn3_1, afn3_2, afn3_3])

    y_shift_matrix = tf.transpose(tf.reshape(y_shift_stick, [4, 4, num_frames]), perm=[2, 0, 1])

    # get affine transformations along Z
    afn0_0 = tf.ones([num_frames])
    afn0_1 = tf.zeros([num_frames])
    afn0_2 = tf.zeros([num_frames])
    afn0_3 = tf.zeros([num_frames])

    afn1_0 = tf.zeros([num_frames])
    afn1_1 = tf.ones([num_frames])
    afn1_2 = tf.zeros([num_frames])
    afn1_3 = tf.zeros([num_frames])

    afn2_0 = tf.zeros([num_frames])
    afn2_1 = tf.zeros([num_frames])
    afn2_2 = tf.ones([num_frames])
    afn2_3 = z_shift

    afn3_0 = tf.zeros([num_frames])
    afn3_1 = tf.zeros([num_frames])
    afn3_2 = tf.zeros([num_frames])
    afn3_3 = tf.ones([num_frames])

    z_shift_stick = tf.stack(
        [afn0_0, afn0_1, afn0_2, afn0_3, afn1_0, afn1_1, afn1_2, afn1_3, afn2_0, afn2_1, afn2_2, afn2_3, afn3_0,
         afn3_1, afn3_2, afn3_3])

    z_shift_matrix = tf.transpose(tf.reshape(z_shift_stick, [4, 4, num_frames]), perm=[2, 0, 1])

    # combine X,Y,Z shifts together
    broadcast_x = tf.tile(x_shift_matrix, [tf.shape(y_shift_matrix)[0], 1, 1])
    broadcast_y = tf.reshape(tf.tile(y_shift_matrix, [1, tf.shape(x_shift_matrix)[0], 1]),shape=[tf.shape(x_shift_matrix)[0] * tf.shape(y_shift_matrix)[0], 4, 4])
    xy_shift_matrix = tf.matmul(broadcast_x,broadcast_y)
    broadcast_xy = tf.tile(xy_shift_matrix, [tf.shape(z_shift_matrix)[0], 1, 1])
    broadcast_z = tf.reshape(tf.tile(z_shift_matrix, [1, tf.shape(xy_shift_matrix)[0], 1]),shape=[tf.shape(xy_shift_matrix)[0] * tf.shape(z_shift_matrix)[0], 4, 4])
    xyz_shift_matrix = tf.matmul(broadcast_xy, broadcast_z)

    return xyz_shift_matrix


def generate_identity_matrices(num_frames):
    "for convenience of generating identity transformation matrices"

    afn0_0 = tf.ones([num_frames])
    afn0_1 = tf.zeros([num_frames])
    afn0_2 = tf.zeros([num_frames])
    afn0_3 = tf.zeros([num_frames])

    afn1_0 = tf.zeros([num_frames])
    afn1_1 = tf.ones([num_frames])
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

    identity_stick = tf.stack(
        [afn0_0, afn0_1, afn0_2, afn0_3, afn1_0, afn1_1, afn1_2, afn1_3, afn2_0, afn2_1, afn2_2, afn2_3, afn3_0,
         afn3_1, afn3_2, afn3_3])

    return tf.transpose(tf.reshape(identity_stick, [4, 4, num_frames]), perm=[2, 0, 1])




def describe_variables(vals):
    "Take a dictionary of variables and variable names, and prints statistics."
    print "==============================================================================="
    for val in vals:
        print val,"\tlen:",len(vals[val]), "min:","%.3f" %  min(vals[val]), "max:","%.3f" % max(vals[val]),"ave:",
        print "%.3f" % np.average(vals[val]), "median:", "%.3f" % np.sort(vals[val], axis=0)[len(vals[val])//2],
        print "first:","%.3f" % vals[val][0], "last:","%.3f" % vals[val][-1],
        print "middle (no formatting):",vals[val][len(vals[val])//2]
    print "================================================================================"




class QueueRunner(object):
  """
  I borrowed this from a standard TensorFlow collection

  The only difference between tf.QueueRunner and this one is that it does not close the queue after terminating
  all threads. So I can initiate and kill threads on the same queue many times.

  Holds a list of enqueue operations for a queue, each to be run in a thread.

  Queues are a convenient TensorFlow mechanism to compute tensors
  asynchronously using multiple threads. For example in the canonical 'Input
  Reader' setup one set of threads generates filenames in a queue; a second set
  of threads read records from the files, processes them, and enqueues tensors
  on a second queue; a third set of threads dequeues these input records to
  construct batches and runs them through training operations.

  There are several delicate issues when running multiple threads that way:
  closing the queues in sequence as the input is exhausted, correctly catching
  and reporting exceptions, etc.

  The `QueueRunner`, combined with the `Coordinator`, helps handle these issues.
  """

  def __init__(self, queue=None, enqueue_ops=None, close_op=None,
               cancel_op=None, queue_closed_exception_types=None,
               queue_runner_def=None, import_scope=None):
    """Create a QueueRunner.

    On construction the `QueueRunner` adds an op to close the queue.  That op
    will be run if the enqueue ops raise exceptions.

    When you later call the `create_threads()` method, the `QueueRunner` will
    create one thread for each op in `enqueue_ops`.  Each thread will run its
    enqueue op in parallel with the other threads.  The enqueue ops do not have
    to all be the same op, but it is expected that they all enqueue tensors in
    `queue`.

    Args:
      queue: A `Queue`.
      enqueue_ops: List of enqueue ops to run in threads later.
      close_op: Op to close the queue. Pending enqueue ops are preserved.
      cancel_op: Op to close the queue and cancel pending enqueue ops.
      queue_closed_exception_types: Optional tuple of Exception types that
        indicate that the queue has been closed when raised during an enqueue
        operation.  Defaults to `(tf.errors.OutOfRangeError,)`.  Another common
        case includes `(tf.errors.OutOfRangeError, tf.errors.CancelledError)`,
        when some of the enqueue ops may dequeue from other Queues.
      queue_runner_def: Optional `QueueRunnerDef` protocol buffer. If specified,
        recreates the QueueRunner from its contents. `queue_runner_def` and the
        other arguments are mutually exclusive.
      import_scope: Optional `string`. Name scope to add. Only used when
        initializing from protocol buffer.

    Raises:
      ValueError: If both `queue_runner_def` and `queue` are both specified.
      ValueError: If `queue` or `enqueue_ops` are not provided when not
        restoring from `queue_runner_def`.
    """
    if queue_runner_def:
      if queue or enqueue_ops:
        raise ValueError("queue_runner_def and queue are mutually exclusive.")
      self._init_from_proto(queue_runner_def,
                            import_scope=import_scope)
    else:
      self._init_from_args(
          queue=queue, enqueue_ops=enqueue_ops,
          close_op=close_op, cancel_op=cancel_op,
          queue_closed_exception_types=queue_closed_exception_types)
    # Protect the count of runs to wait for.
    self._lock = threading.Lock()
    # A map from a session object to the number of outstanding queue runner
    # threads for that session.
    self._runs_per_session = weakref.WeakKeyDictionary()
    # List of exceptions raised by the running threads.
    self._exceptions_raised = []

  def _init_from_args(self, queue=None, enqueue_ops=None, close_op=None,
                      cancel_op=None, queue_closed_exception_types=None):
    """Create a QueueRunner from arguments.

    Args:
      queue: A `Queue`.
      enqueue_ops: List of enqueue ops to run in threads later.
      close_op: Op to close the queue. Pending enqueue ops are preserved.
      cancel_op: Op to close the queue and cancel pending enqueue ops.
      queue_closed_exception_types: Tuple of exception types, which indicate
        the queue has been safely closed.

    Raises:
      ValueError: If `queue` or `enqueue_ops` are not provided when not
        restoring from `queue_runner_def`.
      TypeError: If `queue_closed_exception_types` is provided, but is not
        a non-empty tuple of error types (subclasses of `tf.errors.OpError`).
    """
    if not queue or not enqueue_ops:
      raise ValueError("Must provide queue and enqueue_ops.")
    self._queue = queue
    self._enqueue_ops = enqueue_ops
    self._close_op = close_op
    self._cancel_op = cancel_op
    if queue_closed_exception_types is not None:
      if (not isinstance(queue_closed_exception_types, tuple)
          or not queue_closed_exception_types
          or not all(issubclass(t, errors.OpError)
                     for t in queue_closed_exception_types)):
        raise TypeError(
            "queue_closed_exception_types, when provided, "
            "must be a non-empty list of tf.error types, but saw: %s"
            % queue_closed_exception_types)
    self._queue_closed_exception_types = queue_closed_exception_types
    # Close when no more will be produced, but pending enqueues should be
    # preserved.
    if self._close_op is None:
      self._close_op = self._queue.close()
    # Close and cancel pending enqueues since there was an error and we want
    # to unblock everything so we can cleanly exit.
    if self._cancel_op is None:
      self._cancel_op = self._queue.close(cancel_pending_enqueues=True)
    if not self._queue_closed_exception_types:
      self._queue_closed_exception_types = (errors.OutOfRangeError,)
    else:
      self._queue_closed_exception_types = tuple(
          self._queue_closed_exception_types)

  def _init_from_proto(self, queue_runner_def, import_scope=None):
    """Create a QueueRunner from `QueueRunnerDef`.

    Args:
      queue_runner_def: Optional `QueueRunnerDef` protocol buffer.
      import_scope: Optional `string`. Name scope to add.
    """
    assert isinstance(queue_runner_def, queue_runner_pb2.QueueRunnerDef)
    g = ops.get_default_graph()
    self._queue = g.as_graph_element(
        ops.prepend_name_scope(queue_runner_def.queue_name, import_scope))
    self._enqueue_ops = [g.as_graph_element(
        ops.prepend_name_scope(op, import_scope))
                         for op in queue_runner_def.enqueue_op_name]
    self._close_op = g.as_graph_element(ops.prepend_name_scope(
        queue_runner_def.close_op_name, import_scope))
    self._cancel_op = g.as_graph_element(ops.prepend_name_scope(
        queue_runner_def.cancel_op_name, import_scope))
    self._queue_closed_exception_types = tuple(
        errors.exception_type_from_error_code(code)
        for code in queue_runner_def.queue_closed_exception_types)
    # Legacy support for old QueueRunnerDefs created before this field
    # was added.
    if not self._queue_closed_exception_types:
      self._queue_closed_exception_types = (errors.OutOfRangeError,)

  @property
  def queue(self):
    return self._queue

  @property
  def enqueue_ops(self):
    return self._enqueue_ops

  @property
  def close_op(self):
    return self._close_op

  @property
  def cancel_op(self):
    return self._cancel_op

  @property
  def queue_closed_exception_types(self):
    return self._queue_closed_exception_types

  @property
  def exceptions_raised(self):
    """Exceptions raised but not handled by the `QueueRunner` threads.

    Exceptions raised in queue runner threads are handled in one of two ways
    depending on whether or not a `Coordinator` was passed to
    `create_threads()`:

    * With a `Coordinator`, exceptions are reported to the coordinator and
      forgotten by the `QueueRunner`.
    * Without a `Coordinator`, exceptions are captured by the `QueueRunner` and
      made available in this `exceptions_raised` property.

    Returns:
      A list of Python `Exception` objects.  The list is empty if no exception
      was captured.  (No exceptions are captured when using a Coordinator.)
    """
    return self._exceptions_raised

  @property
  def name(self):
    """The string name of the underlying Queue."""
    return self._queue.name

  # pylint: disable=broad-except
  def _run(self, sess, enqueue_op, coord=None):
    """Execute the enqueue op in a loop, close the queue in case of error.

    Args:
      sess: A Session.
      enqueue_op: The Operation to run.
      coord: Optional Coordinator object for reporting errors and checking
        for stop conditions.
    """
    decremented = False
    try:
      while True:
        if coord and coord.should_stop():
          break
        try:
          sess.run(enqueue_op)
        except self._queue_closed_exception_types:  # pylint: disable=catching-non-exception
          # This exception indicates that a queue was closed.
          with self._lock:
            self._runs_per_session[sess] -= 1
            decremented = True
            if self._runs_per_session[sess] == 0:
              try:
                sess.run(self._close_op)
              except Exception as e:
                # Intentionally ignore errors from close_op.
                logging.vlog(1, "Ignored exception: %s", str(e))
            return
    except Exception as e:
      # This catches all other exceptions.
      if coord:
        coord.request_stop(e)
      else:
        logging.error("Exception in QueueRunner: %s", str(e))
        with self._lock:
          self._exceptions_raised.append(e)
        raise
    finally:
      # Make sure we account for all terminations: normal or errors.
      if not decremented:
        with self._lock:
          self._runs_per_session[sess] -= 1


  def _close_on_stop(self, sess, cancel_op, coord):
    """Close the queue when the Coordinator requests stop.

    Args:
      sess: A Session.
      cancel_op: The Operation to run.
      coord: Coordinator.
    """
    coord.wait_for_stop()
    try:

        # TODO(maksym): the next two lines are the only difference from TF
        # sess.run(cancel_op)
        1 + 1
    except Exception as e:
      # Intentionally ignore errors from cancel_op.
      logging.vlog(1, "Ignored exception: %s", str(e))
  # pylint: enable=broad-except

  def create_threads(self, sess, coord=None, daemon=False, start=False):
    """Create threads to run the enqueue ops for the given session.

    This method requires a session in which the graph was launched.  It creates
    a list of threads, optionally starting them.  There is one thread for each
    op passed in `enqueue_ops`.

    The `coord` argument is an optional coordinator that the threads will use
    to terminate together and report exceptions.  If a coordinator is given,
    this method starts an additional thread to close the queue when the
    coordinator requests a stop.

    If previously created threads for the given session are still running, no
    new threads will be created.

    Args:
      sess: A `Session`.
      coord: Optional `Coordinator` object for reporting errors and checking
        stop conditions.
      daemon: Boolean.  If `True` make the threads daemon threads.
      start: Boolean.  If `True` starts the threads.  If `False` the
        caller must call the `start()` method of the returned threads.

    Returns:
      A list of threads.
    """
    with self._lock:
      try:
        if self._runs_per_session[sess] > 0:
          # Already started: no new threads to return.
          return []
      except KeyError:
        # We haven't seen this session yet.
        pass
      self._runs_per_session[sess] = len(self._enqueue_ops)
      self._exceptions_raised = []

    ret_threads = [threading.Thread(target=self._run, args=(sess, op, coord))
                   for op in self._enqueue_ops]
    if coord:
      ret_threads.append(threading.Thread(target=self._close_on_stop,
                                          args=(sess, self._cancel_op, coord)))

    started_threads = 0
    for t in ret_threads:
      if coord:
        coord.register_thread(t)
      if daemon:
        t.daemon = True
      if start:
        t.start()
        print "[thread_start:",started_threads,"]",
        started_threads+=1

    time.sleep(0.25)
    print "[all threads started. queue.size:",sess.run(self.queue.size()),"]"

    return ret_threads

  def to_proto(self, export_scope=None):
    """Converts this `QueueRunner` to a `QueueRunnerDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Returns:
      A `QueueRunnerDef` protocol buffer, or `None` if the `Variable` is not in
      the specified name scope.
    """
    if (export_scope is None or
        self.queue.name.startswith(export_scope)):
      queue_runner_def = queue_runner_pb2.QueueRunnerDef()
      queue_runner_def.queue_name = ops.strip_name_scope(
          self.queue.name, export_scope)
      for enqueue_op in self.enqueue_ops:
        queue_runner_def.enqueue_op_name.append(
            ops.strip_name_scope(enqueue_op.name, export_scope))
      queue_runner_def.close_op_name = ops.strip_name_scope(
          self.close_op.name, export_scope)
      queue_runner_def.cancel_op_name = ops.strip_name_scope(
          self.cancel_op.name, export_scope)
      queue_runner_def.queue_closed_exception_types.extend([
          errors.error_code_from_exception_type(cls)
          for cls in self._queue_closed_exception_types])
      return queue_runner_def
    else:
      return None

  @staticmethod
  def from_proto(queue_runner_def, import_scope=None):
    """Returns a `QueueRunner` object created from `queue_runner_def`."""
    return QueueRunner(queue_runner_def=queue_runner_def,
                       import_scope=import_scope)



def dequeue_all(sess,queue):
    try:
        while True:
            print ".",
            sess.run(queue.dequeue(), options=tf.RunOptions(timeout_in_ms=500))
    except tf.errors.DeadlineExceededError:
        print "[",sess.run(queue.size()), "]"

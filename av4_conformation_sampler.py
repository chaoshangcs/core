import time
import tensorflow as tf
import numpy as np
import av4_input
from av4_config import FLAGS
import av4_networks
import av4_utils
import av4_cost_functions



class SamplingAgent:
    """ Sampling Agent takes a single protein with a bound ligand and:
    1) samples many possible protein-ligand poses,
    2) samples many many camera views of the correct position
    3) Outputs a single batch of images for which the gradient is highest.
    """
    # TODO: variance option for positives and negatives
    # TODO: clustering
    # TODO: add VDW possibilities
    # TODO: RMSD + the fraction of native contacts
    # TODO: add fast reject for overlapping atoms
    # TODO: in the future be sure not to find poses for a similar ligand
    # TODO: leave a single binding site on the level of data preparation
    # TODO: create fast reject for overlapping atoms of the protein and ligand

    # TODO:
    #        # example 1
    #        # take all (do not remove middle) dig a hole in the landscape
    #        # batch is a training example
    # needs labels

    # TODO:
    #        # example 2
    #        # remove semi-correct conformations
    #        # batch is a training example

    # TODO:
    #       # example 3
    #       # remove semi-correct conformations
    #       # take many images of positive
    #       # image is a training example


    def __init__(self,
                 agent_name,
                 gpu_name,
                 training_queue,
                 side_pixels=FLAGS.side_pixels,
                 pixel_size=FLAGS.pixel_size,
                 batch_size=FLAGS.batch_size,
                 num_threads=FLAGS.num_threads,
                 shift_ranges = FLAGS.shift_ranges,
                 shift_deltas = FLAGS.shift_deltas,
                 initial_pose_evals = FLAGS.initial_pose_evals,
                 train_batch_init_poses = FLAGS.train_batch_init_poses,
                 train_batch_gen_poses = FLAGS.train_batch_gen_poses,
                 sess = FLAGS.main_session):

        # general parameters of the agent
        self.agent_name = agent_name
        self.sess = sess
        self.coord = tf.train.Coordinator()
        self._batch_size = batch_size

        # Set up affine transform queue with the ligand positions.
        lig_pose_tforms = tf.concat([av4_utils.identity_matrices(initial_pose_evals),
                                     av4_utils.exhaustive_affine_transform(shift_ranges,shift_deltas)],
                                    0)
        self._num_images_per_cycle = self.sess.run(tf.shape(lig_pose_tforms))[0]
        affine_tforms_queue = tf.FIFOQueue(capacity=self._num_images_per_cycle, dtypes=tf.float32, shapes=[4, 4])
        self._affine_tforms_queue_allowstop = affine_tforms_queue.enqueue_many(
            [av4_utils.identity_matrices(num_threads*3)])
        self._affine_tforms_queue_enq = affine_tforms_queue.enqueue_many(lig_pose_tforms)


        # Create assign options for receptor and ligand.
        lig_elem = tf.Variable([0],trainable=False, validate_shape=False)
        lig_coord = tf.Variable([[0.0,0.0,0.0]], trainable=False, validate_shape=False)
        rec_elem = tf.Variable([0],trainable=False, validate_shape=False)
        rec_coord = tf.Variable([[0.0,0.0,0.0]], trainable=False, validate_shape=False)
        self._tform = affine_tforms_queue.dequeue()
        tformed_lig_coord,lig_pose_tform = av4_utils.affine_transform(lig_coord, self._tform)

        self._lig_elem_plc = tf.placeholder(tf.int32)
        self._lig_coord_plc = tf.placeholder(tf.float32)
        self._rec_elem_plc = tf.placeholder(tf.int32)
        self._rec_coord_plc = tf.placeholder(tf.float32)

        self._ass_lig_elem = tf.assign(lig_elem, self._lig_elem_plc, validate_shape=False, use_locking=True)
        self._ass_lig_coord = tf.assign(lig_coord, self._lig_coord_plc, validate_shape=False, use_locking=True)
        self._ass_rec_elem = tf.assign(rec_elem, self._rec_elem_plc, validate_shape=False, use_locking=True)
        self._ass_rec_coord = tf.assign(rec_coord, self._rec_coord_plc, validate_shape=False, use_locking=True)

        self._lig_elem_shape = tf.shape(lig_elem)
        self._lig_coord_shape = tf.shape(lig_coord)
        self._rec_elem_shape = tf.shape(rec_elem)
        self._rec_coord_shape = tf.shape(rec_coord)


        # Create batches of images from coordinates and elements.
        complex_image,_,cameraview = av4_input.complex_coords_to_image(lig_elem,
                                                                       tformed_lig_coord,
                                                                       rec_elem,
                                                                       rec_coord,
                                                                       side_pixels,
                                                                       pixel_size)
        # calculate Root Mean Square Deviation for atoms of the transformed molecule compared to the initial one
        lig_RMSD = tf.reduce_mean(tf.square(tformed_lig_coord - lig_coord))**0.5
        # create and enqueue images in many threads, and deque and score images in a main thread
        image_queue = tf.FIFOQueue(capacity=batch_size*5,
                                   dtypes=[tf.float32,tf.float32,tf.float32,tf.float32],
                                   shapes=[[side_pixels,side_pixels,side_pixels], [4,4], [4,4], []])
        self._image_queue_deq = image_queue.dequeue()[0]
        self._image_queue_enq = image_queue.enqueue([complex_image, lig_pose_tform, cameraview, lig_RMSD])
        self._queue_runner = av4_utils.QueueRunner(image_queue, [self._image_queue_enq] * num_threads)
        self._image_b, self._lig_pose_tform_b, self._cameraview_b, self._lig_RMSD_b = image_queue.dequeue_many(batch_size)


        # Evaluate images with the network.
        self._keep_prob = tf.placeholder(tf.float32)
        if gpu_name is not None:
            with tf.device(gpu_name):
                with tf.name_scope("network"):
                    y_conv = av4_networks.max_net.compute_output(self._image_b, self._keep_prob, batch_size)
        # calculate both predictions, and costs for every ligand position in the batch
        self._pred_b = tf.nn.softmax(y_conv)[:,1]
        self._cost_b = av4_cost_functions.cross_entropy_with_RMSD(y_conv, self._lig_RMSD_b)


        # Regeneration of the images from affine transform matrices for training.
        # This part of the pipeline is for re-creation of already scored images from affine transform matrices
        # describing the conformation of the ligand, and a particular cameraview from which the 3D snapshot was taken.
        # Storing affine transform matrices instead of images allows to save memory while agent is running.
        r_tforms_cameraviews_queue = tf.FIFOQueue(capacity=train_batch_init_poses + train_batch_gen_poses,
                                                  dtypes=[tf.float32,tf.float32],
                                                  shapes=[[4,4],[4,4]])
        self._r_tforms_and_cameraviews_queue_allowstop = r_tforms_cameraviews_queue.enqueue_many(
            [av4_utils.identity_matrices(num_threads*3),
             av4_utils.identity_matrices(num_threads*3)])
        self._r_tforms_plc = tf.placeholder(tf.float32)
        self._r_cameraviews_plc = tf.placeholder(tf.float32)
        self._r_tforms_cameraviews_enq = r_tforms_cameraviews_queue.enqueue_many([self._r_tforms_plc,
                                                                                  self._r_cameraviews_plc])
        self._r_tform, self._r_cameraview = r_tforms_cameraviews_queue.dequeue()
        self._r_tformed_lig_coords, _ = av4_utils.affine_transform(lig_coord, self._r_tform)
        self._r_complex_image, _, _ = av4_input.complex_coords_to_image(lig_elem,
                                                                        self._r_tformed_lig_coords,
                                                                        rec_elem,
                                                                        rec_coord,
                                                                        side_pixels,
                                                                        pixel_size,
                                                                        self._r_cameraview)
        # put images back to the image queue
        self._r_lig_RMSD = tf.reduce_mean(tf.square(self._r_tformed_lig_coords - lig_coord))**0.5
        r_image_queue_enq = image_queue.enqueue([self._r_complex_image,
                                                 self._r_tform,
                                                 self._r_cameraview,
                                                 self._r_lig_RMSD])
        self._r_queue_runner = av4_utils.QueueRunner(image_queue, [r_image_queue_enq] * num_threads)
        # dequeue image batch from regenerator, and enque it to the training queue
        self.pass_batch_to_the_training_queue = training_queue.enqueue_many([self._image_b, self._lig_RMSD_b])


    def grid_evaluate_positions(self, my_lig_elem, my_lig_coord, my_rec_elem, my_rec_coord):
        """ Puts ligand in the center of every square of the box around the ligand, performs network evaluation of
        every conformation.
        """
        # Assign coordinates and elements of the ligand and protein. Add transformation matrices to the queue.
        # enqueue all of the transformations for evaluation cycle
        self.sess.run([self._affine_tforms_queue_enq])
        # assign elements and coordinates of protein and ligand;
        # shape of the variable will change from ligand to ligand
        self.sess.run([self._ass_lig_elem,self._ass_lig_coord, self._ass_rec_elem,self._ass_rec_coord],
                      feed_dict={self._lig_elem_plc:my_lig_elem, self._lig_coord_plc:my_lig_coord,
                                 self._rec_elem_plc:my_rec_elem, self._rec_coord_plc:my_rec_coord})
        # re-initialize the evalutions class
        evaluated = self.EvaluationsContainer(self._logf)
        self._logf("shapes of the ligand and protein:" + str(self.sess.run([self._lig_elem_shape,
                                                                            self._lig_coord_shape,
                                                                            self._rec_elem_shape,
                                                                            self._rec_coord_shape])))
        # Evaluate all of the images of ligand and protein complexes in batches.
        # start threads to fill the queue
        self.enqueue_thr = self._queue_runner.create_threads(self.sess, self._logf, self.coord, start=True, daemon=True)
        # evaluate batch of images
        num_batches_per_cycle = self._num_images_per_cycle // self._batch_size
        for i in range(num_batches_per_cycle):
            start = time.time()
            my_pred_b, my_cost_b, my_image_b, my_lig_pose_tform_b, my_cameraview_b, my_lig_RMSD_b = self.sess.run(
                [self._pred_b, self._cost_b, self._image_b, self._lig_pose_tform_b, self._cameraview_b, self._lig_RMSD_b],
                feed_dict = {self._keep_prob:1})
            # save the predictions and cameraviews from the batch into evaluations container
            lig_poses_evaluated = evaluated.add_batch(my_pred_b,
                                                      my_cost_b,
                                                      my_lig_pose_tform_b,
                                                      my_cameraview_b,
                                                      my_lig_RMSD_b)
            self._logf(self.agent_name, stdout=True)
            self._logf("\tligand_atoms:" + str(np.sum(np.array(my_image_b >7, dtype=np.int32))), True)
            self._logf("\tpositions evaluated:" + str(lig_poses_evaluated), True)
            self._logf("\texamples per second:" + str("%.2f" % (self._batch_size / (time.time() - start))) + "\n", True)


        # Select positively and negatively labeled images with the highest cost, and enqueue them into training queue
        sel_lig_tforms, sel_cameraviews = evaluated.get_training_batch(cameraviews_initial_pose=50,
                                                                       generated_poses=50,
                                                                       remember_poses=300)
        # accurately terminate all threads without closing the queue (uses custom QueueRunner class)
        self.coord.request_stop()
        self.sess.run(self._affine_tforms_queue_allowstop)
        self.coord.join()
        self.coord.clear_stop()
        av4_utils.dequeue_all(self.sess, self._tform, self._logf, "affine_tforms_queue")
        av4_utils.dequeue_all(self.sess, self._image_queue_deq, self._logf, "image_queue")
        # regenerate a selected batch of images using the same (empty) image queue
        # enqueue the regenerator
        self.sess.run([self._r_tforms_cameraviews_enq],
                      feed_dict={self._r_tforms_plc: sel_lig_tforms,
                                 self._r_cameraviews_plc: sel_cameraviews})
        # start threads to fill the regenerator queue
        self._r_enqueue_thr = self._r_queue_runner.create_threads(self.sess, self._logf, coord=self.coord, start=True, daemon=True)
        self.sess.run(self.pass_batch_to_the_training_queue)
        self.coord.request_stop()
        # accurately terminate all threads without closing the queue (using custom QueueRunner class)
        self.sess.run(self._r_tforms_and_cameraviews_queue_allowstop)
        self.coord.join()
        self.coord.clear_stop()
        av4_utils.dequeue_all(self.sess, self._r_tform, self._logf, "_r_tforms_and_cameraviews_queue")
        av4_utils.dequeue_all(self.sess,self._image_queue_deq, self._logf, "image_queue")
        return None

    class EvaluationsContainer:
        """" Groups together information about the evaluated positions in a form of affine transform matrices. Reduces
        all of the evaluated poses into training batch.
        """
        # TODO: add a possibility to generate new matrices based on performed evaluations.
        # (aka genetic search in AutoDock)
        def __init__(self,_logf):
            self._logf = _logf
            self.preds = np.array([])
            self.costs = np.array([])
            self.lig_pose_transforms = np.array([]).reshape([0, 4, 4])
            self.cameraviews = np.array([]).reshape([0, 4, 4])
            self.lig_RMSDs = np.array([])


        def add_batch(self, pred_batch, cost_batch, lig_pose_transform_batch, cameraview_batch, lig_RMSD_batch):
            """ Adds batch of predictions for different positions and cameraviews of the ligand.
            """
            self.preds = np.append(self.preds, pred_batch, axis=0)
            self.costs = np.append(self.costs, cost_batch, axis=0)
            self.lig_pose_transforms = np.append(self.lig_pose_transforms, lig_pose_transform_batch, axis=0)
            self.cameraviews = np.append(self.cameraviews, cameraview_batch, axis=0)
            self.lig_RMSDs = np.append(self.lig_RMSDs, lig_RMSD_batch, axis=0)
            return len(self.preds)


        def get_training_batch(self, cameraviews_initial_pose=10, generated_poses=90, remember_poses=300):
            """ Returns examples with a highest cost/gradient.

            [cameraviews_initial_pose] + [generated_poses] should be == to [desired batch size] for training
            """
            # order all parameters by costs in ascending order
            order = np.argsort(self.costs)
            self.preds = self.preds[order]
            self.costs = self.costs[order]
            self.lig_pose_transforms = self.lig_pose_transforms[order]
            self.cameraviews = self.cameraviews[order]
            self.lig_RMSDs = self.lig_RMSDs[order]

            # take examples for which the cost is highest
            init_poses_idx = (np.where(self.lig_RMSDs < 0.01)[0])
            gen_poses_idx = (np.where(self.lig_RMSDs > 0.01)[0])
            sel_init_poses_idx = init_poses_idx[-cameraviews_initial_pose:]
            sel_gen_poses_idx = gen_poses_idx[-generated_poses:]

            # print a lot of statistics for debugging/monitoring purposes
            self._logf("statistics sampled conformations:")
            var_list = {'lig_RMSDs':self.lig_RMSDs,'preds':self.preds,'costs':self.costs}
            self._logf(av4_utils.var_stats(var_list))
            self._logf("statistics for selected (hardest) initial conformations")
            var_list = {'lig_RMSDs':self.lig_RMSDs[sel_init_poses_idx], 'preds':self.preds[sel_init_poses_idx],
                        'costs':self.costs[sel_init_poses_idx]}
            self._logf(av4_utils.var_stats(var_list))
            self._logf("statistics for selected (hardest) generated conformations")
            var_list = {'lig_RMSDs':self.lig_RMSDs[sel_gen_poses_idx], 'preds':self.preds[sel_gen_poses_idx],
                        'costs':self.costs[sel_gen_poses_idx]}
            self._logf(av4_utils.var_stats(var_list))
            sel_idx = np.hstack([sel_init_poses_idx,sel_gen_poses_idx])
            return self.lig_pose_transforms[sel_idx],self.cameraviews[sel_idx]


    def _logf(self, message, stdout=False):
        log_file = (FLAGS.summaries_dir + "/" + FLAGS.run_name + "_logs/" + self.agent_name + ".log")
        with open(log_file, 'a') as fout:
            message = str(message)
            fout.write(message)
        if stdout:
            print message,
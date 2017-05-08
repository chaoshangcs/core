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
                 training_q,
                 logger,
                 side_pixels=FLAGS.side_pixels,
                 pixel_size=FLAGS.pixel_size,
                 batch_size=FLAGS.batch_size,
                 num_threads=FLAGS.num_threads,
                 shift_ranges = FLAGS.shift_ranges,
                 shift_deltas = FLAGS.shift_deltas,
                 initial_pose_evals = FLAGS.initial_pose_evals,
                 train_b_init_poses = FLAGS.train_batch_init_poses,
                 train_b_gen_poses = FLAGS.train_batch_gen_poses,
                 sess = FLAGS.main_session):

        # general parameters of the agent
        self.agent_name = agent_name
        self.sess = sess
        self.coord = tf.train.Coordinator()
        self._batch_size = batch_size
        self._train_b_init_poses = train_b_init_poses
        self._train_b_gen_poses = train_b_gen_poses
        self._logger = logger

        # Set up affine transform queue with the ligand positions.
        lig_pose_tforms = tf.concat([av4_utils.identity_matrices(initial_pose_evals),
                                     av4_utils.exhaustive_affine_transform(shift_ranges,shift_deltas)],
                                    0)
        self._num_images_per_cycle = self.sess.run(tf.shape(lig_pose_tforms))[0]
        affine_tforms_q = tf.FIFOQueue(capacity=self._num_images_per_cycle, dtypes=tf.float32, shapes=[4, 4])
        self._affine_tforms_q_helpstop = affine_tforms_q.enqueue_many([av4_utils.identity_matrices(num_threads*3)])
        self._affine_tforms_q_enq = affine_tforms_q.enqueue_many(lig_pose_tforms)


        # Create assign options for receptor and ligand.
        lig_elem = tf.Variable([0],trainable=False, validate_shape=False)
        lig_coord = tf.Variable([[0.0,0.0,0.0]], trainable=False, validate_shape=False)
        rec_elem = tf.Variable([0],trainable=False, validate_shape=False)
        rec_coord = tf.Variable([[0.0,0.0,0.0]], trainable=False, validate_shape=False)
        self._tform = affine_tforms_q.dequeue()
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
        image_q = tf.FIFOQueue(capacity=train_b_init_poses + train_b_gen_poses,
                               dtypes=[tf.float32,tf.float32,tf.float32,tf.float32],
                               shapes=[[side_pixels,side_pixels,side_pixels], [4,4], [4,4], []])
        self._image_q_deq = image_q.dequeue()[0]
        self._image_q_enq = image_q.enqueue([complex_image, lig_pose_tform, cameraview, lig_RMSD])
        self._q_runner = av4_utils.QueueRunner(image_q, [self._image_q_enq] * num_threads)
        self._image_b, self._lig_pose_tform_b, self._cameraview_b, self._lig_RMSD_b = image_q.dequeue_many(batch_size)


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
        r_tforms_cameraviews_q = tf.FIFOQueue(capacity=train_b_init_poses + train_b_gen_poses,
                                              dtypes=[tf.float32,tf.float32],
                                              shapes=[[4,4],[4,4]])
        r_image_q = tf.FIFOQueue(capacity=train_b_init_poses + train_b_gen_poses,
                                 dtypes=[tf.float32,tf.float32],
                                 shapes=[[side_pixels,side_pixels,side_pixels], []])
        self._r_image_b, self._r_lig_RMSD_b = r_image_q.dequeue_many(train_b_init_poses + train_b_gen_poses)

        # create an image regenerator pipeline
        self._r_tforms_and_cameraviews_q_helpstop = r_tforms_cameraviews_q.enqueue_many(
            [av4_utils.identity_matrices(num_threads*3), av4_utils.identity_matrices(num_threads*3)])
        self._r_tforms_plc = tf.placeholder(tf.float32)
        self._r_cameraviews_plc = tf.placeholder(tf.float32)
        self._r_tforms_cameraviews_enq = r_tforms_cameraviews_q.enqueue_many([self._r_tforms_plc,
                                                                              self._r_cameraviews_plc])
        self._r_tform, self._r_cameraview = r_tforms_cameraviews_q.dequeue()
        self._r_tformed_lig_coords, _ = av4_utils.affine_transform(lig_coord, self._r_tform)
        self._r_complex_image,_ ,_ = av4_input.complex_coords_to_image(lig_elem,
                                                                       self._r_tformed_lig_coords,
                                                                       rec_elem,
                                                                       rec_coord,
                                                                       side_pixels,
                                                                       pixel_size,
                                                                       self._r_cameraview)
        # put images back to the image queue
        self._r_lig_RMSD = tf.reduce_mean(tf.square(self._r_tformed_lig_coords - lig_coord))**0.5
        r_image_q_enq = r_image_q.enqueue([self._r_complex_image, self._r_lig_RMSD])
        self._r_q_runner = av4_utils.QueueRunner(r_image_q, [r_image_q_enq] * num_threads)
        # dequeue image batch from regenerator, and enque it to the training queue
        self.pass_batch_to_the_training_q = training_q.enqueue_many([self._r_image_b, self._r_lig_RMSD_b])


    def grid_evaluate_positions(self, my_lig_elem, my_lig_coord, my_rec_elem, my_rec_coord):
        """ Puts ligand in the center of every square of the box around the ligand, performs network evaluation of
        every conformation.

        Raises: exception when number of positions evaluated in cycle is smaller than the batch size
        """
        # Assign coordinates and elements of the ligand and protein. Add transformation matrices to the queue.
        # enqueue all of the transformations for evaluation cycle
        self.sess.run([self._affine_tforms_q_enq])
        # assign elements and coordinates of protein and ligand;
        # shape of the variable will change from ligand to ligand
        self.sess.run([self._ass_lig_elem,self._ass_lig_coord, self._ass_rec_elem,self._ass_rec_coord],
                      feed_dict={self._lig_elem_plc:my_lig_elem, self._lig_coord_plc:my_lig_coord,
                                 self._rec_elem_plc:my_rec_elem, self._rec_coord_plc:my_rec_coord})
        # re-initialize the evalutions class
        evaluated = self.EvaluationsContainer(self._logger)
        self._logger.info("shapes of the ligand and protein:" + str(self.sess.run([self._lig_elem_shape,
                                                                                   self._lig_coord_shape,
                                                                                   self._rec_elem_shape,
                                                                                   self._rec_coord_shape])))
        # Evaluate all of the images of ligand and protein complexes in batches.
        # start threads to fill the queue
        self.enq_thr = self._q_runner.create_threads(self.sess, self._logger, self.coord, start=True, daemon=True)

        # evaluate images in batches
        num_batches_per_cycle = self._num_images_per_cycle // self._batch_size
        if num_batches_per_cycle <= 0:
            raise Exception('number of grid points to evaluate too small for the batch size')
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
            # write run information to the log file
            self._logger.info("\tligand_atoms:" + str(np.sum(np.array(my_image_b >7, dtype=np.int32))))
            self._logger.info("\tpositions evaluated:" + str(lig_poses_evaluated))
            self._logger.info("\texamples per second:" + str("%.2f" % (self._batch_size / (time.time() - start))) + "\n")


        # Select positively and negatively labeled images with the highest cost, and enqueue them into training queue
        sel_lig_tforms, sel_cameraviews = evaluated.get_training_batch(cameraviews_initial_pose=self._train_b_init_poses,
                                                                       generated_poses=self._train_b_gen_poses)
        # accurately terminate all threads without closing the queue (uses custom QueueRunner class)
        self.coord.request_stop()
        self.sess.run(self._affine_tforms_q_helpstop)
        self.coord.join()
        self.coord.clear_stop()
        av4_utils.dequeue_all(self.sess, self._tform, self._logger, "affine_tforms_queue")
        av4_utils.dequeue_all(self.sess, self._image_q_deq, self._logger, "image_queue")
        # regenerate a selected batch of images using the same (empty) image queue
        # enqueue the regenerator
        self.sess.run([self._r_tforms_cameraviews_enq],
                      feed_dict={self._r_tforms_plc: sel_lig_tforms, self._r_cameraviews_plc: sel_cameraviews})
        # start threads to fill the regenerator queue
        self._r_enq_thr = self._r_q_runner.create_threads(self.sess, self._logger, coord=self.coord, start=True, daemon=True)
        self.sess.run(self.pass_batch_to_the_training_q)
        self.coord.request_stop()
        # accurately terminate all threads without closing the queue (using custom QueueRunner class)
        self.sess.run(self._r_tforms_and_cameraviews_q_helpstop)
        self.coord.join()
        self.coord.clear_stop()
        av4_utils.dequeue_all(self.sess, self._r_tform, self._logger, "_r_tforms_and_cameraviews_queue")
        av4_utils.dequeue_all(self.sess, self._image_q_deq, self._logger, "image_queue")
        return None

    class EvaluationsContainer:
        """" Groups together information about the evaluated positions in a form of affine transform matrices. Reduces
        all of the evaluated poses into training batch.
        """
        # TODO: add a possibility to generate new matrices based on performed evaluations.
        # (aka genetic search in AutoDock)
        def __init__(self, logger):
            self._logger = logger
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


        def get_training_batch(self, cameraviews_initial_pose, generated_poses):
            """ Returns examples with a highest cost/gradient.
            [cameraviews_initial_pose] + [generated_poses] should be == to [desired batch size] for training

            Raises: exception when the number of images requested for the training batch is higher then the number of
            evaluated images in container.
            """

            if not (len(self.preds) == len(self.costs) == len(self.lig_pose_transforms) == len(self.cameraviews)
                        == len(self.lig_RMSDs)):
                raise Exception('Number of records for different categories in EvauationsContainer differs.')

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
            if len(init_poses_idx) < cameraviews_initial_pose:
                raise Exception('Number of ligand initial poses requested is more than EvaluationsContainer has')
            if len(gen_poses_idx) < generated_poses:
                raise Exception('Number of ligand generated poses requested is more than EvaluationsContainer has')

            sel_init_poses_idx = init_poses_idx[-cameraviews_initial_pose:]
            sel_gen_poses_idx = gen_poses_idx[-generated_poses:]

            # print a lot of statistics for debugging/monitoring purposes
            self._logger.info("statistics sampled conformations:")
            var_list = {'lig_RMSDs':self.lig_RMSDs,'preds':self.preds,'costs':self.costs}
            self._logger.info(av4_utils.var_stats(var_list))
            self._logger.info("statistics for initial conformations")
            var_list = {'lig_RMSDs':self.lig_RMSDs[init_poses_idx], 'preds':self.preds[init_poses_idx],
                        'costs':self.costs[init_poses_idx]}
            self._logger.info(av4_utils.var_stats(var_list))
            self._logger.info("statistics for generated conformations")
            var_list = {'lig_RMSDs':self.lig_RMSDs[gen_poses_idx], 'preds':self.preds[gen_poses_idx],
                        'costs':self.costs[gen_poses_idx]}
            self._logger.info(av4_utils.var_stats(var_list))
            self._logger.info("statistics for selected (hardest) initial conformations")
            var_list = {'lig_RMSDs':self.lig_RMSDs[sel_init_poses_idx], 'preds':self.preds[sel_init_poses_idx],
                        'costs':self.costs[sel_init_poses_idx]}
            self._logger.info(av4_utils.var_stats(var_list))
            self._logger.info("statistics for selected (hardest) generated conformations")
            var_list = {'lig_RMSDs':self.lig_RMSDs[sel_gen_poses_idx], 'preds':self.preds[sel_gen_poses_idx],
                        'costs':self.costs[sel_gen_poses_idx]}
            self._logger.info(av4_utils.var_stats(var_list))
            sel_idx = np.hstack([sel_init_poses_idx, sel_gen_poses_idx])
            return self.lig_pose_transforms[sel_idx], self.cameraviews[sel_idx]
import time
import tensorflow as tf
import numpy as np
import av4_input
from av4_main import FLAGS
import av4_networks
import av4_utils


def softmax_cross_entropy_with_RMSD(logits,lig_RMSDs,RMSD_threshold=3.0):
    """Calculates usual sparse softmax cross entropy for two class classification between 1(correct position)
    and 0(incorrect position) and multiplies the resulting cross entropy by RMSD coefficient.
    | RMSD_ligand > RMSD_threshold | RMSDcoeff = 0
    | RMSD_ligand < RMSD_threshold | RMSDcoeff = (RMSD_threshold - RMSD_ligand)/RMSD_threshold
    RMSD threshold is in Angstroms.
    """
    labels = tf.cast((lig_RMSDs < 0.01), tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
    cost_correct_positions = cross_entropy * tf.cast(labels,tf.float32)
    cost_incorrect_positions = cross_entropy * tf.cast((lig_RMSDs > RMSD_threshold), tf.float32)
    cost_semicorrect_positions = cross_entropy \
                                 * tf.cast((lig_RMSDs < RMSD_threshold), tf.float32) \
                                 * tf.cast((lig_RMSDs > 0.01), tf.float32) * (lig_RMSDs/RMSD_threshold)
    return cost_incorrect_positions + cost_semicorrect_positions + cost_correct_positions


class SearchAgent:
    """ Search agent takes a single protein with a bound ligand, samples many possible protein-ligand conformations,
    as well as many camera views of the correct position, and outputs a single batch of images and labels for training
    on which network makes biggest error, and gradient is highest.
    Search agent also outputs a compressed form (affine transform matrices) of other conformations that would make good
    training examples, but did not make it to the batch.
    """
    # TODO: variance option for positives and negatives
    # TODO: clustering
    # TODO: add VDW possibilities
    # TODO: RMSD + the fraction of native contacts
    # TODO: add fast reject for overlapping atoms
    # TODO: in the future be sure not to find poses for a similar ligand
    # TODO: leave a single binding site on the level of data preparation

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


    class EvaluationsContainer:
        """" Groups together information about the evaluated positions in a form of affine transform matrices. Reduces
        all of the evaluated poses into training batch.
        """
        # TODO: add a possibility to generate new matrices based on performed evaluations.
        # (aka genetic search in AutoDock)

        def __init__(self):
            self.preds = np.array([])
            self.costs = np.array([])
            self.lig_pose_transforms = np.array([]).reshape([0, 4, 4])
            self.cameraviews = np.array([]).reshape([0, 4, 4])
            self.lig_RMSDs = np.array([])


        def add_batch(self, pred_batch, cost_batch, lig_pose_transform_batch, cameraview_batch, lig_RMSD_batch):
            """ Adds batch of predictions for different positions and cameraviews of the ligand.
            """
            self.preds = np.append(self.preds, pred_batch, axis=0)
            self.costs = np.append(self.costs,cost_batch, axis=0)
            self.lig_pose_transforms = np.append(self.lig_pose_transforms, lig_pose_transform_batch, axis=0)
            self.cameraviews = np.append(self.cameraviews, cameraview_batch, axis=0)
            self.lig_RMSDs = np.append(self.lig_RMSDs, lig_RMSD_batch, axis=0)
            return len(self.preds)


        def convert_into_training_batch(self, cameraviews_initial_pose=10, generated_poses=90, remember_poses=300):
            """ Takes predictions for which the label * prediction is highest
            (label * prediction is equal to the cost to the network)
            constructs num_batches of bathes with a single positive example and returns a list of batches

            [cameraviews_initial_pose] + [generated_poses] should be == to [desired batch size] for training
            """
            # order all parameters by costs in ascending order
            order = np.argsort(self.costs)
            self.preds = self.preds[order]
            self.costs = self.costs[order]
            self.lig_pose_transforms = self.lig_pose_transforms[order]
            self.cameraviews = self.cameraviews[order]
            self.lig_RMSDs = self.lig_RMSDs[order]

            # Take examples for which the cost is highest.
            # The arbitrary number of RMSD of 0.01 which distinguishes correct from incorrect examples should not
            # affect the results at all. Positions with small ligand_RMSDs < RMSD_threshold will only become training
            # examples after no (ligand_RMSD > RMSD_threshold) is left. It is same with sliding along the threshold.
            init_poses_idx = (np.where(self.lig_RMSDs < 0.01)[0])
            gen_poses_idx = (np.where(self.lig_RMSDs > 0.01)[0])
            sel_init_poses_idx = init_poses_idx[-cameraviews_initial_pose:]
            sel_gen_poses_idx = gen_poses_idx[-generated_poses:]

            # print a lot of statistics for debugging/monitoring purposes
            print "statistics sampled conformations:"
            var_list = {'lig_RMSDs':self.lig_RMSDs,'preds':self.preds,'costs':self.costs}
            av4_utils.describe_variables(var_list)
            print "statistics for selected (hardest) initial conformations"
            var_list = {'lig_RMSDs':self.lig_RMSDs[sel_init_poses_idx], 'preds':self.preds[sel_init_poses_idx],
                        'costs':self.costs[sel_init_poses_idx]}
            av4_utils.describe_variables(var_list)
            print "statistics for selected (hardest) generated conformations"
            var_list = {'lig_RMSDs':self.lig_RMSDs[sel_gen_poses_idx], 'preds':self.preds[sel_gen_poses_idx],
                        'costs':self.costs[sel_gen_poses_idx]}
            av4_utils.describe_variables(var_list)
            sel_idx = np.hstack([sel_init_poses_idx,sel_gen_poses_idx])
            return self.lig_pose_transforms[sel_idx],self.cameraviews[sel_idx]


    def __init__(self,
                 agent_name,
                 side_pixels=FLAGS.side_pixels,
                 pixel_size=FLAGS.pixel_size,
                 batch_size=FLAGS.batch_size,
                 num_threads=FLAGS.num_threads,
                 sess = FLAGS.main_session):
        # Generate a single ligand position from initial coordinates and a given transformation matrix.
        self.agent_name = agent_name
        self.sess = sess
        self.num_threads = num_threads
        self._affine_tforms_queue = tf.FIFOQueue(capacity=80000, dtypes=tf.float32,shapes=[4, 4])

        self.lig_pose_tforms = tf.concat([av4_utils.generate_identity_matrices(1000),
                                          av4_utils.generate_exhaustive_affine_transform()],
                                         0)
        self.identity_matrices = av4_utils.generate_identity_matrices(self.num_threads * 3)

        self.lig_elements = tf.Variable([0],trainable=False, validate_shape=False)
        self.lig_coords = tf.Variable([[0.0,0.0,0.0]], trainable=False, validate_shape=False)
        self.rec_elements = tf.Variable([0],trainable=False, validate_shape=False)
        self.rec_coords = tf.Variable([[0.0,0.0,0.0]], trainable=False, validate_shape=False)
        tformed_lig_coords,lig_pose_tform = av4_utils.affine_transform(self.lig_coords,self._affine_tforms_queue.dequeue())
        # TODO: create fast reject for overlapping atoms of the protein and ligand

        # convert coordinates of the protein and ligand into an image
        complex_image,_,cameraview = av4_input.convert_protein_and_ligand_to_image(self.lig_elements,
                                                                                   tformed_lig_coords,
                                                                                   self.rec_elements,
                                                                                   self.rec_coords,
                                                                                   side_pixels,
                                                                                   pixel_size)

        # calculate Root Mean Square Deviation for atoms of the transformed molecule compared to the initial one
        lig_RMSD = tf.reduce_mean(tf.square(tformed_lig_coords - self.lig_coords))**0.5

        # create and enqueue images in many threads, and deque and score images in a main thread
        self.coord = tf.train.Coordinator()
        self.image_queue = tf.FIFOQueue(capacity=batch_size*5,
                                         dtypes=[tf.float32,tf.float32,tf.float32,tf.float32],
                                         shapes=[[side_pixels,side_pixels,side_pixels], [4,4], [4,4], []])

        self.image_queue_enqueue = self.image_queue.enqueue([complex_image, lig_pose_tform, cameraview, lig_RMSD])
        self.queue_runner = av4_utils.QueueRunner(self.image_queue, [self.image_queue_enqueue]*num_threads)
        self.image_batch,self.lig_pose_tform_batch, self.cameraview_batch,self.lig_RMSD_batch = self.image_queue.dequeue_many(batch_size)
        self.keep_prob = tf.placeholder(tf.float32)
        with tf.name_scope("network"):
            y_conv = av4_networks.max_net(self.image_batch, self.keep_prob, batch_size)

        # calculate both predictions, and costs for every ligand position in the batch
        self.pred_batch = tf.nn.softmax(y_conv)[:,1]
        self.cost_batch = softmax_cross_entropy_with_RMSD(y_conv,self.lig_RMSD_batch)

        # REGENERATOR: r_
        # This part of the pipeline is for re-creation of already scored images from affine transform matrices
        # describing the conformation of ligand, and a particular cameraview from which the 3D snapshot was taken.
        # Double pipeline allows to store only transformation matrices instead of images - needs less memory.

        # REGENERATOR: create queue and enque pipe with two placeholders
        self._tforms_and_cameraviews_queue = tf.FIFOQueue(capacity=80000, dtypes=[tf.float32,tf.float32],shapes=[[4,4],[4,4]])
        self.r_tforms_enq = tf.placeholder(tf.float32)
        self.r_cameraviews_enq = tf.placeholder(tf.float32)
        self.r_tforms_cameraviews_enque = self._tforms_and_cameraviews_queue.enqueue_many([self.r_tforms_enq,
                                                                                           self.r_cameraviews_enq])
        # REGENERATOR: create deque and image generation pipeline
        self.r_tform,self.r_cameraview = self._tforms_and_cameraviews_queue.dequeue()
        self.r_tformed_lig_coords,_ = av4_utils.affine_transform(self.lig_coords, self.r_tform)
        self.r_complex_image,_,_ = av4_input.convert_protein_and_ligand_to_image(self.lig_elements,
                                                                                 self.r_tformed_lig_coords,
                                                                                 self.rec_elements,
                                                                                 self.rec_coords,
                                                                                 side_pixels,
                                                                                 pixel_size,
                                                                                 self.r_cameraview)
        # REGENERATOR: put images back to the image queue
        self.r_lig_RMSD = tf.reduce_mean(tf.square(self.r_tformed_lig_coords - self.lig_coords))**0.5
        self.r_image_queue_enqueue = self.image_queue.enqueue([self.r_complex_image,self.r_tform,self.r_cameraview,self.r_lig_RMSD])
        self.r_queue_runner = av4_utils.QueueRunner(self.image_queue, [self.r_image_queue_enqueue] * num_threads)


    def grid_evaluate_positions(self,my_lig_elements,my_lig_coords,my_rec_elements,my_rec_coords):
        """ Puts ligand in the center of every square of the box around the ligand, performs network evaluation of
        every conformation.
        """
        # Enqueue all of the transformations for the ligand to sample.
        self.sess.run(self._affine_tforms_queue.enqueue_many(self.lig_pose_tforms))

        # Assign elements and coordinates of protein and ligand; shape of the variable will change from ligand to ligand
        self.sess.run([tf.assign(self.lig_elements,my_lig_elements, validate_shape=False, use_locking=True),
                       tf.assign(self.lig_coords,my_lig_coords, validate_shape=False, use_locking=True),
                       tf.assign(self.rec_elements,my_rec_elements, validate_shape=False, use_locking=True),
                       tf.assign(self.rec_coords,my_rec_coords, validate_shape=False, use_locking=True)])


        # re-initialize the evalutions class
        evaluated = self.EvaluationsContainer()

        print "shapes of the ligand and protein:"
        print self.sess.run([tf.shape(self.lig_elements),
                             tf.shape(self.lig_coords),
                             tf.shape(self.rec_elements),
                             tf.shape(self.rec_coords)])

        # start threads to fill the queue
        print "starting threads for the conformation sampler."
        self.enqueue_threads = self.queue_runner.create_threads(self.sess, coord=self.coord, start=True, daemon=True)

        try:
            while True:
                start = time.time()
                my_pred_batch, my_cost_batch, my_image_batch, my_lig_pose_tform_batch, my_cameraview_batch, my_lig_RMSD_batch = \
                    self.sess.run([self.pred_batch,
                                   self.cost_batch,
                                   self.image_batch,
                                   self.lig_pose_tform_batch,
                                   self.cameraview_batch,
                                   self.lig_RMSD_batch],
                                  feed_dict = {self.keep_prob:1},
                                  options=tf.RunOptions(timeout_in_ms=1000))
                # save the predictions and cameraviews from the batch into evaluations container
                lig_poses_evaluated = evaluated.add_batch(my_pred_batch,
                                                               my_cost_batch,
                                                               my_lig_pose_tform_batch,
                                                               my_cameraview_batch,
                                                               my_lig_RMSD_batch)

                print self.agent_name,
                print "\tligand_atoms:",np.sum(np.array(my_image_batch >7,dtype=np.int32)),
                print "\tpositions evaluated:",lig_poses_evaluated,
                print "\texamples per second:", "%.2f" % (FLAGS.batch_size / (time.time() - start))

        except tf.errors.DeadlineExceededError:
            # create training examples for the main queue
            sel_lig_pose_transforms,sel_cameraviews = evaluated.convert_into_training_batch(
                cameraviews_initial_pose=20,generated_poses=200,remember_poses=300)


            # accurately terminate all threads without closing the queue (uses custom QueueRunner class)
            self.sess.run(self._affine_tforms_queue.enqueue_many(self.identity_matrices))
            self.coord.request_stop()
            self.coord.join()
            self.coord.clear_stop()

            # remove all affine transform matrices from the queue to be empty before the next protein/ligand
            av4_utils.dequeue_all(self.sess,self._affine_tforms_queue)
            av4_utils.dequeue_all(self.sess,self.image_queue)

            # regenerate a selected batch of images from ligand transformations and cameraviews
            # enqueue the REGENERATOR
            self.sess.run([self.r_tforms_cameraviews_enque],
                          feed_dict={self.r_tforms_enq: sel_lig_pose_transforms,
                                     self.r_cameraviews_enq: sel_cameraviews})

            # start threads to fill the REGENERATOR queue
            self.r_enqueue_threads = self.r_queue_runner.create_threads(self.sess, coord=self.coord, start=True, daemon=True)

            # dequeue the REGENERATOR batch
            my_r_image_batch = self.sess.run([self.image_batch],
                                             feed_dict={self.keep_prob: 1}, options=tf.RunOptions(timeout_in_ms=1000))

            # TODO: stilll .................
            print my_r_image_batch

            # accurately terminate all threads without closing the queue (uses custom QueueRunner class)
            self.sess.run(self._tforms_and_cameraviews_queue.enqueue_many([self.identity_matrices,self.identity_matrices]))
            self.coord.request_stop()
            self.coord.join()
            self.coord.clear_stop()

            # remove all affine transform matrices from the queue to be empty before the next protein/ligand
            av4_utils.dequeue_all(self.sess,self._tforms_and_cameraviews_queue)
            av4_utils.dequeue_all(self.sess,self.image_queue)
        return None

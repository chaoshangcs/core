import time,re,threading
import tensorflow as tf
import numpy as np
import av4_input
import av4_networks
from av4_main import FLAGS
import av4_conformation_sampler


FLAGS.saved_session = './summaries/4_netstate/saved_state-82099'
FLAGS.predictions_file_path = re.sub("netstate","logs",FLAGS.saved_session)
FLAGS.database_path = '../datasets/holdout_av4'
FLAGS.num_epochs = 110
FLAGS.top_k = FLAGS.num_epochs



class GradientDescendMachine:
    """ Does sompling and training
    Controls sampling, infinite, or timely. Potentially, with many GPUs.
    Dependencies: FLAGS.examples_in_database should be already calculated
    """
    def __init__(self,side_pixels=FLAGS.side_pixels,batch_size=FLAGS.batch_size):

        # create session to compute evaluation
        self.sess = FLAGS.main_session

        # create a filename queue first
        filename_queue,self.ex_in_database = av4_input.index_the_database_into_queue(FLAGS.database_path, shuffle=True)

        # read receptor and ligand from the queue
        self.lig_file,_,_,self.lig_elements,self.lig_coords,self.rec_elements,self.rec_coords = \
            av4_input.read_receptor_and_ligand(filename_queue=filename_queue, epoch_counter=tf.constant(0))

        # create a very large queue of images for central parameter server
        self.traning_queue = tf.FIFOQueue(capacity=1000000,dtypes=[tf.float32],shapes=[side_pixels,side_pixels,side_pixels])

        # create a way to evaluate these images with the network

        # create saver to save and load the network state
#        self.sess.run(tf.global_variables_initializer())
#        saver = tf.train.Saver(var_list=(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Adam_optimizer")
#                                         + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="network")))
#                                         #+ tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="epoch_counter")))
#
#        if FLAGS.saved_session is None:
#            self.sess.run(tf.global_variables_initializer())
#        else:
#            self.sess.run(tf.global_variables_initializer())
#            print "Restoring variables from sleep. This may take a while..."
#            saver.restore(self.sess,FLAGS.saved_session)
#            print "unitialized vars:", self.sess.run(tf.report_uninitialized_variables())

        self.ag1 = av4_conformation_sampler.SearchAgent("AG1", self.traning_queue)
        self.ag2 = av4_conformation_sampler.SearchAgent("AG2", self.traning_queue)

        self.coord = tf.train.Coordinator()
#        threads = tf.train.start_queue_runners(sess = self.sess,coord=coord)


    def do_sampling(self, sample_epochs=None):
        """ COntrols all of the sampling by multiple agents
        """
        self.run_samples = None

        if sample_epochs is not None:
            self.run_samples = self.ex_in_database * sample_epochs

        def lig_rec_elems_coords():
            " Takes elements and coordinates from the next receptor and ligand"
            my_lig_file, my_lig_elements, my_lig_coords, my_rec_elements, my_rec_coords = self.sess.run(
                [self.lig_file, self.lig_elements, self.lig_coords, self.rec_elements, self.rec_coords])
            print "ligand file runs for sampling:", my_lig_file
            return my_lig_elements, my_lig_coords, my_rec_elements, my_rec_coords

        def pose_samplers_stop():
            if (sample_epochs is None) and (self.run_samples is None):
                return False
            elif (sample_epochs is not None) and (self.run_samples > 0):
                self.run_samples = self.run_samples - 1
                return False
            elif (sample_epochs is not None) and (self.run_samples <= 0):
                return True
            else:
                raise RuntimeError('either both or neither self.run_samples, self.sample_epochs should be None')

        def search_agent_1():
            # with TF. device GPU1
            # create an instance of a search agent that will run on this GPU
            while not pose_samplers_stop():
                # def grid_evaluate_positions(self, my_lig_elements, my_lig_coords, my_rec_elements, my_rec_coords):
                lig_elems, lig_coords, rec_elems, rec_coords = lig_rec_elems_coords()
 #               self.ag1.grid_evaluate_positions(lig_elems, lig_coords, rec_elems, rec_coords)

        def search_agent_2():
            # with TF. device GPU2
            # create an instance of a search agent that will run on this GPU
            while not pose_samplers_stop():
                lig_elems, lig_coords, rec_elems, rec_coords = lig_rec_elems_coords()
#                self.ag2.grid_evaluate_positions(lig_elems, lig_coords, rec_elems, rec_coords)


        # Only in this order 1# Initialize Variables 2# Start threads
        self.sess.run(tf.global_variables_initializer())
        tf.get_default_graph().finalize()

        threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        t1 = threading.Thread(target=search_agent_1).start()
        t2 = threading.Thread(target=search_agent_2).start()

        tf.logging.set_verbosity(tf.logging.DEBUG)


        while True:
 #           print "examples in the training queue", self.sess.run(self.traning_queue.size())
            print "examples in the main queue - unknown"
            time.sleep(1)






#Class TrainingController:
#    def __init__(self):
#        for i in range(100):
#            print "doing training"


#coord = tf.train.Coordinator()
#threads = tf.train.start_queue_runners(sess =FLAGS.main_session,coord=coord)

a = GradientDescendMachine()
a.do_sampling(sample_epochs=10)

#search_agent1 = av4_conformation_sampler.SearchAgent()

print "All Done"
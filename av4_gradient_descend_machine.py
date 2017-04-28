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



class SamplingAgentonGPU:

    def __init__(self, agent_name, gpu_name, filename_queue, sampling_coord, training_queue, sess):
        self.sampling_coord = sampling_coord
        self.sess = sess
        self.filename_queue = filename_queue
        self.ag = av4_conformation_sampler.SamplingAgent(agent_name,gpu_name,training_queue)

        self.lig_file, _, _, self.lig_elem, self.lig_coord, self.rec_elem, self.rec_coord = \
            av4_input.read_receptor_and_ligand(filename_queue=self.filename_queue, epoch_counter=tf.constant(0))  # FIXME: change epoch counter

        print "SamplingAgentonGPU:",agent_name,"successfully initialized on device:",gpu_name

    def _count_example(self):
        if (self.sampling_coord.run_samples is not None) and (self.sampling_coord.run_samples > 0):
            with self.sampling_coord.lock:
                self.sampling_coord.run_samples = self.sampling_coord.run_samples - 1
            return None
        elif (self.sampling_coord.run_samples is not None) and (self.sampling_coord.run_samples <= 0):
            self.sampling_coord.request_stop()
            return None
        else:
            return None

    def _do_sampling(self):
        # with TF. device GPU1
        # create an instance of a search agent that will run on this GPU
        while not self.sampling_coord.should_stop():
            try:
                # read receptor and ligand from the queue
                # evaluate all positions for this ligand and receptor
                self.ag.grid_evaluate_positions(*self.sess.run([self.lig_elem,self.lig_coord,self.rec_elem,self.rec_coord]))
                print "next", time.sleep(0.01)
                self._count_example()

            except Exception as ex:
                self.sampling_coord.request_stop(ex=ex)
        return None

    def start(self):
        # start a thread for this agent
        tr = threading.Thread(target=self._do_sampling)
        self.sampling_coord.threads.append(tr)
        tr.start()



class GradientDescendMachine:
    """ Does sompling and training
    Controls sampling, infinite, or timely. Potentially, with many GPUs.
    Dependencies: FLAGS.examples_in_database should be already calculated
    """
    def __init__(self,side_pixels=FLAGS.side_pixels,batch_size=FLAGS.batch_size):

        # try to capture all of the events that happen in many background threads
        tf.logging.set_verbosity(tf.logging.DEBUG)

        # create session to compute evaluation
        self.sess = FLAGS.main_session

        # create a filename queue first
        filename_queue,self.ex_in_database = av4_input.index_the_database_into_queue(FLAGS.database_path, shuffle=True)

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


        # configure sampling
        self.sampling_coord = tf.train.Coordinator()
        self.sampling_coord.lock = threading.Lock()

        self.ag1 = SamplingAgentonGPU("AG1","/gpu:0",filename_queue, self.sampling_coord, self.traning_queue, self.sess )



    def do_sampling(self, sample_epochs=None):
        """ Controls all of the sampling by multiple agents
        """

        self.sampling_coord.threads = []
        if sample_epochs is None:
            self.sampling_coord.run_samples = None
        else:
            self.sampling_coord.run_samples = self.ex_in_database * sample_epochs


        self.sess.run(tf.global_variables_initializer())
        tf.get_default_graph().finalize()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=self.sampling_coord)

        self.ag1.start()

        # in continuous regime continue without stopping the threads
        if sample_epochs is not None:
            self.sampling_coord.join(self.sampling_coord.threads)
            self.sampling_coord.clear_stop()

        return None






a = GradientDescendMachine()
a.do_sampling(sample_epochs=1)


print "All Done"
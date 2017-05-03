import time,re,threading
import tensorflow as tf
import numpy as np
import av4_input
import av4_networks
from av4_config import FLAGS
import av4_conformation_sampler
import av4_cost_functions


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

        # global_step is the total number steps of gradient descent (number of batches)
        self.global_step = 0

        # create a filename queue first
        filename_queue,self.ex_in_database = av4_input.index_the_database_into_queue(FLAGS.database_path, shuffle=True)

        # create a very large queue of images for central parameter server
        self.training_queue = tf.RandomShuffleQueue(capacity=1000000,min_after_dequeue=40000,dtypes=[tf.float32,tf.float32],shapes=[[side_pixels,side_pixels,side_pixels],[]])
        self.training_queue_size = self.training_queue.size()
        tf.summary.scalar("training queue size",self.training_queue_size)

        # create a way to train a network
        image_batch,lig_RMSD_batch = self.training_queue.dequeue_many(100)
        self.keep_prob = tf.placeholder(tf.float32)

        with tf.name_scope("network"):
            logits = av4_networks.max_net.compute_output(image_batch, self.keep_prob, FLAGS.batch_size)


        self.cost = av4_cost_functions.cross_entropy_with_RMSD(logits=logits,lig_RMSDs=lig_RMSD_batch)
        tf.summary.scalar('cost',tf.reduce_mean(self.cost))

        with tf.name_scope("Adam_optimizer"):
            self.train_step_run = tf.train.AdamOptimizer(1e-4).minimize(self.cost)



        # configure sampling
        self.sampling_coord = tf.train.Coordinator()
        self.sampling_coord.lock = threading.Lock()

        self.ag0 = SamplingAgentonGPU("AG1","/gpu:0",filename_queue, self.sampling_coord, self.training_queue, self.sess)
        #self.ag1 = SamplingAgentonGPU("AG2", "/gpu:1", filename_queue, self.sampling_coord, self.training_queue,self.sess)
        #self.ag2 = SamplingAgentonGPU("AG3", "/gpu:2", filename_queue, self.sampling_coord, self.training_queue,self.sess)
        #self.ag3 = SamplingAgentonGPU("AG4", "/gpu:3", filename_queue, self.sampling_coord, self.training_queue, self.sess)
        #self.ag4 = SamplingAgentonGPU("AG5", "/gpu:4", filename_queue, self.sampling_coord, self.training_queue, self.sess)
        #self.ag5 = SamplingAgentonGPU("AG6", "/gpu:5", filename_queue, self.sampling_coord, self.training_queue, self.sess)


        # merge all summaries and create a file writer object
        self.merged_summaries = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter((FLAGS.summaries_dir + '/' + str(FLAGS.run_name) + "_logs"), self.sess.graph)


        # create saver to save and load the network state
        self.saver = tf.train.Saver(var_list=(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Adam_optimizer")
                                         + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="network")))
                                         #+ tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="epoch_counter")))

        if FLAGS.saved_session is None:
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess.run(tf.global_variables_initializer())
            print "Restoring variables from sleep. This may take a while..."
            self.saver.restore(self.sess,FLAGS.saved_session)
            print "unitialized vars:", self.sess.run(tf.report_uninitialized_variables())
        # do not allow to add any nodes to the graph after this point
        tf.get_default_graph().finalize()

    def do_sampling(self, sample_epochs=None):
        """ Controls all of the sampling by multiple agents
        """

        self.sampling_coord.threads = []
        if sample_epochs is None:
            self.sampling_coord.run_samples = None
        else:
            self.sampling_coord.run_samples = self.ex_in_database * sample_epochs
        threads = tf.train.start_queue_runners(sess=self.sess, coord=self.sampling_coord)

        self.ag0.start()
        #self.ag1.start()
        #self.ag2.start()
        #self.ag3.start()
        #self.ag4.start()
        #self.ag5.start()

        # in continuous regime return immediately leaving the threads to run on the background
        # in epoch regime wait for the task to complete, for the threads to stop, then return
        if sample_epochs is not None:
            self.sampling_coord.join(self.sampling_coord.threads)
            self.sampling_coord.clear_stop()
        return None


    def do_training(self,train_epochs=None):
        while True:

            if (self.global_step % 100 == 99):
                self.saver.save(self.sess, FLAGS.summaries_dir + '/' + str(FLAGS.run_name) + "_netstate/saved_state", global_step=self.global_step)
                _,my_summaries, my_cost,my_training_queue_size = self.sess.run([self.train_step_run,self.merged_summaries, self.softmax_RMSD, self.training_queue_size], feed_dict={self.keep_prob:0.5})
                print "global step:", self.global_step, "softmax RMSD cost:", my_cost

                self.summary_writer.add_summary(my_summaries, self.global_step)
            else:
                self.sess.run([self.train_step_run], feed_dict={self.keep_prob: 0.5})

            # increment the batch counter
            self.global_step +=1


        # should be cleaning the queue from the trash -- I think no
        #
        # I will need memory in the future

        # is there a rule for fastest convergence ??
        # maybe, there is a rule Gradient/Second
        # simplest rule is cost update step
        # good networks will , probably, need a lot of sampling at the end if this thing ever converges

        # most of the cost is sampling (almost all) at moment X
        # let's go simplest and do distributed training
        # on all GPUs at the same time

        # ideally, it's infinite sampling per step of training
        # for best results sampling is absolute; any not absolute sampling is an exchange of efficacy for the speed
        # training is always a single step in the future
        # be not afraid of physics -- it brings good first-layer convolutions



a = GradientDescendMachine()

if tf.gfile.Exists(FLAGS.summaries_dir + "/" + str(FLAGS.run_name) +'_netstate' ):
    raise Exception('Summaries folder already exists. Please, change the run name, or delete it manually.')
else:
    tf.gfile.MakeDirs(FLAGS.summaries_dir + "/" + str(FLAGS.run_name) +'_netstate')


a.do_sampling(sample_epochs=None)
a.do_training(train_epochs=1)


print "All Done"
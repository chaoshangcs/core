import time,os
import tensorflow as tf
import numpy as np
from deepVS_input import index_the_database_into_queue, read_receptor_and_ligand
from deepVS_net import *

def train():
    "train a network"
    # it's better if all of the computations use a single session
    sess = tf.Session()

    # create a filename queue first
    filename_queue, examples_in_database = index_the_database_into_queue(FLAGS.database_path, shuffle=True)

    # create an epoch counter
    batch_counter = tf.Variable(0)
    batch_counter_increment = tf.assign(batch_counter,tf.Variable(0).count_up_to(np.round((examples_in_database*FLAGS.num_epochs)/FLAGS.batch_size)))
    epoch_counter = tf.div(batch_counter*FLAGS.batch_size,examples_in_database)

    #read data from files
    _, current_epoch, label, ligand_atoms, ligand_coords, receptor_elements, receptor_coords = read_receptor_and_ligand(filename_queue,epoch_counter=epoch_counter,train=train)

    keep_prob = tf.placeholder(tf.float32)

    #run it through the network
    single_example = deepVS_net(ligand_atoms, ligand_coords, keep_prob)
    
    #calculate the cross entropy
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=single_example,labels=label)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross entropy mean', cross_entropy_mean)

    # Adam optimizer is a very heart of the network
    train_step_run = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cross_entropy)
    
    # merge all summaries and create a file writer object
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter((FLAGS.summaries_dir + '/' + str(FLAGS.run_index) + "_train"), sess.graph)

    # create saver to save and load the network state
    saver = tf.train.Saver()
    if FLAGS.saved_session is None:
        sess.run(tf.global_variables_initializer())
    else:
        print "Restoring variables from sleep. This may take a while..."
        saver.restore(sess,FLAGS.saved_session)

    # launch all threads only after the graph is complete and all the variables initialized
    # previously, there was a hard to find occasional problem where the computations would start on unfinished nodes
    # IE: lhs shape [] is different from rhs shape [100] and others
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    mem_of_entropy = []

    while True:
        start = time.time()
        batch_num = sess.run(batch_counter_increment)

        epo,c_entropy_mean,_ = sess.run([current_epoch,cross_entropy_mean,train_step_run], feed_dict={keep_prob: 0.5})
        mem_of_entropy.append(c_entropy_mean)

        if (batch_num % 100 == 99):
            print "epoch:",epo,"global step:", batch_num, "\tentropy:", c_entropy_mean, "\entropy average:", np.average(mem_of_entropy[-1000:]),
            print "\texamples per second:", "%.2f" % (FLAGS.batch_size / (time.time() - start))

        if (batch_num % 10000 == 9999):
            # once in a while save the network state and write variable summaries to disk
            summaries = sess.run(merged_summaries, feed_dict={keep_prob:1})
            mem_of_entropy = mem_of_entropy[-1000:]
            print 'saving to disk...'
            train_writer.add_summary(summaries, batch_num)
            saver.save(sess, FLAGS.summaries_dir + '/' + str(FLAGS.run_index) + "_netstate/saved_state", global_step=batch_num)

    assert not np.isnan(cross_entropy_mean), 'Model diverged with loss = NaN'

class FLAGS:
    """important model parameters"""

    # size of one pixel generated from protein in Angstroms (float)
    pixel_size = 0.5
    # size of the box around the ligand in pixels
    side_pixels = 40
    # weights for each class for the scoring function
    # number of times each example in the dataset will be read
    num_epochs = 50000 # epochs are counted based on the number of the protein examples
    # usually the dataset would have multiples frames of ligand binding to the same protein
    # av4_input also has an oversampling algorithm.
    # Example: if the dataset has 50 frames with 0 labels and 1 frame with 1 label, and we want to run it for 50 epochs,
    # 50 * 2(oversampling) * 50(negative samples) = 50 * 100 = 5000
    # num_classes = 2
    # parameters to optimize runs on different machines for speed/performance
    # number of vectors(images) in one batch
    batch_size = 1
    # number of background processes to fill the queue with images
    num_threads = 512
    # data directories

    # path to the csv file with names of images selected for training
    database_path = "../../common/data/labeled_av4"
    # directory where to write variable summaries
    summaries_dir = './summaries'
    # optional saved session: network from which to load variable states
    saved_session = None#'./summaries/36_netstate/saved_state-23999'


def main(_):
    """gracefully creates directories for the log files and for the network state launches. After that orders network training to start"""
    summaries_dir = os.path.join(FLAGS.summaries_dir)
    # FLAGS.run_index defines when
    FLAGS.run_index = 1
    while ((tf.gfile.Exists(summaries_dir + "/"+ str(FLAGS.run_index) +'_train' ) or tf.gfile.Exists(summaries_dir + "/" + str(FLAGS.run_index)+'_test' ))
           or tf.gfile.Exists(summaries_dir + "/" + str(FLAGS.run_index) +'_netstate') or tf.gfile.Exists(summaries_dir + "/" + str(FLAGS.run_index)+'_logs')) and FLAGS.run_index < 1000:
        FLAGS.run_index += 1
    else:
        tf.gfile.MakeDirs(summaries_dir + "/" + str(FLAGS.run_index) +'_train' )
        tf.gfile.MakeDirs(summaries_dir + "/" + str(FLAGS.run_index) +'_test')
        tf.gfile.MakeDirs(summaries_dir + "/" + str(FLAGS.run_index) +'_netstate')
        tf.gfile.MakeDirs(summaries_dir + "/" + str(FLAGS.run_index) +'_logs')
    train()

if __name__ == '__main__':
    tf.app.run()
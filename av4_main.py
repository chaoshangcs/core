import time,os
import tensorflow as tf
import numpy as np
from av4_input import index_the_database_into_queue,image_and_label_queue
from av4_networks import *

# telling tensorflow how we want to randomly initialize weights

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

    # create a custom shuffle queue
    _,current_epoch,_,xyz_label,image_batch = image_and_label_queue(batch_size=FLAGS.batch_size,
                                                                                      pixel_size=FLAGS.pixel_size,
                                                                                      side_pixels=FLAGS.side_pixels,
                                                                                      num_threads=FLAGS.num_threads,
                                                                                      filename_queue=filename_queue,
                                                                                      epoch_counter=epoch_counter)


    keep_prob = tf.placeholder(tf.float32)
    logits = ag_net_3(image_batch,keep_prob,FLAGS.batch_size)
    raw_labels = xyz_label[:,-3:]
    #epsilon = 0.001
    #norm_labels = tf.cast(raw_labels/np.pi+epsilon,tf.int32)
    norm_labels = raw_labels / np.pi

    # try square cost
    norm_preds = tf.nn.softmax(logits)[:,:,1]
    variance_penalty = ((-3 * norm_preds**2) + (3*norm_preds) - 1.0) /3.0
    square_cost = tf.reduce_mean(((norm_labels - norm_preds)**2 + variance_penalty),reduction_indices=1)
    mean_square_cost = tf.reduce_mean(square_cost)

    # try softmax cross entropy over three classes
    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=norm_labels,logits=logits)
    #cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # tester line to see if softmax is working correctly
    # wrong = tf.reduce_sum(norm_preds,axis=1)
    tf.summary.scalar('mean square cost', mean_square_cost)

    # randomly shuffle along the batch dimension and calculate an error
    shuffled_labels = tf.random_shuffle(norm_labels)
    #shuffled_square_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=shuffled_labels,logits=logits))
    shuffled_square_cost = tf.reduce_mean(((shuffled_labels - norm_preds)**2 + variance_penalty))
    tf.summary.scalar('shuffled square cost', shuffled_square_cost)

    # Adam optimizer is a very heart of the network
    train_step_run = tf.train.AdamOptimizer(1e-4).minimize(mean_square_cost)

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


    # TODO
    # DEBUG
    num_ligand_atoms = tf.reduce_sum(tf.cast(image_batch[0,:,:,:,0] >0 ,tf.float32))
    num_receptor_atoms = tf.reduce_sum(tf.cast((image_batch[0,:,:,:,1] >0),tf.float32))
    #num_receptor_atoms = tf.reduce_sum(tf.cast(tf.logical_and((image_batch[0, :, :, :, 1] < 7), (image_batch[0] > 0)), tf.float32))
    raw_label = raw_labels[0]
    norm_label = norm_labels[0]
    logit = logits[0]
    cost = square_cost[0]

    # launch all threads only after the graph is complete and all the variables initialized
    # previously, there was a hard to find occasional problem where the computations would start on unfinished nodes
    # IE: lhs shape [] is different from rhs shape [100] and others
    coord = tf.train.Coordinator()
    tf.get_default_graph().finalize()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    np.set_printoptions(precision=2)
    while True:
        start = time.time()
        batch_num = sess.run(batch_counter_increment)
        #print sess.run(train_step_run, feed_dict={keep_prob:0.5})
        epo,my_cost,_ = sess.run([current_epoch,mean_square_cost,train_step_run],
                     feed_dict={keep_prob: 0.5})
        print "epoch:",epo[0],"step:", batch_num,
        print "\tcost:","%.2f" % np.average(my_cost),

        my_raw_label,my_norm_label,my_n_ligand_atoms,my_n_receptor_atoms,my_logit,my_cost = \
            sess.run([raw_label, norm_label,num_ligand_atoms,num_receptor_atoms,logit,cost], feed_dict={keep_prob:0.5})

        if (batch_num % 500 == 499):
            print "l atoms:", my_n_ligand_atoms,
            print "p atoms:", my_n_receptor_atoms,
            print "raw labels:",my_raw_label,
            print "norm label:",my_norm_label,
            print "raw prediction:",my_logit
            print "cost:",my_cost

        print "\texps:", "%.2f" % (FLAGS.batch_size / (time.time() - start))

        if (batch_num % 500 == 499):
            # once in a while save the network state and write variable summaries to disk
            my_cost,my_shuffled_cost,summaries = sess.run(
                [mean_square_cost, shuffled_square_cost, merged_summaries], feed_dict={keep_prob: 1})
            print "cost:",my_cost, "shuffled square cost:", my_shuffled_cost
            train_writer.add_summary(summaries, batch_num)
            saver.save(sess, FLAGS.summaries_dir + '/' + str(FLAGS.run_index) + "_netstate/saved_state", global_step=batch_num)#

#    assert not np.isnan(cross_entropy_mean), 'Model diverged with loss = NaN'


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
    batch_size = 100
    # number of background processes to fill the queue with images
    num_threads = 512
    # data directories

    # path to the csv file with names of images selected for training
    database_path = "../datasets/labeled_av4"
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

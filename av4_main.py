import time,os
import tensorflow as tf
import numpy as np
from av4_input import index_the_database_into_queue,image_and_label_queue
import av4_networks

from av4_config import FLAGS

# telling tensorflow how we want to randomly initialize weights

def train():
    "train a network"
    # it's better if all of the computations use a single session
    sess = tf.Session()

    # create a filename queue first
    filename_queue, examples_in_database = index_the_database_into_queue(FLAGS.database_path, shuffle=True)


    # create an epoch counter
    with tf.name_scope("epoch_counter"):
        batch_counter = tf.Variable(0)
        batch_counter_increment = tf.assign(batch_counter,tf.Variable(0).count_up_to(np.round((examples_in_database*FLAGS.num_epochs)/FLAGS.batch_size)))
        epoch_counter = tf.div(batch_counter*FLAGS.batch_size,examples_in_database)

    # create a custom shuffle queue
    _,current_epoch,label_batch,image_batch = image_and_label_queue(batch_size=FLAGS.batch_size, pixel_size=FLAGS.pixel_size,
                                                                          side_pixels=FLAGS.side_pixels, num_threads=FLAGS.num_threads,
                                                                          filename_queue=filename_queue, epoch_counter=epoch_counter)

#    image_batch = tf.sparse_tensor_to_dense(sparse_image_batch,validate_indices=False)

    keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope("network"):
        predicted_labels = av4_networks.max_net.compute_output(image_batch,keep_prob,FLAGS.batch_size)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predicted_labels,labels=label_batch)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross entropy mean', cross_entropy_mean)

    # randomly shuffle along the batch dimension and calculate an error
    shuffled_labels = tf.random_shuffle(label_batch)
    shuffled_cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predicted_labels,labels=shuffled_labels))
    tf.summary.scalar('shuffled cross entropy mean', shuffled_cross_entropy_mean)

    # Adam optimizer is a very heart of the network
    with tf.name_scope("Adam_optimizer"):
        train_step_run = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    # merge all summaries and create a file writer object
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter((FLAGS.summaries_dir + '/' + str(FLAGS.run_index) + "_train"), sess.graph)


    # create saver to save and load the network state
    saver = tf.train.Saver(var_list=(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="Adam_optimizer") +
                                     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="network") +
                                     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="epoch_counter")))


    if FLAGS.saved_session is None:
        sess.run(tf.global_variables_initializer())
    else:
        print "Restoring variables from sleep. This may take a while..."
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess,FLAGS.saved_session)

        print "all variables restored. Start training"



    # launch all threads only after the graph is complete and all the variables initialized
    # previously, there was a hard to find occasional problem where the computations would start on unfinished nodes
    # IE: lhs shape [] is different from rhs shape [100] and others
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    while True:

#        print tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#        time.sleep(100)

        start = time.time()
        batch_num = sess.run(batch_counter_increment)
        epo,c_entropy_mean,_ = sess.run([current_epoch,cross_entropy_mean,train_step_run], feed_dict={keep_prob: 0.5})
        print "epoch:",epo[0],"global step:", batch_num, "\tcross entropy mean:", c_entropy_mean,

        print "\texamples per second:", "%.2f" % (FLAGS.batch_size / (time.time() - start))

        if (batch_num % 100 == 99):
            # once in a while save the network state and write variable summaries to disk
            c_entropy_mean,sc_entropy_mean,summaries = sess.run(
                [cross_entropy_mean, shuffled_cross_entropy_mean, merged_summaries], feed_dict={keep_prob: 1})
            print "cross entropy mean:",c_entropy_mean, "shuffled cross entropy mean:", sc_entropy_mean
            train_writer.add_summary(summaries, batch_num)
            saver.save(sess, FLAGS.summaries_dir + '/' + str(FLAGS.run_index) + "_netstate/saved_state", global_step=batch_num)

    assert not np.isnan(cross_entropy_mean), 'Model diverged with loss = NaN'




def main(_):
    """gracefully creates directories for the log files and for the network state launches. After that orders network training to start."""
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

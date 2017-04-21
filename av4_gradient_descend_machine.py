import time,re
import tensorflow as tf
import numpy as np
import av4_input
from av4_main import FLAGS
import av4_conformation_sampler


FLAGS.saved_session = './summaries/4_netstate/saved_state-82099'
FLAGS.predictions_file_path = re.sub("netstate","logs",FLAGS.saved_session)
FLAGS.database_path = '../datasets/holdout_av4'
FLAGS.num_epochs = 2
FLAGS.top_k = FLAGS.num_epochs


search_agent1 = av4_conformation_sampler.SearchAgent()


def evaluate_on_train_set():
    "train a network"

    # create session to compute evaluation
    sess = FLAGS.main_session

    # create a filename queue first
    filename_queue,ex_in_database = av4_input.index_the_database_into_queue(FLAGS.database_path, shuffle=True)

    with tf.name_scope("epoch_counter"):
        "create an epoch counter"
        counter = tf.Variable(0)
        counter_incr = tf.assign(counter,tf.Variable(0).count_up_to(np.round(ex_in_database*FLAGS.num_epochs)))
        e_counter = tf.div(counter_incr,ex_in_database)

    # read receptor and ligand from the queue
    lig_file,_,_,lig_elements,lig_coords,rec_elements,rec_coords = av4_input.read_receptor_and_ligand(
        filename_queue=filename_queue,epoch_counter=tf.constant(0))


    # create a very large queue of images for central parameter server
    


    # create saver to save and load the network state
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list=(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="Adam_optimizer")
                                    + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="network")))
                                    #+ tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="epoch_counter")))

    if FLAGS.saved_session is None:
        sess.run(tf.global_variables_initializer())
    else:
        sess.run(tf.global_variables_initializer())
        print "Restoring variables from sleep. This may take a while..."
        saver.restore(sess,FLAGS.saved_session)
        print "unitialized vars:", sess.run(tf.report_uninitialized_variables())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord=coord)


    while True or not coord.should_stop():
        start = time.time()

        my_epoch,my_lig_file,my_lig_elements,my_lig_coords,my_rec_elements,my_rec_coords = \
            sess.run([e_counter,lig_file,lig_elements,lig_coords,rec_elements,rec_coords])

        search_agent1.grid_evaluate_positions(my_lig_elements,my_lig_coords,my_rec_elements,my_rec_coords)

        print "epoch:", my_epoch,"\t",my_lig_file.split("/")[-1],
        print "\tpositional search took :", "%.2f" % (time.time() - start), "seconds."


evaluate_on_train_set()
print "All Done"

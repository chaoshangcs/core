import time,re
import tensorflow as tf
import numpy as np
from av3 import FLAGS,max_net,compute_weighted_cross_entropy_mean
from av3_input import launch_enqueue_workers

# set up global parameters
FLAGS.saved_session = './summaries/36_netstate/netstate-155999'

FLAGS.predictions_file_path = re.sub("netstate","logs",FLAGS.saved_session)


# todo log AUC
# todo cross entropy
# change predictions to all_prediction to avoid confusion
# todo better epoch counter that can stop


class store_predictions:
    """stores all of the predictions
    unique and sorted by protein-ligand pairs"""
    pl_pairs =np.array([],dtype=str)
    # collects all of the predictions (for the same pair) into an array of objects
    predictions = np.array([])
    # collects labels
    labels = np.array([])

    # formatting options in the text file
    def usd_format(self,number_array):
        """converts numpy array into user defined format"""
        return np.char.array(np.around(number_array,decimals=3),itemsize=5)

    def top_100_score(self,labels):
        """ takes sorted in descending order by predictions list of labels
        if the dataset had n=100 drugs, how many drugs are in top 100 of the list
        top_100_score = (TP in head -n)/n, where n is a number of Positives in a whole dataset
        takes list of predictions, corresponding labels, returns float -percent that is correct"""

        # sort the array by predictions
        order = np.flipud(predictions.argsort())
        labels = labels[order]

        # take the top n
        num_positives = np.sum(np.asarray(labels,dtype=bool))
        return np.sum(labels[0:num_positives]) / num_positives * 100

    def auc(self,predictions,labels):
        """calculates area under the curve AUC for binary predictions/labels needs
        sorted in descending order predictions"""

        # calculate TP boolean mask
        TP_mask = (np.round(predictions) == True) * (np.asarray(labels,dtype=bool) == True)

        # slide from top to the bottom; sum TP, sum FP into an array
        num_predictions = len(labels)
        roc_curve = np.array([0,0])
        memory = np.array([0,0])
        TP_above_threshold = 0
        for i in range(num_predictions):
            if TP_mask[i] == True:
                TP_above_threshold +=1
            FP_above_threshold = i - TP_above_threshold

            if not all(memory==[FP_above_threshold,TP_above_threshold]):
                roc_curve = np.vstack((roc_curve,[FP_above_threshold,TP_above_threshold]))
                memory = np.array([FP_above_threshold,TP_above_threshold])

        # reduce into TP and FP rate, integrate with trapezoid to calculate AUC

        auc = np.trapz(np.true_divide(roc_curve[:,1],np.sum(TP_mask)), x=np.true_divide(roc_curve[:,0],(num_predictions-np.sum(TP_mask))))
        # abc stands for Area Below the Curve
        # abc = np.trapz(np.true_divide(roc_curve[:,0],(num_predictions-np.sum(TP_mask))),x=np.true_divide(roc_curve[:,1],np.sum(TP_mask)))

        return auc

    def confusion_matrix(self,predictions,labels):
        """calculaets and returns the confusion matrix"""
        TP = np.sum((np.round(predictions) == True) * (np.asarray(labels, dtype=bool) == True))
        FP = np.sum((np.round(predictions) == True) * (np.asarray(labels, dtype=bool) == False))
        FN = np.sum((np.round(predictions) == False) * (np.asarray(labels, dtype=bool) == True))
        TN = np.sum((np.round(predictions) == False) * (np.asarray(labels, dtype=bool) == False))


        return np.array([[TP,FP],[FN,TN]])



    def add_batch(self,ligand_file_path,receptor_file_path,batch_predictions,batch_labels):
        """maintains sorted by protein-ligand pairs lists of various predictions
        splits the batch into "new" and "known" protein-ligand pairs
        adds predictions of "known" ligand pairs with "," appends new protein-ligand pairs as they are"""


        # extract meaningful names from the file path
        def extract_file_from_path(file_path):
            return file_path.split("/")[-1]

        ligand_filenames = np.char.array(map(extract_file_from_path,ligand_file_path))
        receptor_filenames = np.char.array(map(extract_file_from_path,receptor_file_path))
        batch_pl_pairs = ligand_filenames + "," + receptor_filenames

        # sort the batch by protein-ligand pairs
        order = batch_pl_pairs.argsort()
        batch_pl_pairs = batch_pl_pairs[order]
        batch_predictions = batch_predictions[order]
        batch_labels = batch_labels[order]

        # check if all of the entries in the batch are unique
        if not np.array_equal(batch_pl_pairs,np.unique(batch_pl_pairs)):
            raise Exception("batch has duplicate entries")

        # get binmask with True for each non-unique protein-ligand pair, False for unique protein-ligand pair
        binmask_self = (np.searchsorted(batch_pl_pairs,self.pl_pairs, 'right') - np.searchsorted(batch_pl_pairs,self.pl_pairs,'left')) == 1
        binmask_batch = (np.searchsorted(self.pl_pairs,batch_pl_pairs, 'right') - np.searchsorted(self.pl_pairs,batch_pl_pairs,'left')) == 1

        # check if the entries appended to each other have similar names
        if not np.array_equal(batch_pl_pairs[binmask_batch],self.pl_pairs[binmask_self]):
            raise Exception('Error while merging arrays. Names do not match')

        # check if labels are similar
        if not np.array_equal(batch_labels[binmask_batch],self.labels[binmask_self]):
            raise Exception('Error while merging arrays. Labels for the same example should be similar')

        # split into overlapping and not overlapping entries
        overlap_pl_pairs = batch_pl_pairs[binmask_batch]
        overlap_predictions = np.char.array(self.predictions[binmask_self])
        batch_overlap_predictions = self.usd_format(batch_predictions[binmask_batch])

        # for known entries join all of the predictions together
        overlap_predictions = overlap_predictions + "," + batch_overlap_predictions
        overlap_labels = batch_labels[binmask_batch]

        # merge unique and not unique predictions
        self.pl_pairs = np.hstack((self.pl_pairs[-binmask_self],batch_pl_pairs[-binmask_batch],overlap_pl_pairs))
        self.predictions = np.hstack((self.predictions[-binmask_self],self.usd_format(batch_predictions[-binmask_batch]),overlap_predictions))
        self.labels = np.hstack((self.labels[-binmask_self],batch_labels[-binmask_batch],overlap_labels))

        # now sort everything by the first column
        order = self.pl_pairs.argsort()
        self.pl_pairs = self.pl_pairs[order]
        self.predictions = self.predictions[order]
        self.labels = self.labels[order]


    def save_predictions(self,file_path):
        """sorts in descending order of confidence,computes average predictions,formats and writes into file"""
        # compute average of predictions
        num_examples = len(self.labels)

        if num_examples == 0:
            raise Exception ("nothing to save")

        def string_to_average(string):
            return np.average(np.array(string.split(","),dtype=float))
        prediction_averages = np.around(map(string_to_average,self.predictions),decimals=3)

        # sort by prediction averages
        order = np.flipud(prediction_averages.argsort())
        prediction_averages = prediction_averages[order]
        self.pl_pairs = self.pl_pairs[order]
        self.predictions = self.predictions[order]
        self.labels = self.labels[order]
        # write all of the predictions to the file
        f = open(file_path + "_predictions.txt", 'w')

        for i in range(num_examples):
            f.write((str(prediction_averages[i]) + " "*10)[:10]
                    + (str(self.labels[i]) + " "*50)[:10]
                    + str(self.pl_pairs[i] + " "*50)[:50]
                    + str(self.predictions[i] + " "*50)[:50]
                    + "\n")

        f.close()
        # write and save some metadata

        f = open(file_path + "_scores.txt", 'w')
        f.write("top 100 score: ")
        f.write(str(self.top_100_score(self.labels)))
        f.write("\nAUC: ")
        f.write(str(self.auc(prediction_averages,self.labels)))
        f.write("\nconfusion matrix: ")
        f.write(str(self.confusion_matrix(prediction_averages,self.labels)))
        f.close()


def evaluate_on_train_set():
    "train a network"

    # create session all of the evaluation happens in one
    sess = tf.Session()
    train_image_queue,filename_coordinator = launch_enqueue_workers(sess=sess,pixel_size=FLAGS.pixel_size,side_pixels=FLAGS.side_pixels,
                                                                    num_workers=FLAGS.num_workers, batch_size=FLAGS.batch_size,
                                                                    database_index_file_path=FLAGS.test_set_file_path,num_epochs=3)
    y_, x_image_batch,ligand_filename,receptor_filename = train_image_queue.dequeue_many(FLAGS.batch_size)
    keep_prob = tf.placeholder(tf.float32)
    y_conv = max_net(x_image_batch, keep_prob)

    cross_entropy_mean = compute_weighted_cross_entropy_mean(y_conv, y_, batch_size=FLAGS.batch_size)

    # compute softmax over raw predictions
    predictions = tf.nn.softmax(y_conv)[:,1]

    # restore variables from sleep
    saver = tf.train.Saver()
    saver.restore(sess,FLAGS.saved_session)

    # create a variable to store all predictions
    all_predictions = store_predictions()
    batch_num = 0

    while not filename_coordinator.stop:
        start = time.time()

        my_ligand_filename,my_receptor_filename,my_predictions,labels,my_cross_entropy = sess.run([ligand_filename,receptor_filename,predictions,y_,cross_entropy_mean],feed_dict={keep_prob:1})
        all_predictions.add_batch(my_ligand_filename,my_receptor_filename,my_predictions,labels)
        print "step:", batch_num, "test error:", my_cross_entropy, "examples per second:", "%.2f" % (FLAGS.batch_size / (time.time() - start))

        batch_num +=1

    all_predictions.save_predictions(FLAGS.predictions_file_path)


evaluate_on_train_set()


print "all done!"

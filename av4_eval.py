import time,os
import tensorflow as tf
import numpy as np
import re
from av4_input import image_and_label_queue,index_the_database_into_queue
from av4_main import FLAGS
from av4_networks import nilai_net
from collections import defaultdict


FLAGS.saved_session = '/home/maksym/PyCharmProjects/tensorMe17av4/summaries/99_netstate/saved_state-57999'
FLAGS.predictions_file_path = re.sub("netstate","logs",FLAGS.saved_session)
FLAGS.database_path = '../datasets/ordered_unlabeled_av4'
FLAGS.num_epochs = 25
FLAGS.top_k = FLAGS.num_epochs


class store_predictions_av3:
    """stores all of the predictions
    unique and sorted by protein-ligand pairs"""
    pl_pairs = np.array([], dtype=str)
    # collects all of the predictions (for the same pair) into an array of objects


    predictions = np.array([])
    # collects labels
    labels = np.array([])

    # formatting options in the text file
    def usd_format(self, number_array):
        """converts numpy array into user defined format"""
        return np.char.array(np.around(number_array, decimals=3), itemsize=5)

    def top_100_score(self, predictions, labels):
        """ takes sorted in descending order by predictions list of labels
        if the dataset had n=100 drugs, how many drugs are in top 100 of the list
        top_100_score = (TP in head -n)/n, where n is a number of Positives in a whole dataset
        takes list of predictions, corresponding labels, returns float -percent that is correct"""

        # sort the array by predictions
        order = np.flipud(predictions.argsort())
        labels = labels[order]

        # take the top n
        num_positives = np.sum(np.asarray(labels, dtype=bool))

        # if no positive labels return nan
        if num_positives == 0:
            return float('nan')
        else:
            return np.sum(labels[0:num_positives]) / num_positives * 100

    def auc(self, predictions, labels):
        """calculates area under the curve AUC for binary predictions/labels needs
        sorted in descending order predictions"""

        # sort the array by predictions in descending order in case it has not been done
        order = np.flipud(predictions.argsort())
        labels = labels[order]

        # clean labels, calculate the number of positive labels
        labeled_true = (np.asarray(labels, dtype=bool) == True)
        num_positives = np.sum(labeled_true)
        num_predictions = len(labeled_true)

        # If no positive result return nan
        if num_positives == 0:
            return float('nan')

        # slide from top to the bottom;
        # each time slide the threshold so as to predict one more label as positive
        roc_curve = np.array([0.0, 0.0])
        TP_above_threshold = 0
        for predict_as_positive in range(num_predictions):
            if labeled_true[predict_as_positive] == True:
                TP_above_threshold += 1
            # calculate True Positives Rate
            # TPR = TP / num_real_positives
            TPR = TP_above_threshold / float(num_positives)

            # FPR = FP / num_real_negatives
            FPR = (predict_as_positive + 1 - TP_above_threshold) / (num_predictions - float(num_positives))

            roc_curve = np.vstack((roc_curve, [FPR, TPR]))

        roc_curve = np.vstack((roc_curve, [1.0, 1.0]))

        # reduce into TP and FP rate, integrate with trapezoid to calculate AUC
        auc = np.trapz(roc_curve[:, 1], x=roc_curve[:, 0])

        return auc

    def confusion_matrix(self, predictions, labels):
        """calculaets and returns the confusion matrix"""
        TP = np.sum((np.round(predictions) == True) * (np.asarray(labels, dtype=bool) == True))
        FP = np.sum((np.round(predictions) == True) * (np.asarray(labels, dtype=bool) == False))
        FN = np.sum((np.round(predictions) == False) * (np.asarray(labels, dtype=bool) == True))
        TN = np.sum((np.round(predictions) == False) * (np.asarray(labels, dtype=bool) == False))

        return np.array([[TP, FP], [FN, TN]])

    def add_batch(self, ligand_file_paths,ligand_frames,batch_predictions, batch_labels):
        """maintains sorted by protein-ligand pairs lists of various predictions
        splits the batch into "new" and "known" protein-ligand pairs
        adds predictions of "known" ligand pairs with "," appends new protein-ligand pairs as they are"""


        # since with multithreading and a large batch pool size some of the entries may not be unique(rare),
        # and it is faster to add entries that are unique, duplicates would be discarded
        ligand_file_paths,unique_indices = np.unique(ligand_file_paths, return_index=True)
        batch_predictions = batch_predictions[unique_indices]
        batch_labels = batch_labels[unique_indices]
        ligand_frames = ligand_frames[unique_indices]

        # extract meaningful names from the file path
        def create_name_from_path(file_path,ligand_frame):
            return file_path.split("/")[-1] +"_frame"+str(ligand_frame)

        ligand_filenames = np.char.array(map(create_name_from_path,ligand_file_paths,ligand_frames))
        #receptor_filenames = np.char.array(map(extract_file_from_path, receptor_file_path))
        batch_pl_pairs = ligand_filenames #+ "," + receptor_filenames

        # sort the batch by protein-ligand pairs
        order = batch_pl_pairs.argsort()
        batch_pl_pairs = batch_pl_pairs[order]
        batch_predictions = batch_predictions[order]
        batch_labels = batch_labels[order]

        # check if all of the entries in the batch are unique
        if not np.array_equal(batch_pl_pairs, np.unique(batch_pl_pairs)):
            raise Exception("batch has duplicate entries")

        # get binmask with True for each non-unique protein-ligand pair, False for unique protein-ligand pair
        binmask_self = (np.searchsorted(batch_pl_pairs, self.pl_pairs, 'right') - np.searchsorted(batch_pl_pairs,
                                                                                                  self.pl_pairs,
                                                                                                  'left')) == 1
        binmask_batch = (np.searchsorted(self.pl_pairs, batch_pl_pairs, 'right') - np.searchsorted(self.pl_pairs,
                                                                                                   batch_pl_pairs,
                                                                                                   'left')) == 1

        # check if the entries appended to each other have similar names
        if not np.array_equal(batch_pl_pairs[binmask_batch], self.pl_pairs[binmask_self]):
            raise Exception('Error while merging arrays. Names do not match')

        # check if labels are similar
        if not np.array_equal(batch_labels[binmask_batch], self.labels[binmask_self]):
            raise Exception('Error while merging arrays. Labels for the same example should be similar')

        # split into overlapping and not overlapping entries
        overlap_pl_pairs = batch_pl_pairs[binmask_batch]
        overlap_predictions = np.char.array(self.predictions[binmask_self])
        batch_overlap_predictions = self.usd_format(batch_predictions[binmask_batch])

        # for known entries join all of the predictions together
        overlap_predictions = overlap_predictions + "," + batch_overlap_predictions
        overlap_labels = batch_labels[binmask_batch]

        # merge unique and not unique predictions
        self.pl_pairs = np.hstack((self.pl_pairs[-binmask_self], batch_pl_pairs[-binmask_batch], overlap_pl_pairs))
        self.predictions = np.hstack(
            (self.predictions[-binmask_self], self.usd_format(batch_predictions[-binmask_batch]), overlap_predictions))
        self.labels = np.hstack((self.labels[-binmask_self], batch_labels[-binmask_batch], overlap_labels))

        # now sort everything by the first column
        order = self.pl_pairs.argsort()
        self.pl_pairs = self.pl_pairs[order]
        self.predictions = self.predictions[order]
        self.labels = self.labels[order]

    def save_predictions(self, file_path):
        """sorts in descending order of confidence,computes average predictions,formats and writes into file"""
        # compute average of predictions
        num_examples = len(self.labels)

        if num_examples == 0:
            raise Exception("nothing to save")

        def string_to_average(string):
            return np.average(np.array(string.split(","), dtype=float))

        prediction_averages = np.around(map(string_to_average, self.predictions), decimals=3)

        # sort by prediction averages
        order = np.flipud(prediction_averages.argsort())
        prediction_averages = prediction_averages[order]
        self.pl_pairs = self.pl_pairs[order]
        self.predictions = self.predictions[order]
        self.labels = self.labels[order]
        # write all of the predictions to the file
        f = open(file_path + "_predictions.txt", 'w')

        for i in range(num_examples):
            f.write((str(prediction_averages[i]) + " " * 10)[:10]
                    + (str(self.labels[i]) + " " * 50)[:10]
                    + str(self.pl_pairs[i] + " " * 50)[:50]
                    + str(self.predictions[i] + " " * 50)[:50]
                    + "\n")

        f.close()
        # write and save some metadata

        f = open(file_path + "_scores.txt", 'w')
        f.write("top 100 score: ")
        f.write(str(self.top_100_score(self.predictions, self.labels)))
        f.write("\nAUC: ")
        f.write(str(self.auc(prediction_averages, self.labels)))
        f.write("\nconfusion matrix: ")
        f.write(str(self.confusion_matrix(prediction_averages, self.labels)))
        f.close()

        # write a file in Kaggle MAP{K} submision format
        # the form is:
        # Protein1, Ligand3 Ligand4 Ligand2
        # Protein2, Ligand5 Ligand9 Ligand7

#        raw_database_array = np.genfromtxt(FLAGS.test_set_file_path, delimiter=',', dtype=str)
#        receptor_set = raw_database_array[:, 2]
#        receptor_set = list(set(map(lambda x: x.split('.')[0].split('/')[-1], receptor_set)))
#        submission = {}
#        for i in range(num_examples):
#            # get the name of the ligand and protein
#            ligand, receptor = self.pl_pairs[i].split(',')
#            ligand = ligand.split('/')[-1].split('.')[0]
#            receptor = receptor.split('/')[-1].split('.')[0]
#            # add all protein-ligand pairs to submission
#            if not receptor in submission.keys():
#                submission[receptor] = {}
#                submission[receptor]['ligands'] = [ligand]
#                submission[receptor]['score'] = [prediction_averages[i]]
#            else:
#                submission[receptor]['ligands'].append(ligand)
#                submission[receptor]['score'].append(prediction_averages[i])
#
#        # write and save submisison to file
#        # if failed to predict any ligand for a receptor
#        # use placeholder 'L' as predict result
#        # e.g. P1234,L
#        with open(file_path + '_submission.csv', 'w') as f:
#            f.write('Id,Expected\n')
#            for key in receptor_set:
#                if key in submission.keys():
#                    ligands = np.array(submission[key]['ligands'])
#                    scores = np.array(submission[key]['score'])
#                    ligands = ligands[np.flipud(scores.argsort())]
#                    f.write(key + ',' + ' '.join(ligands) + '\n')
#                else:
#                    f.write(key + ',' + 'L' + '\n')


class store_predictions:

    # store add of the prediction results
    #:return:


    raw_predictions = defaultdict(list)
    processed_predictions = defaultdict(list)

    def add_batch(self, ligand_file_paths, batch_current_epoch, batch_predictions):
        ligand_file_name = map(lambda filename:os.path.basename(filename).split('.')[0],ligand_file_paths)

        for ligand,current_epoch,prediction in zip(ligand_file_name, batch_current_epoch, batch_predictions):
            #   if in_the_range:
            self.raw_predictions[ligand].append(prediction)

    def reduce(self):
        '''
        if a ligand has more than one predictions
        use mean as final predictions
        '''
        for key, value in self.raw_predictions.items():

            if len(value) > 1:
                predictions_size = map(lambda x: len(x), value)
                if len(set(predictions_size)) > 1:
                    raise Exception(key, " has different number of predictions ", set(predictions_size))
                self.processed_predictions[key].append(np.mean(value, axis=0))
            else:
                self.processed_predictions[key].append(value)

    def final_predictions(self, predictions_list):
        length = min(len(predictions_list, 10))
        return np.mean(predictions_list[:length])

    def fill_na(self):
        for key in self.raw_predictions.keys():
            value_len = len(self.raw_predictions[key])
            if value_len>FLAGS.top_k:
                print "{} have more predictions than expected, {} reuqired {} found.".format(key,FLAGS.top_k,value_len)
            else:
                for i in range(FLAGS.top_k-value_len):
                    self.raw_predictions[key]

    def save_multiframe_predictions(self):
        records = []
        for key, value in self.raw_predictions.items():
            value_len = len(self.raw_predictions[key])
            if value_len>FLAGS.top_k:
                print "{} have more predictions than expected, {} reuqired {} found.".format(key,FLAGS.top_k,value_len)
                records.append([key]+value[:FLAGS.top_k])
            else:
                records.append([key]+value)

        columns = ['Id'] + ['Predicted_%d' % i for i in range(1, len(records[0]))]
        with open(FLAGS.predictions_file_path+'_multiframe_submission.csv','w') as fout:
            fout.write(','.join(columns)+'\n')
            for record in records:
                fout.write(','.join(map(str,record))+'\n')

    def save_average(self):
        '''
        take average of multiple predcition
        :return:
        '''
        records = []
        for key,value in self.raw_predictions.items():
            records.append([key,np.mean(np.array(value))])

        columns = ['ID', 'Predicted']
        with open(FLAGS.predictions_file_path + '_average_submission.csv','w') as fout:
            fout.write(','.join(columns)+'\n')
            for record in records:
                fout.write(','.join(map(str,record))+'\n')

    def save_max(self):

        records = []
        for key,value in self.raw_predictions.items():
            records.append([key, np.max(np.array(value))])

        columns = ['ID', 'Predicted']
        with open(FLAGS.predictions_file_path+'_max_submission.csv','w') as fout:
            fout.write(','.join(columns)+'\n')
            for record in records:
                fout.write(','.join(map(str,record))+'\n')


    def save(self):
        self.save_average()
        self.save_max()
        self.save_multiframe_predictions()


def evaluate_on_train_set():
    "train a network"

    # create session which all the evaluation happens in
    sess = tf.Session()

    # create a filename queue first
    filename_queue, examples_in_database = index_the_database_into_queue(FLAGS.database_path, shuffle=True)

    # create an epoch counter
    # there is an additional step with variable initialization in order to get the name of "count up to" in the graph
    batch_counter = tf.Variable(0)
    sess.run(tf.global_variables_initializer())
    batch_counter_increment = tf.assign(batch_counter,tf.Variable(0).count_up_to(np.round((examples_in_database*FLAGS.num_epochs)/FLAGS.batch_size)))

    batch_counter_var_name = sess.run(tf.report_uninitialized_variables())

    epoch_counter = tf.div(batch_counter*FLAGS.batch_size,examples_in_database)

    # create a custom shuffle queue
    ligand_files,current_epoch,label_batch,sparse_image_batch = image_and_label_queue(batch_size=FLAGS.batch_size, pixel_size=FLAGS.pixel_size,
                                                                          side_pixels=FLAGS.side_pixels, num_threads=FLAGS.num_threads,
                                                                          filename_queue=filename_queue, epoch_counter=epoch_counter,train=False)

    image_batch = tf.sparse_tensor_to_dense(sparse_image_batch,validate_indices=False)

    keep_prob = tf.placeholder(tf.float32)
    y_conv = nilai_net(image_batch,keep_prob,FLAGS.batch_size)

    # compute softmax over raw predictions
    predictions = tf.nn.softmax(y_conv)[:,1]

    # restore variables from sleep
    saver = tf.train.Saver()
    saver.restore(sess,FLAGS.saved_session)

    # use
    sess.run(tf.contrib.framework.get_variables_by_name(batch_counter_var_name[0])[0].initializer)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord=coord)

    # create an instance of a class to store predictions
    all_predictios = store_predictions()
    #all_predictions_av3 = store_predictions_av3()

    # add_batch(self, ligand_file_path, batch_predictions, batch_labels)

    print "starting evalution..."

    try:
        while True or not coord.should_stop():
            start = time.time()
            batch_num = sess.run([batch_counter_increment])
            my_ligand_files, my_ligand_frames, my_predictions, my_labels = sess.run(
                [ligand_files, current_epoch, predictions, label_batch],
                feed_dict={keep_prob: 1})
            print "current_epoch:", my_ligand_frames[0], "batch_num:", batch_num,
            print "\tprediction averages:", np.mean(my_predictions),
            print "\texamples per second:", "%.2f" % (FLAGS.batch_size / (time.time() - start))

            all_predictios.add_batch(my_ligand_files, my_ligand_frames, my_predictions)
            # add_batch(self, ligand_file_path, batch_predictions, batch_labels)
            #all_predictions_av3.add_batch(my_ligand_files,my_ligand_frames,my_predictions,my_labels)

            print "my labels:",my_labels

    except tf.errors.OutOfRangeError:
        print "exiting the loop"

    all_predictios.save()
    #all_predictions_av3.save_predictions(FLAGS.predictions_file_path)

evaluate_on_train_set()
print "All Done"

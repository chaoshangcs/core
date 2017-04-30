import tensorflow as tf


class FLAGS:
    # TODO: reformat every part of config to make it suitable to training & testing % evaluations

    """important model parameters"""

    # size of one pixel generated from protein in Angstroms (float)
    pixel_size = 1
    # size of the box around the ligand in pixels
    side_pixels = 20
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
    num_threads = 8
    # data directories

    # path to the csv file with names of images selected for training
    database_path = "../datasets/unlabeled_av4"
    # directory where to write variable summaries
    summaries_dir = './summaries'
    # optional saved session: network from which to load variable states
    saved_session = None #'./summaries/1_netstate/saved_state-113999'
    # main session for multiagent training
    main_session = tf.Session()

    ex_in_database=None



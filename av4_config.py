import tensorflow as tf


class FLAGS:
    # TODO: reformat every part of config to make it suitable to training & testing % evaluations

    """important model parameters"""


    # GENERAL MODEL PARAMETERS
    # size of the 3d pixel to render drug-protein image
    pixel_size = 1
    # size of the side of the box (box is centered on ligand's center of mass)
    side_pixels = 20
    # epochs are counted by positive examples
    num_epochs = 50000
    # number of background threads per agent on GPU
    num_threads = 8
    # batch size for the sampling agent
    batch_size = 100

    # SAMPLING PARAMETERS
    # new positions; exhaustive sampling
    shift_ranges = [9,9,9]
    shift_deltas = [3,3,3]
    initial_pose_evals = 50
    # parameters for a single output batch
    train_batch_init_poses = 50
    train_batch_gen_poses = 50


    # FILE LOCATION PARAMETERS
    # path with the training set
    database_path = "../datasets/unlabeled_av4"
    # directory where to write variable/graph summaries
    summaries_dir = './summaries'
    # optional saved session: network from which to load variable states
    saved_session = None
    # it's a good tradition to name the run with a number (easy to group)
    run_name = '14_test'

    # TECHNICAL (DO NOT MODIFY) PARAMETERS
    # main session for multiagent training
    main_session = tf.Session()
    # number of examples in the database
    ex_in_database = None
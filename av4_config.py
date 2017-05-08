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
    num_training_epochs = 500
    # number of background threads per agent on GPU
    num_threads = 8
    # batch size for the sampling agent
    batch_size = 128
    # capacity of the training queue
    train_q_capacity = 1000000
    # number of examples to keep in the train queue for good shuffling
    # (consider that typical batch size is 128)
    train_q_min_after_dequeue = 100000


    # SAMPLING PARAMETERS
    # new positions; exhaustive sampling
    shift_ranges = [4,4,4]
    shift_deltas = [2.1,2.1,2.1]
    initial_pose_evals = 64
    # parameters for a single output batch
    train_batch_init_poses = 64
    train_batch_gen_poses = 64


    # FILE LOCATION PARAMETERS
    # path with the training set
    database_path = "../datasets/labeled_av4"
    # directory where to write variable/graph summaries
    summaries_dir = './summaries'
    # optional saved session: network from which to load variable states
    saved_session = None
    # it's a good tradition to name the run with a number (easy to group)
    run_name = '1_run'


    # TECHNICAL (DO NOT MODIFY) PARAMETERS
    # main session for multiagent training
    main_session = tf.Session()
    # number of examples in the database
    ex_in_database = None
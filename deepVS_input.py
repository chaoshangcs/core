import tensorflow as tf
from glob import glob
import os,time
from av4_utils import generate_deep_affine_transform,affine_transform

def index_the_database_into_queue(database_path,shuffle):
    """Indexes av4 database and returns two lists of filesystem path: ligand files, and protein files.
    Ligands are assumed to end with _ligand.av4, proteins should be in the same folders with ligands.
    Each protein should have its own folder named similarly to the protein name (in the PDB)."""
    # TODO controls epochs here
    ligand_file_list = []
    receptor_file_list = []
    # for the ligand it's necessary and sufficient to have an underscore in it's name
    print "number of ligands:", len(glob(os.path.join(database_path+'/**/',"*[_]*.av4")))

    for ligand_file in glob(os.path.join(database_path+'/**/',"*[_]*.av4")):
        receptor_file = "/".join(ligand_file.split("/")[:-1]) + "/" + ligand_file.split("/")[-1][:4] + '.av4'
        if os.path.exists(receptor_file):
            ligand_file_list.append(ligand_file)
            receptor_file_list.append(receptor_file)
        else:

            # TODO: remove another naming system from Xiao's scripts                #
            receptor_file = os.path.join(os.path.dirname(ligand_file),os.path.basename(ligand_file).split("_")[0]+'.av4')
            if os.path.exists(receptor_file):                                       # remove later
                ligand_file_list.append(ligand_file)                                #
                receptor_file_list.append(receptor_file)                            #

    index_list = range(len(ligand_file_list))
    examples_in_database = len(index_list)

    if examples_in_database ==0:
        raise Exception('av4_input: No files found in the database path:',database_path)
    print "Indexed ligand-protein pairs in the database:",examples_in_database

    # create a filename queue (tensor) with the names of the ligand and receptors
    index_tensor = tf.convert_to_tensor(index_list,dtype=tf.int32)
    ligand_files = tf.convert_to_tensor(ligand_file_list,dtype=tf.string)
    receptor_files = tf.convert_to_tensor(receptor_file_list,dtype=tf.string)

    filename_queue = tf.train.slice_input_producer([index_tensor,ligand_files,receptor_files],num_epochs=None,shuffle=shuffle)
    return filename_queue,examples_in_database


def read_receptor_and_ligand(filename_queue,epoch_counter,train):
    """Reads ligand and protein raw bytes based on the names in the filename queue. Returns tensors with coordinates
    and atoms of ligand and protein for future processing.
    Important: by default it does oversampling of the positive examples based on training epoch."""

    def decode_av4(serialized_record):
        # decode everything into int32
        tmp_decoded_record = tf.decode_raw(serialized_record, tf.int32)
        # first four bytes describe the number of frames in a record
        number_of_frames = tf.slice(tmp_decoded_record, [0], [1])
        # labels are saved as int32 * number of frames in the record
        labels = tf.slice(tmp_decoded_record, [1], number_of_frames)
        # elements are saved as int32 and their number is == to the number of atoms
        number_of_atoms = ((tf.shape(tmp_decoded_record) - number_of_frames - 1) / (3 * number_of_frames + 1))
        elements = tf.slice(tmp_decoded_record, number_of_frames + 1, number_of_atoms)

        # coordinates are saved as a stack of X,Y,Z where the first(vertical) dimension
        # corresponds to the number of atoms
        # second (horizontal dimension) is x,y,z coordinate of every atom and is always 3
        # third (depth) dimension corresponds to the number of frames

        coords_shape = tf.concat([number_of_atoms, [3], number_of_frames],0)
        tmp_coords = tf.slice(tmp_decoded_record, number_of_frames + number_of_atoms + 1,
                              tf.shape(tmp_decoded_record) - number_of_frames - number_of_atoms - 1)
        multiframe_coords = tf.bitcast(tf.reshape(tmp_coords, coords_shape), type=tf.float32)

        return labels,elements,multiframe_coords

    # read raw bytes of the ligand and receptor
    idx = filename_queue[0]
    ligand_file = filename_queue[1]
    serialized_ligand = tf.read_file(ligand_file)
    serialized_receptor = tf.read_file(filename_queue[2])

    # decode bytes into meaningful tensors
    ligand_labels, ligand_elements, multiframe_ligand_coords = decode_av4(serialized_ligand)
    receptor_labels, receptor_elements, multiframe_receptor_coords = decode_av4(serialized_receptor)

    def count_frame_from_epoch(epoch_counter,ligand_labels,train):
        """Some simple arithmetics is used to sample all of the available frames
        if the index of the examle is even, positive label is taken every even epoch
        if the index of the example is odd, positive label is taken every odd epoch
        current negative example increments once every two epochs, and slides along all of the negative examples"""

        def select_pos_frame(): return tf.constant(0)
        def select_neg_frame(): return tf.mod(tf.div(1+epoch_counter,2), tf.shape(ligand_labels) - 1) +1

        current_frame = tf.cond(tf.equal(tf.mod(epoch_counter + idx + 1, 2), 1), select_pos_frame, select_neg_frame)

#        if train==True:
#        current_frame = tf.cond(tf.equal(tf.mod(epoch_counter+idx+1,2),1),select_pos_frame,select_neg_frame)
#        else:

#            current_frame = tf.mod(epoch_counter,tf.shape(ligand_labels))
        return current_frame

    current_frame = count_frame_from_epoch(epoch_counter,ligand_labels,train)
    ligand_coords = tf.gather(tf.transpose(multiframe_ligand_coords, perm=[2, 0, 1]),current_frame)
    label = tf.gather(ligand_labels,current_frame)

    return ligand_file,tf.squeeze(epoch_counter),tf.squeeze(label),ligand_elements,tf.squeeze(ligand_coords),receptor_elements,tf.squeeze(multiframe_receptor_coords)


def data_and_label_queue(batch_size,pixel_size,side_pixels,num_threads,filename_queue,epoch_counter,train=True):
    """Creates shuffle queue for training the network"""

    # read one receptor and stack of ligands; choose one of the ligands from the stack according to epoch
    ligand_file,current_epoch,label,ligand_elements,ligand_coords,receptor_elements,receptor_coords = read_receptor_and_ligand(filename_queue,epoch_counter=epoch_counter,train=train)

    # create a batch of proteins and ligands to read them together
    multithread_batch = tf.train.batch([ligand_file,current_epoch, label, ligand_elements, ligand_coords], batch_size, num_threads=num_threads,
                                       capacity=batch_size * 3,dynamic_pad=True,shapes=[[],[], [], [None], [None]])

    return multithread_batch
import tensorflow as tf
import numpy as np


def cross_entropy_with_RMSD(logits, lig_RMSDs, RMSD_threshold=3.0):
    """ Arbitrary function with assumptions:
    RMSD == 0 is a correct position     ; label = 1
    RMSD > 0 is an incorrect position   ; label = 0

    For positions RMSD > RMSD threshold; or RMSD == 0:
    the cost is a regular cross entropy cost

    For positions:
    0 < RMSD < RMSD_threshold
    final cost is multiplied by (RMSD_threshold - RMSD_ligand)/RMSD_threshold
    """
    labels = tf.cast((lig_RMSDs < 0.01), tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits)
    cost_correct_positions = cross_entropy * tf.cast(labels,tf.float32)
    cost_incorrect_positions = cross_entropy * tf.cast((lig_RMSDs > RMSD_threshold), tf.float32)
    cost_semicorrect_positions = cross_entropy \
                                 * tf.cast((lig_RMSDs < RMSD_threshold), tf.float32) \
                                 * tf.cast((lig_RMSDs > 0.01), tf.float32) * (lig_RMSDs/RMSD_threshold)
    return cost_incorrect_positions + cost_semicorrect_positions + cost_correct_positions
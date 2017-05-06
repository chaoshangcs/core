from __future__ import print_function

import os
import sys
import re
import time
import pprint
import argparse
import subprocess
import multiprocessing
from glob import glob
from functools import partial
from utils import smina_param
import prody
import numpy as np
import pandas as pd
import config
from database_action import DatabaseAction, db
from db_v2 import AffinityDatabase



FLAGS = None

def get_arguments():
    parser = argparse.ArgumentParser(description='Affinity Database')
    parser.add_argument('--create',dest='db_create', action='store_true')
    parser.add_argument('--continue',dest='db_continue', action='store_true')
    parser.add_argument('--delete',dest='db_delete', action='store_true')
    parser.add_argument('--progress', dest='db_progress', action='store_true')
    parser.add_argument('--list_param', dest='db_param',action='store_true')
    parser.add_argument('--action', type=str)
    parser.add_argument('--param', type=str)
    parser.add_argument('--retry_failed', action='store_true')
    parser.add_argument('--folder_name', type=str)
    parser.add_argument('--table_idx', type=int)
    parser.add_argument('--receptor_idx', type=int)
    parser.add_argument('--ligand_idx', type=int)
    parser.add_argument('--crystal_idx', type=int)
    parser.add_argument('--docked_idx', type=int)
    parser.add_argument('--download_idx', type=int)

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS

def run_multiprocess(target_list, func):
        print(len(target_list))
        if len(target_list) == 0:
            return 
        if type(target_list[0]).__name__ in ['unicode','str']:
            target_list = list(target_list)
        else:
            target_list = map(list, target_list)
        print (len(target_list))
        pool = multiprocessing.Pool(config.process_num)
        pool.map_async(func, target_list)
        pool.close()
        pool.join()
        
        #map(func, target_list)


    


def get_job_data(func_name, table_idx, table_param, progress=False):
    
    if func_name == 'download':
        download_list = open(config.list_of_PDBs_to_download).readline().strip().split(', ')
        finished_list = db.get_all_success(table_idx)
        failed_list = db.get_all_failed(table_idx)
        if FLAGS.retry_failed:
            rest_list = list(set(download_list) - set(finished_list) | set(failed_list))
        else:
            rest_list =list(set(download_list) - set(finished_list) - set(failed_list))

        total = len(set(download_list))
        finished = len(set(finished_list)-set(failed_list))
        failed = len(set(failed_list))

    elif func_name in ['split_ligand','split_receptor']:
        download_idx = table_param['download_idx']
        download_list = db.get_all_success(download_idx)

        finished_list = db.get_all_success(table_idx)
        finished_list = map(lambda x:(x[0],),finished_list)
        failed_list = db.get_all_failed(table_idx)
        failed_list = map(lambda x:(x[0],), failed_list)
        if FLAGS.retry_failed:
            rest_list = list(set(download_list) - set(finished_list) | set(failed_list))
        else:
            rest_list = list(set(download_list) - set(finished_list) - set(failed_list))

        total = len(set(download_list))
        finished = len(set(finished_list)-set(failed_list))
        failed = len(set(failed_list))

    elif func_name in ['reorder', 'dock']:
        rec_idx = table_param['receptor_idx']
        rec_list = db.get_all_success(rec_idx)

        lig_idx = table_param['ligand_idx']
        lig_list = db.get_all_success(lig_idx)

        finished_list = db.get_all_success(table_idx)
        failed_list = db.get_all_failed(table_idx)
        if FLAGS.retry_failed:
            rest_list = list(set(rec_list) & set(lig_list) - set(finished_list) | set(failed_list))
        else:
            rest_list = list(set(rec_list) & set(lig_list) - set(finished_list) - set(failed_list))

        total = len(set(rec_list) & set(lig_list))
        finished = len(set(finished_list)-set(failed_list))
        failed = len(set(failed_list))

    elif func_name in ['rmsd', 'overlap']:
        cry_idx = table_param['crystal_idx']
        cry_list = db.get_all_success(cry_idx)

        doc_idx = table_param['docked_idx']
        doc_list = db.get_all_success(doc_idx)

        finished_list = db.get_all_success(table_idx)
        finished_list = map(lambda x: x[:-1], finished_list)
        failed_list = db.get_all_failed(table_idx)
        failed_list = map(lambda x: x[:-1], failed_list)
        if FLAGS.retry_failed:
            rest_list = list(set(cry_list) & set(doc_list) - set(finished_list) | set(failed_list))
        else:
            rest_list = list(set(cry_list) & set(doc_list) - set(finished_list) - set(failed_list))

        total = len(set(cry_list) & set(doc_list))
        finished = len(set(finished_list)-set(failed_list))
        failed = len(set(failed_list))

    elif func_name == 'native_contact':
        rec_idx = table_param['receptor_idx']
        rec_list = db.get_all_success(rec_idx)

        cry_idx = table_param['crystal_idx']
        cry_list = db.get_all_success(cry_idx)

        doc_idx = table_param['docked_idx']
        doc_list = db.get_all_success(doc_idx)

        finished_list = db.get_all_success(table_idx)
        finished_list = map(lambda x: x[:-1], finished_list)
        failed_list = db.get_all_failed(table_idx)
        failed_list = map(lambda x: x[:-1], failed_list)
        if FLAGS.retry_failed:
            rest_list = list(set(rec_list) & set(cry_list) & set(doc_list) - set(finished_list) | set(failed_list))
        else:
            rest_list = list(set(rec_list) & set(cry_list) & set(doc_list) - set(finished_list) - set(failed_list))

        total = len(set(rec_list) & set(cry_list) & set(doc_list))
        finished = len(set(finished_list)- set(failed_list))
        failed = len(set(failed_list))
    elif func_name == 'binding_affinity':
        
        finished_list = db.get_all_success(table_idx)
        failed_list = db.get_all_failed(table_idx)

        total = len(set(finished_list) | set(failed_list))
        finished = len(set(finished_list) - set(failed_list))
        failed = len(set(failed_list))

        # binding affinity finished at the first time it launched
        # no rest entry left to continue
        rest_list = [[]]

    else:
        raise Exception("unknown func_name %s" % func_name)

    if progress:
        return (total, finished, failed)
    else:
        return rest_list

def db_create():
    if FLAGS.action == 'download':
        if FLAGS.folder_name is None:
            raise Exception("folder_name required")

        folder_name = FLAGS.folder_name
        table_param = {
            'func':'download',
            'output_folder': folder_name,
        }


    elif FLAGS.action == 'split_receptor':
        if FLAGS.folder_name is None:
            raise Exception("folder_name required")
        if FLAGS.download_idx is None:
            raise Exception('download_idx required')

        folder_name = FLAGS.folder_name
        download_idx = FLAGS.download_idx
        download_folder = db.get_folder(download_idx)
        table_param = {
            'func':'split_receptor',
            'output_folder':folder_name,
            'download_idx':download_idx,
            'input_download_folder':'{}_{}'.format(download_idx, download_folder),
            'depend':[download_idx]
        }


    elif FLAGS.action == 'split_ligand':
        if FLAGS.folder_name is None:
            raise Exception("folder_name required")
        if FLAGS.download_idx is None:
            raise Exception('download_idx required')
        
        folder_name = FLAGS.folder_name
        download_idx = FLAGS.download_idx
        download_folder = db.get_folder(download_idx)
        table_param = {
            'func':'split_ligand',
            'output_folder': folder_name,
            'download_idx': download_idx,
            'input_download_folder': '{}_{}'.format(download_idx, download_folder),
            'depend':[download_idx],
            'fit_box_size':20
        } 


        
    elif FLAGS.action == 'reorder':
        if FLAGS.folder_name is None:
            raise Exception("folder_name required")
        if FLAGS.receptor_idx is None:
            raise Exception('receptor_idx required')
        if FLAGS.ligand_idx is None:
            raise Exception('ligand_idx required')

        folder_name = FLAGS.folder_name
        receptor_idx = FLAGS.receptor_idx
        receptor_folder = db.get_folder(receptor_idx)
        ligand_idx = FLAGS.ligand_idx
        ligand_folder = db.get_folder(ligand_idx)
        table_param = {
            'func': 'reorder',
            'output_folder': folder_name,
            'receptor_idx':receptor_idx,
            'input_receptor_folder':'{}_{}'.format(receptor_idx,receptor_folder),
            'ligand_idx': ligand_idx,
            'input_ligand_folder': '{}_{}'.format(ligand_idx, ligand_folder),
            'depend':[receptor_idx, ligand_idx],
            'smina_param':config.smina_dock_pm['reorder']
        }


    elif FLAGS.action == 'smina_dock':
        if FLAGS.folder_name is None:
            raise Exception("folder_name required")
        if FLAGS.receptor_idx is None:
            raise Exception('receptor_idx required')
        if FLAGS.ligand_idx is None:
            raise Exception('ligand_idx required')
        if FLAGS.param is None:
            raise Exception('param required')

        dock_param = FLAGS.param
        if not dock_param in config.smina_dock_pm.keys():
            raise KeyError("dock param {} doesn't exists. ".format(dock_param)\
                            + "available options are: {}".format(', '.join(config.smina_dock_pm.keys())))
        dock_param = config.smina_dock_pm[dock_param]
        folder_name = FLAGS.folder_name
        receptor_idx = FLAGS.receptor_idx
        receptor_folder = db.get_folder(receptor_idx)
        ligand_idx = FLAGS.ligand_idx
        ligand_folder = db.get_folder(ligand_idx)
        table_param = {
            'func': 'smina_dock',
            'output_folder': folder_name,
            'receptor_idx':receptor_idx,
            'input_receptor_folder': '{}_{}'.format(receptor_idx, receptor_folder),
            'ligand_idx': ligand_idx,
            'input_ligand_folder': '{}_{}'.format(ligand_idx, ligand_folder),
            'depend':[receptor_idx, ligand_idx],
            'smina_param':dock_param
        }

    
    elif FLAGS.action == 'rmsd':
        if FLAGS.crystal_idx is None:
            raise Exception('crystal_idx required')
        if FLAGS.docked_idx is None:
            raise Exception('docked_idx required')

        crystal_idx = FLAGS.crystal_idx
        crystal_folder = db.get_folder(crystal_idx)
        docked_idx = FLAGS.docked_idx
        docked_folder = db.get_folder(docked_idx)
        table_param = {
            'func':'rmsd',
            'crystal_idx': crystal_idx,
            'input_crystal_folder':'{}_{}'.format(crystal_idx, crystal_folder),
            'docked_idx': docked_idx,
            'input_docked_folder':'{}_{}'.format(docked_idx, docked_folder),
            'depend':[crystal_idx, docked_idx]
        }


    elif FLAGS.action == 'overlap':
        if FLAGS.crystal_idx is None:
            raise Exception('crystal_idx require')
        if FLAGS.docked_idx is None:
            raise Exception('docked_idx required')
        if FLAGS.param is None:
            raise Exception('param required')
        overlap_param = FLAGS.param
        if not overlap_param in config.overlap_pm.keys():
            raise KeyError("dock param {} doesn't exists. ".format(overlap_param) \
                           + "available options are: {}".format(', '.join(config.overlap_pm.keys())))

        overlap_param = config.overlap_pm[overlap_param]
        crystal_idx = FLAGS.crystal_idx
        crystal_folder = db.get_folder(crystal_idx)
        docked_idx = FLAGS.docked_idx
        docked_folder = db.get_folder(docked_idx)
        
        
        table_param = {
            'func':'overlap',
            'crystal_idx': crystal_idx,
            'input_crystal_folder':'{}_{}'.format(crystal_idx, crystal_folder),
            'docked_idx': docked_idx,
            'input_docked_folder':'{}_{}'.format(docked_idx, docked_folder),
            'depend':[crystal_idx, docked_idx],
        }
        table_param.update(overlap_param)
        


    elif FLAGS.action == 'native_contact':
        if FLAGS.receptor_idx is None:
            raise Exception('receptor_idx required')
        if FLAGS.crystal_idx is None:
            raise Exception('crystal_idx require')
        if FLAGS.docked_idx is None:
            raise Exception('docked_idx required')
        if FLAGS.param is None:
            raise Exception('param required')
        native_contact_param = FLAGS.param
        if not native_contact_param in config.native_contact_pm.keys():
            raise KeyError("dock param {} doesn't exists. ".format(native_contact_param) \
                           + "available options are: {}".format(', '.join(config.native_contact_pm.keys())))

        native_contact_param = config.native_contact_pm[native_contact_param]

        receptor_idx = FLAGS.receptor_idx
        receptor_folder = db.get_folder(receptor_idx)
        crystal_idx = FLAGS.crystal_idx
        crystal_folder = db.get_folder(crystal_idx)
        docked_idx = FLAGS.docked_idx
        docked_folder = db.get_folder(docked_idx)
        table_param = {
            'func':'native_contact',
            'receptor_idx': receptor_idx,
            'input_receptor_folder':'{}_{}'.format(receptor_idx, receptor_folder),
            'crystal_idx': crystal_idx,
            'input_crystal_folder':'{}_{}'.format(crystal_idx, crystal_folder),
            'docked_idx': docked_idx,
            'input_docked_folder':'{}_{}'.format(docked_idx, docked_folder),
            'depend': [receptor_idx, crystal_idx, docked_idx],
        }

        table_param.update(native_contact_param)

    elif FLAGS.action == 'binding_affinity':
        if FLAGS.param is None:
            raise Exception('param required')

        bind_param = FLAGS.param
        if not bind_param in config.bind_pm.keys():
            raise Exception('No binidng affinity file for key {} in config\n'.format(bind_param)\
                             + 'Available choices are {}'.format(str(config.bind_pm.keys())))

        
        table_param = {
            'func':'binding_affinity',
            'bind_param': config.bind_pm[bind_param]
        }

    else:
        raise Exception("Doesn't support action {}".format(FLAGS.action))


    func_name = table_param['func']
    func = DatabaseAction[func_name]
    if func_name == 'smina_dock':
        table_type = 'docked_ligand'
        data_type = 'dock'
    elif func_name == 'reorder':
        table_type = 'reorder_ligand'
        data_type = 'reorder'
    else:
        table_type = func_name
        data_type = func_name

    table_idx = db.create_table(table_type, table_param)

    data = get_job_data(data_type, table_idx, table_param)
    run_multiprocess(data, partial(func, table_idx, table_param))


def db_continue():
    if FLAGS.table_idx is None:
        raise Exception("table_idx required")


    table_idx = FLAGS.table_idx
    table_name, table_param = db.get_table(table_idx)

    func_name = table_param['func']
    func = DatabaseAction[func_name]
    if func_name == 'smina_dock':
        table_type = 'docked_ligand'
        data_type = 'dock'
    elif func_name == 'reorder':
        table_type = 'reorder_ligand'
        data_type = 'reorder'
    else:
        table_type = func_name
        data_type = func_name

    data = get_job_data(data_type, table_idx, table_param)
    run_multiprocess(data, partial(func, table_idx, table_param))

def db_delete():
    if FLAGS.table_idx is None:
        raise Exception('table_idx required')

    table_idx = FLAGS.table_idx
    db.delete_table(table_idx)

def db_progress():
    if FLAGS.table_idx is None:
        raise Exception('table_idx required')
    
    table_idx = FLAGS.table_idx

    if table_idx:
        table_idxes = [table_idx]
    else:
        table_idxes = sorted(db.get_all_dix())


    print("Progress\n")
    if len(table_idxes):
        print("Total jobs |  Finished  | Finished(%) |   Failed   |  Failed(%)  |   Remain   |  Remain(%)  | Table name ")
    for table_idx in table_idxes:
        table_name, table_param = db.get_table(table_idx)
        
        func_name = table_param['func']
        if func_name == 'smina_dock':
            data_type = 'dock'
        elif func_name == 'reorder':
            data_type='reorder'
        else:
            data_type = func_name

        
        
        total, finished, failed = get_job_data(data_type, table_idx, table_param, progress=True)
        print( "{:<13d} {:<11d} {:<15.2f} {:<11d} {:<14.2f} {:<11d} {:<12.2f} {}". \
                format(total,
                       finished, 100.*finished/total  if total else 0,
                       failed, 100.*failed/total if total else 0,
                       total - finished - failed, 100.*(total-finished-failed)/total if total else 0,
                       table_name))


def db_param():
    if FLAGS.table_idx is None:
        raise Exception('table_idx required')

    table_idx = FLAGS.table_idx

    table_name, table_param = db.get_table(table_idx)

    print("Parameter for Table: {}".format(table_name))
    pprint.pprint(table_param)


def main():
    if FLAGS.db_create:
        db_create()
    if FLAGS.db_continue:
        db_continue()
    if FLAGS.db_delete:
        db_delete()
    if FLAGS.db_progress:
        db_progress()
    if FLAGS.db_param:
        db_param()


if __name__ == '__main__':
    FLAGS = get_arguments()
    main()
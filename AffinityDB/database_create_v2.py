from __future__ import print_function

import os
import sys
import re
import time
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
    parser.add_argument('--action', type=str)
    parser.add_argument('--dock_param', type=str, default='vinardo')
    parser.add_argument('--overlap_param', type=str, default='default')
    parser.add_argument('--native_contact_param', type=str, default='default')
    parser.add_argument('--retry_failed', action='store_true')
    parser.add_argument('--folder_name', type=str)
    parser.add_argument('--table_sn', type=int)
    parser.add_argument('--receptor_sn', type=int)
    parser.add_argument('--ligand_sn', type=int)
    parser.add_argument('--crystal_sn', type=int)
    parser.add_argument('--docked_sn', type=int)
    parser.add_argument('--download_sn', type=int)

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

        pool = multiprocessing.Pool(config.process_num)
        pool.map_async(func, target_list)
        pool.close()
        pool.join()
        
        #map(func, target_list)


    


def get_job_data(func_name, table_sn, table_param, progress=False):
    
    if func_name == 'download':
        download_list = open(config.list_of_PDBs_to_download).readline().strip().split(', ')
        finished_list = db.get_all_success(table_sn)
        failed_list = db.get_all_failed(table_sn)
        if FLAGS.retry_failed:
            rest_list = list(set(download_list) - set(finished_list) | set(failed_list))
        else:
            rest_list =list(set(download_list) - set(finished_list) - set(failed_list))

        total = len(set(download_list))
        finished = len(set(finished_list)-set(failed_list))
        failed = len(set(failed_list))

    elif func_name in ['split_ligand','split_receptor']:
        download_sn = table_param['download_sn']
        download_list = db.get_all_success(download_sn)

        finished_list = db.get_all_success(table_sn)
        finished_list = map(lambda x:(x[0],),finished_list)
        failed_list = db.get_all_failed(table_sn)
        failed_list = map(lambda x:(x[0],), failed_list)
        if FLAGS.retry_failed:
            rest_list = list(set(download_list) - set(finished_list) | set(failed_list))
        else:
            rest_list = list(set(download_list) - set(finished_list) - set(failed_list))

        total = len(set(download_list))
        finished = len(set(finished_list)-set(failed_list))
        failed = len(set(failed_list))

    elif func_name in ['reorder', 'dock']:
        rec_sn = table_param['receptor_sn']
        rec_list = db.get_all_success(rec_sn)

        lig_sn = table_param['ligand_sn']
        lig_list = db.get_all_success(lig_sn)

        finished_list = db.get_all_success(table_sn)
        failed_list = db.get_all_failed(table_sn)
        if FLAGS.retry_failed:
            rest_list = list(set(rec_list) & set(lig_list) - set(finished_list) | set(failed_list))
        else:
            rest_lsit = list(set(rec_list) & set(lig_list) - set(finished_list) - set(failed_list))

        total = len(set(rec_list) & set(lig_list))
        finished = len(set(finished_list)-set(failed_list))
        failed = len(set(failed_list))

    elif func_name in ['rmsd', 'overlap']:
        cry_sn = table_param['crystal_sn']
        cry_list = db.get_all_success(cry_sn)

        doc_sn = table_param['docked_sn']
        doc_list = db.get_all_success(doc_sn)

        finished_list = db.get_all_success(table_sn)
        finished_list = map(lambda x: x[:-1], finished_list)
        failed_list = db.get_all_failed(table_sn)
        failed_list = map(lambda x: x[:-1], failed_list)
        if FLAGS.retry_failed:
            rest_list = list(set(cry_list) & set(doc_list) - set(finished_list) | set(failed_list))
        else:
            rest_list = list(set(cry_list) & set(doc_list) - set(finished_list) - set(failed_list))

        total = len(set(cry_list) & set(doc_list))
        finished = len(set(finished_list)-set(failed_list))
        failed = len(set(failed_list))

    elif func_name == 'native_contact':
        rec_sn = table_param['receptor_sn']
        rec_list = db.get_all_success(rec_sn)

        cry_sn = table_param['crystal_sn']
        cry_list = db.get_all_success(cry_sn)

        doc_sn = table_param['docked_sn']
        doc_list = db.get_all_success(doc_sn)

        finished_list = db.get_all_success(table_sn)
        finished_list = map(lambda x: x[:-1], finished_list)
        failed_list = db.get_all_failed(table_sn)
        failed_list = map(lambda x: x[:-1], failed_list)
        if FLAGS.retry_failed:
            rest_list = list(set(rec_list) & set(cry_list) & set(doc_list) - set(finished_list) | set(failed_list))
        else:
            rest_list = list(set(rec_list) & set(cry_list) & set(doc_list) - set(finished_list) - set(failed_list))

        total = len(set(rec_list) & set(cry_list) & set(doc_list))
        finished = len(set(finished_list)- set(failed_list))
        failed = len(set(failed_list))

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
            'folder': folder_name,
        }


    elif FLAGS.action == 'split_receptor':
        if FLAGS.folder_name is None:
            raise Exception("folder_name required")
        if FLAGS.download_sn is None:
            raise Exception('download_sn required')

        folder_name = FLAGS.folder_name
        download_sn = FLAGS.download_sn
        download_folder = db.get_folder(download_sn)
        table_param = {
            'func':'split_receptor',
            'folder':folder_name,
            'download_sn':download_sn,
            'download_folder':'{}_{}'.format(download_sn, download_folder),
            'depend':[download_sn]
        }


    elif FLAGS.action == 'split_ligand':
        if FLAGS.folder_name is None:
            raise Exception("folder_name required")
        if FLAGS.download_sn is None:
            raise Exception('download_sn required')
        
        folder_name = FLAGS.folder_name
        download_sn = FLAGS.download_sn
        download_folder = db.get_folder(download_sn)
        table_param = {
            'func':'split_ligand',
            'folder': folder_name,
            'download_sn': download_sn,
            'download_folder': '{}_{}'.format(download_sn, download_folder),
            'depend':[download_sn]
        } 


        
    elif FLAGS.action == 'reorder':
        if FLAGS.folder_name is None:
            raise Exception("folder_name required")
        if FLAGS.receptor_sn is None:
            raise Exception('receptor_sn required')
        if FLAGS.ligand_sn is None:
            raise Exception('ligand_sn required')

        folder_name = FLAGS.folder_name
        receptor_sn = FLAGS.receptor_sn
        receptor_folder = db.get_folder(receptor_sn)
        ligand_sn = FLAGS.ligand_sn
        ligand_folder = db.get_folder(ligand_sn)
        table_param = {
            'func': 'reorder',
            'folder': folder_name,
            'receptor_sn':receptor_sn,
            'receptor_folder':'{}_{}'.format(receptor_sn,receptor_folder),
            'ligand_sn': ligand_sn,
            'ligand_folder': '{}_{}'.format(ligand_sn, ligand_folder),
            'depend':[receptor_sn, ligand_sn],
            'smina_param':config.smina_dock_pm['reorder']
        }


    elif FLAGS.action == 'smina_dock':
        if FLAGS.folder_name is None:
            raise Exception("folder_name required")
        if FLAGS.receptor_sn is None:
            raise Exception('receptor_sn required')
        if FLAGS.ligand_sn is None:
            raise Exception('ligand_sn required')
        if FLAGS.dock_param is None:
            raise Exception('dock_param required')

        dock_param = FLAGS.dock_param
        if not dock_param in config.smina_dock_pm.keys():
            raise KeyError("dock param {} doesn't exists. ".format(dock_param)\
                            + "available options are: {}".format(', '.join(config.smina_dock_pm.keys())))
        dock_param = config.smina_dock_pm[dock_param]
        folder_name = FLAGS.folder_name
        receptor_sn = FLAGS.receptor_sn
        receptor_folder = db.get_folder(receptor_sn)
        ligand_sn = FLAGS.ligand_sn 
        ligand_folder = db.get_folder(ligand_sn)
        table_param = {
            'func': 'smina_dock',
            'folder': folder_name,
            'receptor_sn':receptor_sn,
            'receptor_folder': '{}_{}'.format(receptor_sn, receptor_folder),
            'ligand_sn': ligand_sn,
            'ligand_folder': '{}_{}'.format(ligand_sn, ligand_folder),
            'depend':[receptor_sn, ligand_sn],
            'smina_param':dock_param
        }

    
    elif FLAGS.action == 'rmsd':
        if FLAGS.crystal_sn is None:
            raise Exception('crystal_sn required')
        if FLAGS.docked_sn is None:
            raise Exception('docked_sn required')

        crystal_sn = FLAGS.crystal_sn
        crystal_folder = db.get_folder(crystal_sn)
        docked_sn = FLAGS.docked_sn
        docked_folder = db.get_folder(docked_sn)
        table_param = {
            'func':'rmsd',
            'crystal_sn': crystal_sn,
            'crystal_folder':'{}_{}'.format(crystal_sn, crystal_folder),
            'docked_sn': docked_sn,
            'docked_folder':'{}_{}'.format(docked_sn, docked_folder),
            'depend':[crystal_sn, docked_sn]
        }


    elif FLAGS.action == 'overlap':
        if FLAGS.crystal_sn is None:
            raise Exception('crystal_sn require')
        if FLAGS.docked_sn is None:
            raise Exception('docked_sn required')

        crystal_sn = FLAGS.crystal_sn
        crystal_folder = db.get_folder(crystal_sn)
        docked_sn = FLAGS.docked_sn
        docked_folder = db.get_folder(docked_sn)
        table_param = {
            'func':'overlap',
            'crystal_sn': crystal_sn,
            'crystal_folder':'{}_{}'.format(crystal_sn, crystal_folder),
            'docked_sn': docked_sn,
            'docked_folder':'{}_{}'.format(docked_sn, docked_folder),
            'depend':[crystal_sn, docked_sn],
            'clash_cutoff_A':4.0,
            'clash_size_cutoff':0.3
        }


    elif FLAGS.action == 'native_contact':
        if FLAGS.receptor_sn is None:
            raise Exception('receptor_sn required')
        if FLAGS.crystal_sn is None:
            raise Exception('crystal_sn require')
        if FLAGS.docked_sn is None:
            raise Exception('docked_sn required')

        receptor_sn = FLAGS.receptor_sn
        receptor_folder = db.get_folder(receptor_sn)
        crystal_sn = FLAGS.crystal_sn
        crystal_folder = db.get_folder(crystal_sn)
        docked_sn = FLAGS.docked_sn
        docked_folder = db.get_folder(docked_sn)
        table_param = {
            'func':'native_contact',
            'receptor_sn': receptor_sn,
            'receptor_folder':'{}_{}'.format(receptor_sn, receptor_folder),
            'crystal_sn': crystal_sn,
            'crystal_folder':'{}_{}'.format(crystal_sn, crystal_folder),
            'docked_sn': docked_sn,
            'docked_folder':'{}_{}'.format(docked_sn, docked_folder),
            'depend': [receptor_sn, crystal_sn, docked_sn],
            'distance_threshold': 4.0
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

    table_sn = db.create_table(table_type, table_param)

    data = get_job_data(data_type, table_sn, table_param)
    run_multiprocess(data, partial(func, table_sn, table_param))


def db_continue():
    if FLAGS.table_sn is None:
        raise Exception("table_sn required")


    table_sn = FLAGS.table_sn
    table_name, table_param = db.get_table(table_sn)

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

    data = get_job_data(data_type, table_sn, table_param)
    run_multiprocess(data, partial(func, table_sn, table_param))

def db_delete():
    if FLAGS.table_sn is None:
        raise Exception('table_sn required')

    table_sn = FLAGS.table_sn
    db.delete_table(table_sn)

def db_progress():
    if FLAGS.table_sn is None:
        raise Exception('table_sn required')
    
    table_sn = FLAGS.table_sn

    if table_sn:
        table_sns = [table_sn]
    else:
        table_sns = sorted(db.get_all_sns())

    print("Progress\n")
    if len(table_sns):
        print("Total jobs |  Finished  | Finished(%) |   Failed   |  Failed(%)  |   Remain   |  Remain(%)  | Table name ")
    for table_sn in table_sns:
        table_name, table_param = db.get_table(table_sn)
        
        func_name = table_param['func']
        func = DatabaseAction[func_name]
        if func_name == 'smina_dock':
            table_type = 'docked_ligand'
            data_type = 'dock'
        elif func_name == 'reorder':
            table_type = 'reorder_ligand'
            data_type='reorder'
        else:
            table_type = func_name
            data_type = func_name

        
        
        total, finished, failed = get_job_data(data_type, table_sn, table_param, progress=True)
        print( "{:<13d} {:<11d} {:<15.2f} {:<11d} {:<14.2f} {:<11d} {:<12.2f} {}". \
                format(total,
                       finished, 100.*finished/total  if total else 0,
                       failed, 100.*failed/total if total else 0,
                       total - finished - failed, 100.*(total-finished-failed)/total if total else 0,
                       table_name))


def main():
    if FLAGS.db_create:
        db_create()
    if FLAGS.db_continue:
        db_continue()
    if FLAGS.db_delete:
        db_delete()
    if FLAGS.db_progress:
        db_progress()


if __name__ == '__main__':
    FLAGS = get_arguments()
    main()
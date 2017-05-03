"""
Processing data
"""

import os
import sys
import re 
import time
import subprocess
from functools import partial
from glob import glob
from utils import log, smina_param, timeit, count_lines
import numpy as np 
#import openbabel
import prody
import config
from config import data_dir
from db_v2 import AffinityDatabase
from parse_binding_DB import read_PDB_bind

db = AffinityDatabase()

def _makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download(table_idx, param, input_data):                                                                                    # todo (maksym) remove all datums
    try:                                                                                                                # folder = output_folder
        receptor = input_data                                                                                                # datum = pdb_id
        output_folder = param['output_folder']                                                                                        # papram = params
        output_folder_name = '{}_{}'.format(table_idx, output_folder)                                                                  # todo: sn == idx (everywhere)
        dest_dir = os.path.join(data_dir, output_folder_name)
        _makedir(dest_dir)

        pdb_path = os.path.join(dest_dir, receptor+'.pdb')
        if not os.path.exists(pdb_path):
            download_address = 'https://files.rcsb.org/download/{}.pdb'.format(receptor)
            os.system('wget -P {} {}'.format(dest_dir, download_address))

        header = prody.parsePDBHeader(pdb_path)

        record = [receptor, header['experiment'], header['resolution'], 1, 'success']                                     # datum = success report
        records = [record]
        db.insert(table_idx, records)                                                                                       # db.insert(success_report)
    except Exception as e:
        record = [input_data, 'unk', 0, 0, str(e)]
        records = [record]
        db.insert(table_idx, records)


def split_ligand(table_idx, param, input_data):
    try:
        if type(input_data).__name__ in ['tuple', 'list']:
            input_data = input_data[0]                                                                                            # do not allow x = x[0]
        
        receptor = input_data

        fit_box_size = param['fit_box_size']

        output_folder = param['output_folder']                                                                                        # which folder ? output_folder
        output_folder = '{}_{}'.format(table_idx, output_folder)                                                                       # all table_sn become table_idx
        input_download_folder = param['input_download_folder']                                                                      # rename all these into standard "source folder"
        pdb_dir = os.path.join(data_dir, input_download_folder)                                                               # download_folder = source_folder
        pdb_path = os.path.join(pdb_dir, receptor+'.pdb')
        
        parsed_pdb = prody.parsePDB(pdb_path)                                                                               # parsed = parsed_pdb
        parsed_header = prody.parsePDBHeader(pdb_path)                                                                         # parsed
        
        output_lig_dir = os.path.join(data_dir, output_folder, receptor)                                                              # data_dir as not an argument of the function (should come as an argument)
        _makedir(output_lig_dir)

        ligands = []
        for chem in parsed_header['chemicals']:                                                               # ligands = ligands_in_pdb
            ligands.append([chem.chain, str(chem.resnum), chem.resname])



        for chain, resnum, resname in ligands:
            try:
                lig = parsed_pdb.select('chain {} resnum {}'.format(chain, resnum))
                heavy_lig = lig.select('not hydrogen')
                heavy_atom = heavy_lig.numAtoms()
                heavy_coord =heavy_lig.getCoords()
                max_size_on_axis = max(heavy_coord.max(axis=0) - heavy_coord.min(axis=0))
                


                lig_name = '_'.join([receptor,chain,resnum,resname,'ligand']) + '.pdb'
                prody.writePDB(os.path.join(output_lig_dir, lig_name), lig)


                record = [receptor, chain, resnum, resname, heavy_atom, max_size_on_axis, 1, 'success']                                     # data = success_message
                records = [record]
                db.insert(table_idx, records)
            except Exception as e:
                record =  [receptor, chain, resnum, resname, 0, 0, 0, str(e)]                                                # data = failure_message
                records = [record]
                db.insert(table_idx, records)

    except Exception as e:
        print e
        raise Exception(str(e))

def split_receptor(table_idx, param, datum):                                                                             # param = params;
    try:                                                                                                                # datum = pdb_name
        if type(datum).__name__ in ['tuple','list']:
            datum = datum[0]

        receptor = datum                                                                                                # receptor = pdb_name
        output_folder = param['output_folder']
        output_folder = '{}_{}'.format(table_idx, output_folder)
        input_download_folder = param['input_download_folder']
        
        input_pdb_dir = os.path.join(data_dir,input_download_folder)                                                                # pdb_dir = input_dir
        input_pdb_path = os.path.join(input_pdb_dir, receptor+'.pdb')
        
        parsed_pdb = prody.parsePDB(input_pdb_path)
        parsed_header = prody.parsePDBHeader(input_pdb_path)

        output_rec_dir = os.path.join(data_dir, output_folder, receptor)
        _makedir(output_rec_dir)

        ligands = []
        for chem in parsed_header['chemicals']:
            chain, resnum, resname = chem.chain, chem.resnum, chem.resname
            ligands.append([chain, str(resnum), resname])
        
        for chain, resnum, resname in ligands:
            try:
                rec = parsed_pdb.select('not (chain {} resnum {})'.format(chain, resnum))
                rec = rec.select('not water')
                heavy_atom = rec.select('not hydrogen').numAtoms()
                rec_name = '_'.join([receptor, chain, resnum, resname, 'receptor']) + '.pdb'
                prody.writePDB(os.path.join(output_rec_dir, rec_name), rec)


                record = [receptor, chain, resnum, resname, heavy_atom, parsed_header['experiment'], parsed_header['resolution'] , 1 , 'success'] # datum = success_message
                records = [record]
                db.insert(table_idx, records)
            except Exception as e:
                record = [receptor, chain, resnum, resname, 0, 0, str(e)]                                                # datum = failure_message
                records = [record]
                db.insert(table_idx, records) 

    except Exception as e:
        print e
        raise Exception(str(e))


def reorder(table_idx, param, input_data):
    try:
        receptor, chain, resnum, resname = input_data
        
        output_folder = param['output_folder']
        output_folder = '{}_{}'.format(table_idx, output_folder)
        input_lig_folder = param['input_ligand_folder']
        input_rec_folder = param['input_receptor_folder']
        smina_pm = smina_param()
        smina_pm.param_load(param['smina_param'])

        out_dir = os.path.join(data_dir, output_folder, receptor)
        _makedir(out_dir)
        out_name = '_'.join(input_data + ['ligand']) + '.pdb'
        out_path = os.path.join(out_dir, out_name)

        input_lig_dir = os.path.join(data_dir, input_lig_folder, receptor)                                                         # lig_dir = input_lig_dir
        lig_name = '_'.join(input_data + ['ligand']) + '.pdb'
        input_lig_path = os.path.join(input_lig_dir, lig_name)

        input_rec_dir = os.path.join(data_dir, input_rec_folder, receptor)                                                          # rec_dir = input_rec_dir
        rec_name = '_'.join(input_data + ['receptor']) + '.pdb'
        input_rec_path = os.path.join(input_rec_dir, rec_name)

        kw = {
            'receptor': input_rec_path,
            'ligand': input_lig_path,
            'autobox_ligand':input_lig_path,
            'out':out_path
        }


        cmd = smina_pm.make_command(**kw)
        #print cmd                                                                                                       # print "smina parameters for reordering:', cmd
        cl = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        cl.wait()
        prody.parsePDB(out_path)

        record = input_data + [1, 'success']                                                                                 # datum = success_message
        records = [record]
        db.insert(table_idx, records)

    except Exception as e:
        record = input_data + [0, str(e)]
        records = [record]
        db.insert(table_idx, records)

def smina_dock(table_idx, param, input_data):
    try:
        receptor, chain, resnum, resname = input_data
        
        output_folder = param['output_folder']                                                                                        # folder = output_folder
        output_folder = '{}_{}'.format(table_idx, output_folder)
        input_lig_folder = param['input_ligand_folder']
        input_rec_folder = param['input_receptor_folder']
        smina_pm = smina_param()
        smina_pm.param_load(param['smina_param'])

        out_dir = os.path.join(data_dir, output_folder, receptor)
        _makedir(out_dir)
        out_name = '_'.join(input_data + ['ligand']) + '.pdb'
        out_path = os.path.join(out_dir, out_name)

        input_lig_dir = os.path.join(data_dir, input_lig_folder , receptor)                                                         # lig_dir = input_lig_dir
        lig_name = '_'.join(input_data + ['ligand']) + '.pdb'
        input_lig_path = os.path.join(input_lig_dir, lig_name)

        input_rec_dir = os.path.join(data_dir, input_rec_folder, receptor)
        rec_name = '_'.join(input_data + ['receptor']) + '.pdb'
        input_rec_path = os.path.join(input_rec_dir, rec_name)

        kw = {
            'receptor': input_rec_path,
            'ligand': input_lig_path,
            'autobox_ligand':input_lig_path,
            'out':out_path
        }


        cmd = smina_pm.make_command(**kw)
        #print cmd
        cl = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        cl.wait()
        prody.parsePDB(out_path)

        record = input_data + [1, 'success']
        records = [record]
        db.insert(table_idx, records)

    except Exception as e:
        record = input_data + [0, str(e)]
        records = [record]
        db.insert(table_idx, records)

def overlap(table_idx, param, input_data):                                                                                    # a prticularly bad datum

    try:
        receptor, chain, resnum, resname = input_data
        input_docked_folder = param['input_docked_folder']
        input_crystal_folder = param['input_crystal_folder']
        clash_cutoff_A = param['clash_cutoff_A']                                                                        #
        #clash_size_cutoff                                                                                              # make sure we compute a real number


        lig_name = '_'.join([receptor, chain, resnum, resname, 'ligand']) + '.pdb'
        input_docked_dir = os.path.join(data_dir, input_docked_folder, receptor)
        input_docked_path = os.path.join(input_docked_dir, lig_name)

        input_crystal_dir = os.path.join(data_dir, input_crystal_folder, receptor)
        input_crystal_path = os.path.join(input_crystal_dir, lig_name)

        docked_coords = prody.parsePDB(input_docked_path).getCoordsets()                                                             # docked = docked_coords
        crystal_coords = prody.parsePDB(input_crystal_path).getCoords()                                                              # crystal = prody_crystal
                                                                                                                        # the reason is to know that this thing is an object
        
        expanded_docked = np.expand_dims(docked_coords, -2)                                                                    # 1
        diff = expanded_docked - crystal_coords                                                                                # 2
        distance = np.sqrt(np.sum(np.power(diff, 2), axis=-1))                                                          # 3 in one line


                                                                                                                        # !!!!!! Formula is not correct                                                                               # sum = min
        all_clash = (distance < clash_cutoff_A).astype(float)                                                    #1
        atom_clash = (np.sum(all_clash, axis=-1) > 0).astype(float)                                                     #2
        position_clash_ratio = np.mean(atom_clash, axis=-1)                                                             #3 : 1,2,3 in one line

        records = []
        for i, ratio in enumerate(position_clash_ratio):
            records.append(input_data + [i + 1, ratio, 1, 'success'])

        db.insert(table_idx, records)

    except Exception as e:
        record = input_data + [1, 0, 0, str(e)]
        records = [record]
        db.insert(table_idx, records)                                                                                       # failure mssg

def rmsd(table_idx, param, input_data):
    
    try:
        receptor, chain, resnum, resname = input_data
        input_docked_folder = param['input_docked_folder']
        input_crystal_folder = param['input_crystal_folder']
        lig_name = '_'.join([receptor, chain, resnum, resname, 'ligand']) + '.pdb'
        

        input_docked_dir = os.path.join(data_dir,input_docked_folder, receptor)
        input_docked_path = os.path.join(input_docked_dir, lig_name)

        input_crystal_dir = os.path.join(data_dir, input_crystal_folder, receptor)
        input_crystal_path = os.path.join(input_crystal_dir, lig_name)

        docked_coords = prody.parsePDB(input_docked_path).getCoordsets()
        crystal_coord = prody.parsePDB(input_crystal_path).getCoords()

        rmsd = np.sqrt(np.mean(np.sum(np.square(docked_coords - crystal_coord), axis=01), axis=-1))

        records = []
        for i, rd in enumerate(rmsd):
            records.append(input_data + [i + 1, rd, 1, 'success'])
        db.insert(table_idx, records)
    except Exception as e:
        record = input_data + [1, 0, 0, str(e)]
        records = [record]
        db.insert(table_idx, records)

def native_contact(table_idx, param, input_data):

    try:
        receptor, chain, resnum, resname = input_data
        input_docked_folder = param['input_docked_folder']
        input_crystal_folder = param['input_crystal_folder']
        input_rec_folder = param['input_receptor_folder']
        distance_threshold = param['distance_threshold']
        lig_name = '_'.join([receptor, chain, resnum, resname, 'ligand']) + '.pdb'
        rec_name = '_'.join([receptor, chain, resnum, resname, 'receptor']) + '.pdb'

        input_docked_dir = os.path.join(data_dir, input_docked_folder, receptor)
        input_docked_path = os.path.join(input_docked_dir, lig_name)

        input_crystal_dir = os.path.join(data_dir, input_crystal_folder, receptor)
        input_crystal_path = os.path.join(input_crystal_dir, lig_name)

        input_rec_dir = os.path.join(data_dir, input_rec_folder, receptor)
        input_rec_path = os.path.join(input_rec_dir, rec_name)

        parsed_docked =  prody.parsePDB(input_docked_path).select('not hydrogen')
        parsed_crystal = prody.parsePDB(input_crystal_path).select('not hydrogen')
        parsed_rec = prody.parsePDB(input_rec_path).select('not hydrogen')

        cry_atom_num = parsed_crystal.numAtoms()
        lig_atom_num = parsed_docked.numAtoms()

        assert cry_atom_num == lig_atom_num

        docked_coords = parsed_docked.getCoordsets()
        crystal_coord = parsed_crystal.getCoords()
        rec_coord = parsed_rec.getCoords()

        exp_crystal_coord = np.expand_dims(crystal_coord, -2)
        cry_diff = exp_crystal_coord - rec_coord
        cry_distance = np.sqrt(np.sum(np.square(cry_diff), axis=-1))

        exp_docked_coords = np.expand_dims(docked_coords, -2)
        docked_diff = exp_docked_coords - rec_coord
        docked_distance = np.sqrt(np.sum(np.square(docked_diff),axis=-1))

        cry_contact = (cry_distance < distance_threshold).astype(int)
        
        num_contact = np.sum(cry_contact).astype(float)

        lig_contact = (docked_distance < distance_threshold).astype(int)

        contact_ratio = np.sum(cry_contact * lig_contact, axis=(-1,-2)) / num_contact

        records = []
        for i , nt in enumerate(contact_ratio):
            records.append(input_data + [i + 1, nt, 1, 'success'])

        db.insert(table_idx, records)
    except Exception as e:
        record = input_data + [0, 0, 0, str(e)]
        records = [record]
        db.insert(table_idx, records)

def binding_affinity(table_idx, param, input_data):
    try:
        pdb_bind_index = param['pdb_bind_index']
        pdb_bind_index = config.binding_affinity_files[pdb_bind_index]
        PDB_bind = read_PDB_bind(pdb_bind_index=pdb_bind_index)
        records = [[PDB_bind.pdb_names[i].upper(), PDB_bind.ligand_names[i],
                 PDB_bind.log_affinities[i], PDB_bind.normalized_affinities[i],
                 1, 'success']
                 for i in range(len(PDB_bind.pdb_names))]
        db.insert(table_idx, records)
    except Exception as e:
        print (e)

DatabaseAction={
    'download':download,
    'split_ligand':split_ligand,
    'split_receptor':split_receptor,
    'reorder':reorder,
    'smina_dock':smina_dock,
    'overlap':overlap,
    'rmsd':rmsd,
    'native_contact':native_contact,
    'binding_affinity':binding_affinity
}
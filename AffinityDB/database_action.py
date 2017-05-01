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

def download(table_sn, param, datum):
    try:
        receptor = datum    
        folder = param['folder']
        folder_name = '{}_{}'.format(table_sn, folder)
        dest_dir = os.path.join(data_dir, folder_name)
        _makedir(dest_dir)

        pdb_path = os.path.join(dest_dir, receptor+'.pdb')
        if not os.path.exists(pdb_path):
            download_address = 'https://files.rcsb.org/download/{}.pdb'.format(receptor)
            os.system('wget -P {} {}'.format(dest_dir, download_address))
            
        parsed = prody.parsePDB(pdb_path)
        header = prody.parsePDBHeader(pdb_path)
        

        datum = [receptor,header['experiment'], header['resolution'], 1, 'success']
        data = [datum]
        db.insert(table_sn, data)
    except Exception as e:
        datum = [datum,'unk',0, 0, str(e)]
        data = [datum]
        db.insert(table_sn, data)


def split_ligand(table_sn, param, datum):
    try:
        if type(datum,).__name__ in ['tuple','list']:
            datum = datum[0]
        
        receptor = datum
        folder = param['folder']
        folder = '{}_{}'.format(table_sn, folder)
        download_folder = param['download_folder']
        pdb_dir = os.path.join(data_dir, download_folder)
        pdb_path = os.path.join(pdb_dir, receptor+'.pdb')
        
        parsed = prody.parsePDB(pdb_path)
        header = prody.parsePDBHeader(pdb_path)
        
        lig_dir = os.path.join(data_dir, folder, receptor)
        _makedir(lig_dir)

        ligands = []
        for chem in header['chemicals']:
            chain, resnum, resname = chem.chain, chem.resnum, chem.resname
            ligands.append([chain, str(resnum), resname])


        for chain, resnum, resname in ligands:
            try:
                lig = parsed.select('chain {} resnum {}'.format(chain, resnum))
                heavy_atom = lig.select('not hydrogen').numAtoms()
                lig_name = '_'.join([receptor,chain,resnum,resname,'ligand']) + '.pdb'
                prody.writePDB(os.path.join(lig_dir, lig_name), lig)

                data = [receptor, chain, resnum, resname, heavy_atom, 1, 'success']
                data = [data]
                db.insert(table_sn, data)
            except Exception as e:
                data =  [receptor, chain, resnum, resname, 0, 0, str(e)]
                data = [data]
                db.insert(table_sn, data)

    except Exception as e:
        print e
        raise Exception(str(e))

def split_receptor(table_sn, param, datum):
    try:
        if type(datum).__name__ in ['tuple','list']:
            datum = datum[0]

        receptor = datum
        folder = param['folder']
        folder = '{}_{}'.format(table_sn, folder)
        download_folder = param['download_folder']
        
        pdb_dir = os.path.join(data_dir,download_folder)
        pdb_path = os.path.join(pdb_dir, receptor+'.pdb')
        
        parsed = prody.parsePDB(pdb_path)
        header = prody.parsePDBHeader(pdb_path)
        
        
        rec_dir = os.path.join(data_dir, folder, receptor)
        _makedir(rec_dir)

        ligands = []
        for chem in header['chemicals']:
            chain, resnum, resname = chem.chain, chem.resnum, chem.resname
            ligands.append([chain, str(resnum), resname])
        
        for chain, resnum, resname in ligands:
            try:
                rec = parsed.select('not (chain {} resnum {})'.format(chain, resnum))
                rec = rec.select('not water')
                heavy_atom = rec.select('not hydrogen').numAtoms()
                rec_name = '_'.join([receptor, chain, resnum, resname, 'receptor']) + '.pdb'
                prody.writePDB(os.path.join(rec_dir, rec_name), rec)


                datum = [receptor, chain, resnum, resname, heavy_atom, header['experiment'], header['resolution'] , 1 , 'success']
                data = [datum]
                db.insert(table_sn, data)
            except Exception as e:
                datum = [receptor, chain, resnum, resname, 0, 0, str(e)]
                data = [datum]
                db.insert(table_sn, data) 

    except Exception as e:
        print e
        raise Exception(str(e))

def lig_fit(table_sn, param, datum):
    try:
        receptor, chain, resnum, resname = datum

        lig_folder = param['ligand_folder']
        box_size = param['box_size']
        lig = np.load
    except:
        pass

def reorder(table_sn, param, datum):
    try:
        receptor, chain, resnum, resname = datum
        
        folder = param['folder']
        folder = '{}_{}'.format(table_sn, folder)
        lig_folder = param['ligand_folder']
        rec_folder = param['receptor_folder']
        smina_pm = smina_param()
        smina_pm.param_load(param['smina_param'])

        out_dir = os.path.join(data_dir, folder, receptor)
        _makedir(out_dir)
        out_name = '_'.join(datum +  ['ligand']) + '.pdb'
        out_path = os.path.join(out_dir, out_name)

        lig_dir = os.path.join(data_dir, lig_folder , receptor)
        lig_name = '_'.join(datum + ['ligand']) + '.pdb'
        lig_path = os.path.join(lig_dir, lig_name)

        rec_dir = os.path.join(data_dir, rec_folder, receptor)
        rec_name = '_'.join(datum + ['receptor']) + '.pdb'
        rec_path = os.path.join(rec_dir, rec_name)

        kw = {
            'receptor': rec_path,
            'ligand': lig_path,
            'autobox_ligand':lig_path,
            'out':out_path
        }


        cmd = smina_pm.make_command(**kw)
        print cmd
        cl = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        cl.wait()
        prody.parsePDB(out_path)

        datum = datum + [ 1, 'success']
        data = [datum]
        db.insert(table_sn, data)

    except Exception as e:
        datum = datum + [ 0, str(e)]
        data = [datum]
        db.insert(table_sn, data)      

def smina_dock(table_sn,param, datum):
    try:
        receptor, chain, resnum, resname = datum
        
        folder = param['folder']
        folder = '{}_{}'.format(table_sn, folder)
        lig_folder = param['ligand_folder']
        rec_folder = param['receptor_folder']
        smina_pm = smina_param()
        smina_pm.param_load(param['smina_param'])

        out_dir = os.path.join(data_dir, folder, receptor)
        _makedir(out_dir)
        out_name = '_'.join(datum +  ['ligand']) + '.pdb'
        out_path = os.path.join(out_dir, out_name)

        lig_dir = os.path.join(data_dir, lig_folder , receptor)
        lig_name = '_'.join(datum + ['ligand']) + '.pdb'
        lig_path = os.path.join(lig_dir, lig_name)

        rec_dir = os.path.join(data_dir, rec_folder, receptor)
        rec_name = '_'.join(datum + ['receptor']) + '.pdb'
        rec_path = os.path.join(rec_dir, rec_name)

        kw = {
            'receptor': rec_path,
            'ligand': lig_path,
            'autobox_ligand':lig_path,
            'out':out_path
        }


        cmd = smina_pm.make_command(**kw)
        print cmd
        cl = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        cl.wait()
        prody.parsePDB(out_path)

        datum = datum + [1, 'success']
        data = [datum]
        db.insert(table_sn, data)

    except Exception as e:
        datum = datum + [0, str(e)]
        data = [datum]
        db.insert(table_sn, data)

def overlap(table_sn, param, datum):

    try:
        receptor, chain, resnum, resname = datum
        docked_folder = param['docked_folder']
        crystal_folder = param['crystal_folder']
        clash_cutoff_A = param['clash_cutoff_A']
        clash_size_cutoff = param['clash_size_cutoff']


        lig_name = '_'.join([receptor, chain, resnum, resname, 'ligand']) + '.pdb'
        docked_dir = os.path.join(data_dir, docked_folder, receptor)
        docked_path = os.path.join(docked_dir, lig_name)

        crystal_dir = os.path.join(data_dir, crystal_folder, receptor)
        crystal_path = os.path.join(crystal_dir, lig_name)

        docked = prody.parsePDB(docked_path).getCoordsets()
        crystal = prody.parsePDB(crystal_path).getCoords()

        expanded_docked = np.expand_dims(docked, -2)
        diff = expanded_docked - crystal
        distance = np.sqrt(np.sum(np.power(diff, 2), axis=-1))
        all_clash = (distance < config.clash_cutoff_A).astype(float)
        atom_clash = (np.sum(all_clash, axis=-1) > 0).astype(float)
        position_clash = np.mean(atom_clash, axis=-1) > config.clash_size_cutoff
        position_clash_ratio = np.mean(atom_clash, axis=-1)

        data = []
        for i, ratio in enumerate(position_clash_ratio):
            data.append(datum+[i+1, ratio, 1, 'success'])

        db.insert(table_sn, data)

    except Exception as e:
        datum = datum + [1, 0, 0, str(e)]
        data = [datum]
        db.insert(table_sn, data)

def rmsd(table_sn, param, datum):
    
    try:
        receptor, chain, resnum, resname = datum
        docked_folder = param['docked_folder']
        crystal_folder = param['crystal_folder']
        lig_name = '_'.join([receptor, chain, resnum, resname, 'ligand']) + '.pdb'
        

        docked_dir = os.path.join(data_dir,docked_folder, receptor)
        docked_path = os.path.join(docked_dir, lig_name)

        crystal_dir = os.path.join(data_dir, crystal_folder, receptor)
        crystal_path = os.path.join(crystal_dir, lig_name)

        docked_coords = prody.parsePDB(docked_path).getCoordsets()
        crystal_coord = prody.parsePDB(crystal_path).getCoords()

        rmsd = np.sqrt(np.mean(np.sum(np.square(docked_coords - crystal_coord), axis=01), axis=-1))

        data = []
        for i, rd in enumerate(rmsd):
            data.append(datum+[i+1, rd, 1, 'success'])
        db.insert(table_sn, data)
    except Exception as e:
        datum = datum + [ 1, 0, 0, str(e)]
        data = [datum]
        db.insert(table_sn, data)

def native_contact(table_sn, param, datum):

    try:
        receptor, chain, resnum, resname = datum
        docked_folder = param['docked_folder']
        crystal_folder = param['crystal_folder']
        rec_folder = param['receptor_folder']
        distance_threshold = param['distance_threshold']
        lig_name = '_'.join([receptor, chain, resnum, resname, 'ligand']) + '.pdb'
        rec_name = '_'.join([receptor, chain, resnum, resname, 'receptor']) + '.pdb'

        docked_dir = os.path.join(data_dir, docked_folder, receptor)
        docked_path = os.path.join(docked_dir, lig_name)

        crystal_dir = os.path.join(data_dir, crystal_folder, receptor)
        crystal_path = os.path.join(crystal_dir, lig_name)

        rec_dir = os.path.join(data_dir, rec_folder, receptor)
        rec_path = os.path.join(rec_dir, rec_name)

        parsed_docked =  prody.parsePDB(docked_path).select('not hydrogen')
        parsed_crystal = prody.parsePDB(crystal_path).select('not hydrogen')
        parsed_rec = prody.parsePDB(rec_path).select('not hydrogen')

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

        data = []
        for i , nt in enumerate(contact_ratio):
            data.append(datum + [i+1, nt, 1, 'success'])

        db.insert(table_sn, data)
    except Exception as e:
        datum = datum + [ 0, 0, 0, str(e)]
        data = [datum]
        db.insert(table_sn, data)

def binding_affinity(table_sn, param, datum):
    try:
        pdb_bind_index = param['pdb_bind_index']
        pdb_bind_index = config.binding_affinity_files[pdb_bind_index]
        PDB_bind = read_PDB_bind(pdb_bind_index=pdb_bind_index)
        data = [[PDB_bind.pdb_names[i].upper(), PDB_bind.ligand_names[i],
                 PDB_bind.log_affinities[i], PDB_bind.normalized_affinities[i],
                 1, 'success']
                 for i in range(len(PDB_bind.pdb_names))]
        #print(data)
        db.insert(table_sn, data)
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
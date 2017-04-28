"""
Processing data
"""

import os
import sys
import re 
import time
import argparse 
import multiprocessing
import subprocess
from functools import partial
from glob import glob
from utils import log, smina_param, timeit, count_lines
import numpy as np 
import openbabel
import prody
import config
from config import lock 
from db import database
from config import data_dir
from _db import AffinityDatabase

db = AffinityDatabase()
FLAGS = None

def _makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def _table_folder(table_name):
    splited = table_name.split('_')
    table_type = '_'.join(splited[:-1])
    table_sn = splited[-1]
    return '_'.join([table_sn, table_type])

def download_pdb(param,receptor):

    try:    
        table_name = param['table_name']
        dest_dir = os.path.join(data_dir, _table_folder(table_name))
        _makedir(dest_dir)

        pdb_path = os.path.join(dest_dir, receptor+'.pdb')
        if not os.path.exists(pdb_path):
            download_address = 'https://files.rcsb.org/download/{}.pdb'.format(receptor)
            os.system('wget -P {} {}'.format(dest_dir, download_address))
            
        parsed = prody.parsePDB(pdb_path)
        header = prody.parsePDBHeader(pdb_path)
        

        datum = [receptor,header['experiment'], header['resolution'], 1, 'success']
        data = [datum]
        db.insert(table_name, data)
    except Exception as e:
        table_name=param['table_name']
        datum = [receptor,'unk',0, 0, str(e)]
        data = [datum]
        db.insert(table_name, data)


def split_pdb(param,receptor):
    try:
        if type(receptor).__name__ in ['tuple','list']:
            receptor = receptor[0]

        print receptor
        lig_table = param['lig_table']
        rec_table = param['rec_table']
        download_table = param['download_table']
        
        pdb_dir = os.path.join(data_dir,_table_folder(download_table))
        pdb_path = os.path.join(pdb_dir, receptor+'.pdb')
        
        parsed = prody.parsePDB(pdb_path)
        header = prody.parsePDBHeader(pdb_path)
        
        lig_dir = os.path.join(data_dir, _table_folder(lig_table), receptor)
        _makedir(lig_dir)
        print lig_dir
        rec_dir = os.path.join(data_dir, _table_folder(rec_table), receptor)
        _makedir(rec_dir)

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
                db.insert(lig_table, data)
            except Exception as e:
                data =  [receptor, chain, resnum, resname, 0, 0, str(e)]
                data = [data]
                db.insert(lig_table, data)
        
        for chain, resnum, resname in ligands:
            try:
                rec = parsed.select('not (chain {} resnum {})'.format(chain, resnum))
                rec = rec.select('not water')
                heavy_atom = rec.select('not hydrogen').numAtoms()
                rec_name = '_'.join([receptor, chain, resnum, resname, 'receptor']) + '.pdb'
                prody.writePDB(os.path.join(rec_dir, rec_name), rec)


                datum = [receptor, chain, resnum, resname, heavy_atom, 1, 'success']
                data = [datum]
                db.insert(rec_table, data)
            except Exception as e:
                datum = [receptor, chain, resnum, resname, 0, 0, str(e)]
                data = [datum]
                db.insert(rec_table, data) 

    except Exception as e:
        print e
        raise Exception(str(e))

def smina_dock(param, datum):
    try:
        receptor, chain, resnum, resname = datum
        table_name = param['table_name']
        lig_table = param['lig_table']
        rec_table = param['rec_table']
        smina_pm = param['smina_pm']

        out_dir = os.path.join(data_dir, _table_folder(table_name), receptor)
        _makedir(out_dir)
        out_name = '_'.join(datum +  ['ligand']) + '.pdb'
        out_path = os.path.join(out_dir, out_name)

        lig_dir = os.path.join(data_dir, _table_folder(lig_table) , receptor)
        lig_name = '_'.join(datum + ['ligand']) + '.pdb'
        lig_path = os.path.join(lig_dir, lig_name)

        rec_dir = os.path.join(data_dir, _table_folder(rec_table), receptor)
        rec_name = '_'.join(datum + ['receptor']) + '.pdb'
        rec_path = os.path.join(rec_dir, rec_name)

        kw = {
            'receptor': rec_path,
            'ligand': lig_path,
            'out':out_path
        }

        if not smina_pm.name == 'reorder':
            kw.update({'autobox_ligand': lig_path})

        cmd = smina_pm.make_command(**kw)
        print cmd
        cl = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        cl.wait()
        prody.parsePDB(out_path)

        datum = datum + [ 1, 'success']
        data = [datum]
        db.insert(table_name, data)

    except Exception as e:
        table_name = param['table_name']
        datum = datum + [ 0, str(e)]
        data = [datum]
        db.insert(table_name, data)

def overlap_with_ligand(param,datum):

    try:
        receptor, chain, resnum, resname =datum
        table_name = param['table_name']
        docked_table = param['docked_table']
        crystal_table = param['crystal_table']
        clash_cutoff_A = param['clash_cutoff_A']
        clash_size_cutoff = param['clash_size_cutoff']


        lig_name = '_'.join([receptor, chain, resnum, resname, 'ligand']) + '.pdb'
        docked_dir = os.path.join(data_dir, _table_folder(docked_table), receptor)
        docked_path = os.path.join(docked_dir, lig_name)

        crystal_dir = os.path.join(data_dir,_table_folder(crystal_table), receptor)
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
            data.append(datum+[i+1,ratio, 1, 'success'])

        db.insert(table_name, data)

    except Exception as e:
        table_name = param['table_name']
        datum = datum + [ 1, 0, 0, str(e)]
        data = [datum]
        db.insert(table_name, data)

def calculate_rmsd(param, datum):
    
    try:
        receptor, chain, resnum, resname = datum
        table_name = param['table_name']
        docked_table = param['docked_table']
        crystal_table = param['crystal_table']
        lig_name = '_'.join([receptor, chain, resnum, resname, 'ligand']) + '.pdb'
        

        docked_dir = os.path.join(data_dir,_table_folder(docked_table), receptor)
        docked_path = os.path.join(docked_dir, lig_name)

        crystal_dir = os.path.join(data_dir,_table_folder(crystal_table), receptor)
        crystal_path = os.path.join(crystal_dir, lig_name)

        docked_coords = prody.parsePDB(docked_path).getCoordsets()
        crystal_coord = prody.parsePDB(crystal_path).getCoords()

        rmsd = np.sqrt(np.mean(np.sum(np.square(docked_coords - crystal_coord), axis=01), axis=-1))

        data = []
        for i, rd in enumerate(rmsd):
            data.append(datum+[i+1, rd, 1, 'success'])
        db.insert(table_name, data)
    except Exception as e:
        table_name = param['table_name']
        datum = datum + [ 1, 0, 0, str(e)]
        data = [datum]
        db.insert(table_name, data)

def calculate_native_contact(param, datum):

    try:
        receptor, chain, resnum, resname = datum
        table_name = param['table_name']
        docked_table = param['docked_table']
        crystal_table = param['crystal_table']
        rec_table = param['rec_table']
        distance_threshold = param['distance_threshold']
        lig_name = '_'.join([receptor, chain, resnum, resname, 'ligand']) + '.pdb'
        rec_name = '_'.join([receptor, chain, resnum, resname, 'receptor']) + '.pdb'

        docked_dir = os.path.join(data_dir,_table_folder(docked_table), receptor)
        docked_path = os.path.join(docked_dir, lig_name)

        crystal_dir = os.path.join(data_dir,_table_folder(crystal_table), receptor)
        crystal_path = os.path.join(crystal_dir, lig_name)

        rec_dir = os.path.join(data_dir, _table_folder(rec_table),receptor)
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

        db.insert(table_name, data)
    except Exception as e:
        table_name = param['table_name']
        datum = datum + [ 0, 0, 0, str(e)]
        data = [datum]
        db.insert(table_name, data)

def run_multiprocess(target_list, func):
        
        target_list = map(list, target_list)
        print len(target_list)
        pool = multiprocessing.Pool(config.process_num)
        pool.map_async(func, target_list)
        pool.close()
        pool.join()


def main():
    if FLAGS.download:
        print "Downloading pdb fro rcsb..."
        download_list = open(config.list_of_PDBs_to_download).readline().strip().split(', ')
        table_param = {
            'datasource':'rcsb'
        }
        table_name = db.get_table('download',[],table_param)
        downloaded = db.get_all_success(table_name)
        job_param = {
            'table_name':table_name
        }
        run_multiprocess(set(download_list)-set(downloaded), partial(download_pdb, job_param))


    if FLAGS.split:
        print "Split ligand and receptor..."
        depend = FLAGS.depend
        if not len(depend) == 1:
            raise Exception('split option depend on download data')

        table_param = {
            'depend':depend
        }
        rec_table = db.get_table('splited_receptor',depend,table_param)
        lig_table = db.get_table('splited_ligand',depend,table_param)

        job_param = {
            'rec_table':rec_table,
            'lig_table':lig_table,
            'download_table':depend[0]
        }
        pdb_list = db.get_all_success(depend[0])

        run_multiprocess(pdb_list, partial(split_pdb, job_param))

    if FLAGS.reorder:
        print 'reorder ligand'
        depend = FLAGS.depend
        if not len(depend) == 2:
            raise Exception('reorder optiom depend on splited ligands and receptors')
        
        table_param = { }
        job_param = { }

        for tab in depend:
            if db.table_type(tab) == 'splited_ligand':
                table_param.update({'lig_table':tab})
            if db.table_type(tab) == 'splited_receptor':
                table_param.update({'rec_table':tab})
                 
        if not len(table_param.keys()) == 2:
            raise Exception('reorder optiom depend on splited ligands and receptors')

        job_param.update(table_param)
    
        smina_pm = smina_param('reorder')
        smina_pm.param_load(config.reorder_pm)

        job_param.update({'smina_pm':smina_pm})
        table_param.update({'smina_param':smina_pm.param_dump()})

        table_name = db.get_table('reorder_ligand',depend,table_param)
        job_param.update({'table_name':table_name})

        rec_list = db.get_all_success(table_param['rec_table'])
        lig_list = db.get_all_success(table_param['lig_table'])
        finish_list = db.get_all_success(table_name)
        
    
        reorder_list = list(set(rec_list) & set(lig_list) - set(finish_list))

        run_multiprocess(reorder_list,partial(smina_dock,job_param))

    if FLAGS.vinardo_dock:
        print 'docking by vinardo ligand'
        depend = FLAGS.depend
        if not len(depend) == 2:
            raise Exception('reorder optiom depend on reorder ligands and splited receptors')
        
        table_param = { }
        job_param = { }

        for tab in depend:
            if db.table_type(tab) == 'reorder_ligand':
                table_param.update({'lig_table':tab})
            if db.table_type(tab) == 'splited_receptor':
                table_param.update({'rec_table':tab})
                 
        if not len(table_param.keys()) == 2:
            raise Exception('dock optiom depend on reorder ligands and splited receptors')

        job_param.update(table_param)
    
        smina_pm = smina_param('vinardo')
        smina_pm.param_load(config.vinardo_pm)

        job_param.update({'smina_pm':smina_pm})
        table_param.update({'smina_param':smina_pm.param_dump()})

        table_name = db.get_table('docked_ligand',depend,table_param)
        job_param.update({'table_name':table_name})

        rec_list = db.get_all_success(table_param['rec_table'])
        lig_list = db.get_all_success(table_param['lig_table'])
        finish_list = db.get_all_success(table_name)
        
    
        reorder_list = list(set(rec_list) & set(lig_list) - set(finish_list))
        run_multiprocess(reorder_list,partial(smina_dock,job_param))
        
    if FLAGS.overlap:
        print 'calculate overlap'
        depend = FLAGS.depend
        if not len(depend) == 2:
            raise Exception('overlap calculate depend on docked ligands and reorder ligands')

        table_param = {}
        for tab in depend:
            if db.table_type(tab) == 'reorder_ligand':
                table_param.update({'crystal_table':tab})
            if db.table_type(tab) == 'docked_ligand':
                table_param.update({'docked_table':tab})
        if not len(table_param.keys()) == 2:
            raise Exception('overlap calculate depend on docked ligands and reorder ligands')
        
        table_param.update(config.overlap_default)
        table_name = db.get_table('overlap',depend,table_param)
    
        job_param = table_param 
        job_param.update({'table_name':table_name})

        cry_list = db.get_all_success(table_param['crystal_table'])
        lig_list = db.get_all_success(table_param['docked_table'])
        finish_list = db.get_all_success(table_name)

        rest_list = list(set(cry_list) & set(lig_list) - set(finish_list))
        run_multiprocess(rest_list,partial(overlap_with_ligand,job_param))

    if FLAGS.rmsd:
        print 'calculate rmsd'
        depend = FLAGS.depend
        if not len(depend) == 2:
            raise Exception('rmsd calculate depend on docked ligands and reorder ligands')

        table_param = {}
        for tab in depend:
            if db.table_type(tab) == 'reorder_ligand':
                table_param.update({'crystal_table':tab})
            if db.table_type(tab) == 'docked_ligand':
                table_param.update({'docked_table':tab})
        if not len(table_param.keys()) == 2:
            raise Exception('rmsd calculate depend on docked ligands and reorder ligands')
        
        table_name = db.get_table('rmsd',depend,table_param)
    
        job_param = table_param 
        job_param.update({'table_name':table_name})

        cry_list = db.get_all_success(table_param['crystal_table'])
        lig_list = db.get_all_success(table_param['docked_table'])
        finish_list = db.get_all_success(table_name)

        rest_list = list(set(cry_list) & set(lig_list) - set(finish_list))
        run_multiprocess(rest_list,partial(calculate_rmsd,job_param))

    if FLAGS.native_contact:
        print 'calculate native contact'
        depend = FLAGS.depend
        if not len(depend) == 3:
            raise Exception('native contact calculate depend on docked ligands reorder ligands and splited receptor')

        table_param = {}
        for tab in depend:
            if db.table_type(tab) == 'reorder_ligand':
                table_param.update({'crystal_table':tab})
            if db.table_type(tab) == 'docked_ligand':
                table_param.update({'docked_table':tab})         
            if db.table_type(tab) == 'splited_receptor':
                table_param.update({'rec_table':tab})

        if not len(table_param.keys()) == 3:
            raise Exception('native contact calculate depend on docked ligands reorder ligands and splited receptor')

        table_param.update(config.native_contace_default)
        table_name = db.get_table('native_contact',depend,table_param)

        job_param = table_param 
        job_param.update({'table_name':table_name})

        rec_list = db.get_all_success(table_param['rec_table'])
        cry_list = db.get_all_success(table_param['crystal_table'])
        lig_list = db.get_all_success(table_param['docked_table'])
        finish_list = db.get_all_success(table_name)

        rest_list = list(set(rec_list) & set(cry_list) & set(lig_list) - set(finish_list))
        run_multiprocess(rest_list, partial(calculate_native_contact, job_param))

    if FLAGS.test:
        print 'calculate native contact'
        depend = FLAGS.depend
        if type(depend).__name__ == 'list':
            depend = depend[0]
        if not db.table_type(depend) == 'docked_ligand':
            raise Exception("native contace calculate depend docked ligands")
        
        table_param = {
            'docked_table':depend,
            'crystal_table':db.get_depend_table(depend,'reorder_ligand'),
            'rec_table':db.get_depend_table(depend,'spleted_receptor')
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Database Create Option")
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--reorder', action='store_true')
    parser.add_argument('--vinardo_dock', action='store_true')
    parser.add_argument('--smina_dock', action='store_true')
    parser.add_argument('--overlap', action='store_true')
    parser.add_argument('--rmsd', action='store_true')
    parser.add_argument('--native_contact', action='store_true')
    parser.add_argument('--depend', nargs='*')
    parser.add_argument('--test', action='store_true')

    FLAGS, unparsed = parser.parse_known_args()
    main()
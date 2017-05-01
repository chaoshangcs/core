import os
import sys 
import re 
import numbers 
from collections import namedtuple
import pandas as pd 
from database_action import db 
import six 

import tensorflow as tf
import numpy as np
import sys
import os
import re
import prody
import config
import pandas as pd 
sys.path.append('..')
from av4_atomdict import atom_dictionary

def _receptor(path):
    return os.path.basename(path).split('_')[0]

def atom_to_number(atomname):
    atomic_tag_number = atom_dictionary.ATM[atomname.lower()]
    return atomic_tag_number

def save_av4(filepath,labels,elements,multiframe_coords):
    labels = np.asarray(labels,dtype=np.int32)
    elements = np.asarray(elements,dtype=np.int32)
    multiframe_coords = np.asarray(multiframe_coords,dtype=np.float32)

    if not (int(len(multiframe_coords[:,0]) == int(len(elements)))):
        raise Exception('Number of atom elements is not equal to the number of coordinates')

    if multiframe_coords.ndim==2:
        if not int(len(labels))==1:
            raise Exception ('Number labels is not equal to the number of coordinate frames')
    else:
        if not (int(len(multiframe_coords[0, 0, :]) == int(len(labels)))):
            raise Exception('Number labels is not equal to the number of coordinate frames')

    number_of_examples = np.array([len(labels)], dtype=np.int32)
    av4_record = number_of_examples.tobytes()
    av4_record += labels.tobytes()
    av4_record += elements.tobytes()
    av4_record += multiframe_coords.tobytes()
    f = open(filepath, 'w')
    f.write(av4_record)
    f.close()

def convert_data_to_av4(base_dir, rec_path, lig_path):

    dest_dir = os.path.join(base_dir, _receptor(rec_path))
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    prody_receptor = prody.parsePDB(rec_path)
    prody_ligand = prody.parsePDB(lig_path)

    multiframe_ligand_coords = prody_ligand.getCoords()
    multiframe_ligand_coords = np.expand_dims(multiframe_ligand_coords,-1)
    labels = np.array([1])
    try:
        receptor_elements = map(atom_to_number,prody_receptor.getElements())
        ligand_elements = map(atom_to_number,prody_ligand.getElements())
    except:
        return None, None
    
    rec_name = os.path.basename(rec_path).replace('.pdb','.av4')
    lig_name = os.path.basename(lig_path).replace('.pdb','.av4')
    
    
    av4_rec_path = os.path.join(dest_dir,rec_name)
    av4_lig_path = os.path.join(dest_dir,lig_name)
    save_av4(av4_rec_path,[0], receptor_elements, prody_receptor.getCoords())
    save_av4(av4_lig_path, labels , ligand_elements, multiframe_ligand_coords)
    return av4_rec_path, av4_lig_path

class table(pd.DataFrame):
    def apply_rest(self, key, val):
        if isinstance(val, numbers.Number) or isinstance(val, six.string_types):
            self = self[self[key] == val]
        elif isinstance(val, list):
            if len(val) == 2:
                minimum, maximum = val
                if minimum is not None:
                    self = self[self[key] >= minimum]
                if maximum is not None:
                    self = self[self[key] <= maximum]
            else:
                raise Exception("require restriction size 2, get %d" % len(val))
        elif isinstance(val, tuple):
            if len(val) == 2:
                minimum, maximum = val
                if minimum is not None:
                    self = self[self[key] > minimum]
                if maximum is not None:
                    
                    self = self[self[key] < maximum]
            else:
                raise Exception("require restriction size 2, get %d" % len(val))
        elif val is None:
            pass
        else:
            raise Exception("Restrictions type {} doesn't support.".format(type(val).__name__))
        
        return table(self)

    def __and__(self, other):
        self = self.merge(other).drop_duplicates().dropna()
        return table(self)

    def __or__(self, other):
        self = self.merge(other, how='outer').drop_duplicates().dropna()
    
    def __sub__(self, other):
        
        s = set(map(tuple, list(slef.values)))
        o = set(map(tuple, list(other.values)))

        diff = s - o
        columns = self.columns

        if len(diff):
            self = table(list(diff), columns=columns)
        else:
            self = table()

        return self

class retrive_av4:

    def __init__(self, folder_name):
        
        self.folder_name = folder_name
        self.affinity = None

    def receptor(self, receptor_sn):
        _, _, rec = db.get_success_data(receptor_sn, dataframe=True)
        primary_key = db.primary_key_for(receptor_sn)
        rec = rec[primary_key]
        self.rec = table(rec)
        rec_folder = db.get_folder(receptor_sn)
        self.rec_folder = '{}_{}'.format(receptor_sn, rec_folder)


    def crystal(self, crystal_sn):
        _, _, cry = db.get_success_data(crystal_sn, dataframe=True)
        primary_key = db.primary_key_for(crystal_sn)
        cry = cry[primary_key]
        self.cry = table(cry)
        cry_folder = db.get_folder(crystal_sn)
        self.cry_folder = '{}_{}'.format(crystal_sn, cry_folder)

    def norm_affinity(self, affinity_sn, rest):
        
        _, _, affinity = db.get_success_data(affinity_sn, dataframe=True)
        affinity = table(affinity).apply_rest('norm_affinity',rest)
        primary_key = db.primary_key_for(affinity_sn)
        affinity = affinity[primary_key+['norm_affinity']]
        if self.affinity is None:
            self.affinity = table(affinity)
        else:
            self.affinity = self.affinity and table(affinity)

    def log_affinity(self, affinity_sn, rest):
        
        _, _, affinity = db.get_success_data(affinity_sn, dataframe=True)
        affinity = table(affinity).apply_rest('log_affinity',rest)
        primary_key = db.primary_key_for(affinity_sn)
        affinity = affinity[primary_key+['log_affinity']]
        if self.affinity is None:
            self.affinity = table(affinity)
        else:
            self.affinity = self.affinity and table(affinity)

    def get_receptor_and_ligand(self):
        
        valid = self.affinity & self.rec & self.cry 
        collection = []
        for i in range(len(valid)):
            item = valid.ix[i]
            rec = item['receptor']
            file = '{}_{}_{}_{}'.format(*item[['receptor', 'chain', 'resnum', 'resname']])
            receptor_path = os.path.join(config.data_dir, 
                                    self.rec_folder, 
                                    item['receptor'],
                                    file+'_receptor.pdb')

            ligand_path = os.path.join(config.data_dir,
                                       self.cry_folder ,
                                       item['receptor'],
                                       file+'_ligand.pdb')
            if 'log_affinity' in valid.columns:
                affinity = item['log_affinity']
            else:
                affinity = item['norm_affinity']

            
            collection.append([receptor_path, ligand_path, affinity])

        #print(set(map(lambda x:len(x),collection)))
        
        if not os.path.exists(config.table_dir):
            os.makedirs(config.table_dir)
        df = pd.DataFrame(collection,columns=['receptor','ligand','affinity'])
        
        df.to_csv(os.path.join(config.table_dir,'raw.csv'), index=False)

        export_dir = os.path.join(config.database_root, self.folder_name)

        index = []
        for rec, lig, aff in collection:
            rec_path, lig_path = convert_data_to_av4(export_dir, rec, lig)
            if rec_path is None:
                continue
            index.append([rec_path, lig_path, aff])

        df = pd.DataFrame(index, columns=['receptor','ligand','affinity'])
        df.to_csv(os.path.join(config.table_dir,'index.csv'), index=False)



def test():
    
    ra = retrive_av4('av4') # output's filder name
    ra.receptor(2) # splited receptor table sn
    ra.crystal(3) # splited ligand table sn
    ra.log_affinity(4, [None,-27]) # affinity table sn , [minimum, maximum]

    ra.get_receptor_and_ligand() # convert file into av4 format
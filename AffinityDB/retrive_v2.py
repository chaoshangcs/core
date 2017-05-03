import os
import sys 
import re 
import numbers 
from collections import namedtuple
import pandas as pd 
from database_action import db 
import six 
import time
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
    labels = np.asarray(labels,dtype=np.float32)
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

def convert_data_to_av4(base_dir, rec_path, lig_path, doc_path, position, affinity):

    dest_dir = os.path.join(base_dir, _receptor(rec_path))
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    prody_receptor = prody.parsePDB(rec_path)
    prody_ligand = prody.parsePDB(lig_path)

    receptor_elem = prody_receptor.getElements()
    ligand_elem =prody_ligand.getElements()

    ligand_coords = prody_ligand.getCoords()
    labels = np.array([affinity], dtype=np.float32)

    if len(position):
        # docked list not empty
        prody_docked = prody.parsePDB(doc_path)
        docked_elem = prody_docked.getElemens()

        assert all(np.asarray(docked_elem) == np.asarray(ligand_elem))

        docked_coords = prody_docked.getCoordsets()[position]
        for docked_coord in docked_coords:
            ligand_coords = np.dstack((ligand_coords, docked_coord))
            labels = np.concatenate((labels, [1.]))
    else:
        ligand_coords = np.expand_dims(ligand_coords,-1)
        


    try:
        receptor_elements = map(atom_to_number,receptor_elem)
        ligand_elements = map(atom_to_number,ligand_elem)
        
    except:
        return None, None
    
    rec_name = os.path.basename(rec_path).replace('.pdb','.av4')
    lig_name = os.path.basename(lig_path).replace('.pdb','.av4')
    
    
    av4_rec_path = os.path.join(dest_dir,rec_name)
    av4_lig_path = os.path.join(dest_dir,lig_name)
    save_av4(av4_rec_path,[0], receptor_elements, prody_receptor.getCoords())
    save_av4(av4_lig_path, labels , ligand_elements, ligand_coords)
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
        self.export_dir = os.path.join(config.export_dir, folder_name)
        while os.path.exists(self.export_dir):
            timestr = time.strftime('%m_%d_%H', time.gmtime())
        
            self.export_dir = self.export_dir + '_' + timestr

        self.ligand = None
        self.position = None
        self.receptor_folder = None 
        self.ligand_folder = None 
        self.docked_folder = None 
        self.affinity_key = None

    def receptor(self, receptor_idx):
        _, _, df = db.get_success_data(receptor_idx, dataframe=True)
        primary_key = db.primary_key_for(receptor_idx)
        df = df[primary_key]
        df = table(df) 
        if self.ligand is None:
            self.ligand  = df 
        else:
            self.ligand = self.ligand and df

        folder_name = db.get_folder(receptor_idx)
        self.receptor_folder = '{}_{}'.format(receptor_idx, folder_name)


    def crystal(self, crystal_idx):
        _, _, df  = db.get_success_data(crystal_idx, dataframe=True)
        primary_key = db.primary_key_for(crystal_idx)
        df = df[primary_key]
        df = table(df) 

        if self.ligand is None:
            self.ligand = df 
        else:
            self.ligand = self.ligand & df 
        
        folder_name  = db.get_folder(crystal_idx)
        self.ligand_folder = '{}_{}'.format(crystal_idx, folder_name)

    def docked(self, docked_idx):
        _, _, df = db.get_success_data(docked_idx, dataframe=True)
        primary_key = db.primary_key_for(docked_idx)
        df = df[primary_key]
        df = table(df)

        if self.ligand is None:
            self.ligand = df 
        else:
            self.ligand = self.ligand & df 
        
        folder_name = db.get_folder(docked_idx)
        self.docked_folder = '{}_{}'.format(docked_idx, folder_name )

    def overlap(self, overlap_idx, rest):
        _, _, df = db.get_success_data(overlap_idx, dataframe=True)
        primary_key = db.primary_key_for(overlap_idx)
        df = df[primary_key]
        df = table(df).apply_rest('overlap',rest)

        if self.position is None:
            self.position = df 
        else:
            self.position = self.docked & df 



    def rmsd(self, rmsd_idx, rest):
        
        _, _, df = db.get_success_data(crystal_idx, dataframe=True)
        primary_key = db.primary_key_for(rmsd_idx)
        df= df[primary_key + ['rmsd']]
        df = table(df).apply_rest('rmsd',rest)
        
        if self.position is None:
            self.position = df
        else:
            self.position = self.position & df
        
    def native_contact(self, native_contact_idx, rest):
        
        _, _, df = db.get_success_data(native_contact_idx, dataframe=True)
        primary_key = db.primary_key_for(native_contact_idx)
        df = df[primary_key + ['native_contact']]
        df = table(df).apply_rest('native_contact',rest)

        if self.position is None:
            self.position = df 
        else: 
            self.position = self.position & df 

    def norm_affinity(self, affinity_idx, rest):
        
        self.affinity_key = 'norm_affinity'
        _, _, df = db.get_success_data(affinity_idx, dataframe=True)
        primary_key = db.primary_key_for(affinity_idx)
        df = df[primary_key+[self.affinity_key]]
        df = table(df).apply_rest(self.affinity_key,rest)
        
        if self.ligand is None:
            self.ligand = df
        else:
            self.ligand = self.ligand & df 

    def log_affinity(self, affinity_idx, rest):
        
        self.affinity_key = 'log_affinity'
        _, _, df = db.get_success_data(affinity_idx, dataframe=True)
        primary_key = db.primary_key_for(affinity_idx)
        df = df[primary_key+[self.affinity_key]]
        df = table(df).apply_rest(self.affinity_key,rest)

        if self.ligand is None:
            self.ligand = df
        else:
            self.ligand = self.ligand & df 



    def get_receptor_and_ligand(self):
        
        if self.position is None:
            valid = self.ligand 
        
            
            collection = []
            for i in range(len(valid)):
                item = valid.ix[i]
                receptor = item['receptor']

                file = '_'.join(item[['receptor', 'chain', 'resnum', 'resname']])
                
                receptor_path = os.path.join(config.data_dir, 
                                        self.receptor_folder, 
                                        receptor,
                                        file+'_receptor.pdb')

                ligand_path = os.path.join(config.data_dir,
                                        self.ligand_folder,
                                        receptor,
                                        file+'_ligand.pdb')

                docked_path = ''
                positions = []

                affinity = item[self.affinity_key]

                collection.append([receptor_path, 
                                ligand_path, 
                                docked_path, 
                                positions, 
                                affinity])
            
        
        else:
            valid = self.ligand and self.position  

            
            collection =[]
            
            for keys, group in valid.groupby(['receptor','chain','resnum','resname', aff_key]):
                receptor = keys[0]
                file = '_'.join(keys[:4])

                receptor_path = os.path.join(config.data_dir, 
                                        self.receptor_folder, 
                                        receptor,
                                        file+'_receptor.pdb')

                ligand_path = os.path.join(config.data_dir,
                                        self.ligand_folder,
                                        receptor,
                                        file+'_ligand.pdb')

                docked_path = os.path.join(config.data_dir,
                                        self.docked_folder,
                                        receptor,
                                        file+'_ligand.pdb')

                positions = sorted(group['position'])

                affinity = list(set(group[self.affinity_key]))
                assert len(affinity) == 1
                affinity = affinity[0]

                collection.append([receptor_path, 
                                   ligand_path, 
                                   docked_path, 
                                   positions, 
                                   affinity])

        #print(set(map(lambda x:len(x),collection)))
        export_table_dir = os.path.join(self.export_dir,'index')
        if not os.path.exists(export_table_dir):
            os.makedirs(export_table_dir)

        df = pd.DataFrame(collection,columns=['receptor','ligand','docked','position','affinity'])
        df.to_csv(os.path.join(export_table_dir,'raw.csv'), index=False, sep='\t')

        data_export_dir = os.path.join(self.export_dir,'data')

        index = []
        for receptor, ligand, docked, position, aff in collection:
            rec_path, lig_path = convert_data_to_av4(data_export_dir,
                                                     receptor,
                                                     ligand,
                                                     docked,
                                                     position,
                                                     aff)
            if rec_path is None:
                continue
            index.append([rec_path, lig_path, affinity])

        df = pd.DataFrame(index, columns=['receptor','ligand','affinity'])
        df.to_csv(os.path.join(export_table_dir,'index.csv'), index=False)



def test():
    
    ra = retrive_av4('av4') # output's filder name
    ra.receptor(2) # splited receptor table sn
    ra.crystal(3) # splited ligand table sn
    ra.log_affinity(4, None) # affinity table idx , [minimum, maximum]
    ra.get_receptor_and_ligand() # convert file into av4 format

if __name__ == '__main__':
    test()
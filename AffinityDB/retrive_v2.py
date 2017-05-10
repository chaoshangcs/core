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
import cPickle
import tensorflow as tf 
sys.path.append('..')
from av4_atomdict import atom_dictionary


def _receptor(path):
    return os.path.basename(path).split('_')[0]

def atom_to_number(atomname):
    atomic_tag_number = atom_dictionary.ATM[atomname.lower()]
    return atomic_tag_number

def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _byte_feature(value):
    return tf.train.Feature(byte_list=tf.train.BytesList(value=value))

def save_with_format(filepath,labels,elements,multiframe_coords,d_format='tfr'):
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

    if d_format == 'av4':
        av4_record = number_of_examples.tobytes()
        av4_record += labels.tobytes()
        av4_record += elements.tobytes()
        av4_record += multiframe_coords.tobytes()
        f = open(filepath, 'w')
        f.write(av4_record)
        f.close()

    
    elif d_format == 'pkl':
        dump_cont = [number_of_examples,labels, elements, multiframe_coords]
        cPickle.dump(dump_cont,open(filepath,'w'))

    elif d_format == 'tfr':
        writer = tf.python_io.TFRecordWriter(filepath)

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'number_of_examples': _int_feature(number_of_examples),
                    'labels': _float_feature(labels),
                    'elements': _int_feature(elements),
                    'multiframe_coords': _float_feature(multiframe_coords.reshape(-1))
                }
            )
        )
        serialized = example.SerializeToString()
        writer.write(serialized)


def save_tfr_one(save_path, rec_labels, rec_elements, rec_coords, lig_labels, lig_elements, lig_coords):
    rec_labels = np.asarray(rec_labels, dtype=np.float32)
    rec_elements = np.asarray(rec_elements, dtype=np.int32)
    rec_coords = np.asarray(rec_coords, dtype=np.float32)

    lig_labels = np.asarray(lig_labels, dtype=np.float32)
    lig_elements = np.asarray(lig_elements, dtype=np.int32)
    lig_coords = np.asarray(lig_coords, dtype=np.float32)

    if not (int(len(rec_coords[:,0]) == int(len(rec_elements)))):
        raise Exception('Receptor: Number of atom elements is not equal to the number of coordinates')

    if rec_coords.ndim==2:
        if not int(len(rec_labels))==1:
            raise Exception ('Receptor: Number labels is not equal to the number of coordinate frames')
            
    else:
        if not (int(len(rec_coords[0, 0, :]) == int(len(rec_labels)))):
            raise Exception('Rcecpeor: Number labels is not equal to the number of coordinate frames')


    if not (int(len(lig_coords[:,0]) == int(len(lig_elements)))):
        raise Exception('Ligand: Number of atom elements is not equal to the number of coordinates')

    if lig_coords.ndim==2:
        if not int(len(lig_labels))==1:
            raise Exception ('Ligand: Number labels is not equal to the number of coordinate frames')
            
    else:
        if not (int(len(lig_coords[0, 0, :]) == int(len(lig_labels)))):
            raise Exception('Ligand: Number labels is not equal to the number of coordinate frames')

    number_of_examples = np.array([len(lig_labels)], dtype=np.int32)

    writer = tf.python_io.TFRecordWriter(save_path)

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'number_of_examples': _int_feature(number_of_examples),
                'ligand_labels': _float_feature(lig_labels),
                'ligand_elements': _int_feature(lig_elements),
                'ligand_coords': _float_feature(lig_coords.reshape(-1)),
                'receptor_labels':_float_feature(rec_labels),
                'receptor_elements':_int_feature(rec_elements),
                'receptor_coords':_float_feature(rec_coords.reshape(-1))
                }
            )
        )
    serialized = example.SerializeToString()
    writer.write(serialized)


def convert_and_save_data(base_dir, rec_path, lig_path, doc_path, position, affinity, d_format):

   
    dest_dir = os.path.join(base_dir,d_format, _receptor(rec_path))
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

    if d_format == 'tfr_one':
        save_name = os.path.basename(rec_path).replace('_receptor.pdb','.tfr')
        save_path = os.path.join(dest_dir, save_name)
        save_tfr_one(save_path, 
                     [0], 
                     receptor_elements, 
                     prody_receptor.getCoords(), 
                     labels,
                     ligand_elements, 
                     ligand_coords)
        return save_path, None
    
    rec_name = os.path.basename(rec_path).replace('.pdb','.%s' % d_format)
    lig_name = os.path.basename(lig_path).replace('.pdb','.%s' % d_format)
    
    
    rec_path = os.path.join(dest_dir,rec_name)
    lig_path = os.path.join(dest_dir,lig_name)
    save_with_format(rec_path,[0], receptor_elements, prody_receptor.getCoords(), d_format)
    save_with_format(lig_path, labels , ligand_elements, ligand_coords, d_format)
    return rec_path, lig_path

class table(pd.DataFrame):
    def apply_rest(self, key, val):
        new = self
        if isinstance(val, numbers.Number) or isinstance(val, six.string_types):
            new = new[new[key] == val]
        elif isinstance(val, list):
            if len(val) == 2:
                minimum, maximum = val
                if minimum is not None:
                    new = new[new[key] >= minimum]
                if maximum is not None:
                    new = new[new[key] <= maximum]
            else:
                raise Exception("require restriction size 2, get %d" % len(val))
        elif isinstance(val, tuple):
            if len(val) == 2:
                minimum, maximum = val
                if minimum is not None:
                    new = new[new[key] > minimum]
                if maximum is not None:
                    
                    new = new[new[key] < maximum]
            else:
                raise Exception("require restriction size 2, get %d" % len(val))
        elif val is None:
            pass
        else:
            raise Exception("Restrictions type {} doesn't support.".format(type(val).__name__))
        
        return self.wrap(new)

    @classmethod
    def wrap(cls, dataframe):
        return cls(dataframe)

    def __and__(self, other):
        new = self
        new = new.merge(other).drop_duplicates().dropna()
        return self.wrap(new)

    def __or__(self, other):
        new = self
        new = new.merge(other, how='outer').drop_duplicates().dropna()
        return self.wrap(new)

    def __sub__(self, other):
        
        new = self
        s = set(map(tuple, list(new.values)))
        o = set(map(tuple, list(other.values)))

        diff = s - ofilepathfilepath
        columns = self.columns

        if len(diff):
            new = self.wrap(table(list(diff), columns=columns))
        else:
            new = self.wrap(pd.DataFrame())

        return new

class retrive_data(object):

    def __init__(self):
        
        # table for available ligand
        # columns : ['receptor','chain','resnum', 'resname'] + other
        # path of receptor : [receptor_folder]/[receptor]/[receptor]_[chain]_[resnum]_[resname]_receptor.pdb
        # path of ligand   : [ligand_folder]/[receptor]/[receptor]_[chain]_[resnnum]_[resname]_ligand.pdb
        # path of docked ligand : [docked_folder]/[receptor]/[receptor]_[chain]_[resnum]_[resname]_ligand.pdb
        self.ligand = None
        
        # table for available position 
        # columns : ['receptor','chain','resnum','resname','position'] + other
        self.position = None
        
        # where can get splited receptor
        # e.g. 2_splited_receptor
        self.receptor_folder = None 
        
        # where can get splited ligand
        # e.g. 3_splited_ligand
        self.ligand_folder = None 
        
        # where can get the docked ligand
        # e.g. 4_docked
        self.docked_folder = None 
        
        # log_affinity or norm_affinity
        # it will be the label for the ligand
        self.affinity_key = None

    def __and__(self, other):

        assert self.receptor_folder == other.receptor_folder
        assert self.ligand_folder == other.ligand_folder 
        assert self.docked_folder == other.docked_folder \
                or self.docked_folder == None \
                or other.docked_folder == None

        assert self.affinity_key == other.affinity_key

        new = self.same()

        if new.ligand is None:
            new.ligand = other.ligand 
        elif other.ligand is not None:
            new.ligand = new.ligand & other.ligand 
        
        if new.position is None:
            new.position = other.position
        elif other.position is not None:
            new.position = new.position & other.position 
        
        if new.docked_folder is None:
            new.docked_folder = other.docked_folder

        return new 

    def __or__(self, other):
        
        assert self.receptor_folder == other.receptor_folder
        assert self.ligand_folder == other.ligand_folder
        assert self.docked_folder == other.docked_folder \
                or self.docked_folder == None \
                or other.docked_folder == None

        assert self.affinity_key == other.affinity_key

        new = self.same()

        if new.ligand is None:
            new.ligand = other.ligand 
        elif other.ligand is not None:
            new.ligand = new.ligand | other.ligand 

        if new.position is None:
            new.position = other.position
        elif other.position is not None:
            new.position = new.position | other.position 
        
        if new.docked_folder is None:
            new.docked_folder = other.docked_folder

        return new
      

    def same(self):
        
        new = retrive_data()
        new.receptor_folder = self.receptor_folder
        new.ligand_folder = self.ligand_folder
        new.docked_folder = self.docked_folder
        new.ligand = self.ligand
        new.position = self.position
        new.affinity_key = self.affinity_key

        return new


    def receptor(self, receptor_idx):
        # load available receptor from table with idx: receptor_idx

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
        
        return self


    def crystal(self, crystal_idx):
        # load available ligand from table with idx: crystal_idx

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

        return self

    def docked(self, docked_idx):
        # load available docked ligand from table with idx: docked_idx

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

        return self

    def overlap(self, overlap_idx, rest):
        # select position with overlap value in restriction: rest
        # e.g. rest=[0.1,0.5] overlap ratio between 0.1 and 0.5

        _, _, df = db.get_success_data(overlap_idx, dataframe=True)
        primary_key = db.primary_key_for(overlap_idx)
        df = df[primary_key]
        df = table(df).apply_rest('overlap',rest)

        if self.position is None:
            self.position = df 
        else:
            self.position = self.docked & df 

        return self

    def rmsd(self, rmsd_idx, rest):
        # select position with rmsd value in restriction: rest
        # e.g. rest=[None, 2] rmsd ration between minimum and 2
        
        _, _, df = db.get_success_data(crystal_idx, dataframe=True)
        primary_key = db.primary_key_for(rmsd_idx)
        df= df[primary_key + ['rmsd']]
        df = table(df).apply_rest('rmsd',rest)
        
        if self.position is None:
            self.position = df
        else:
            self.position = self.position & df

        return self

    def native_contact(self, native_contact_idx, rest):
        # select position with native contact ration in restriction: rest
        # e.g. rest=None  no restriction on native contact


        _, _, df = db.get_success_data(native_contact_idx, dataframe=True)
        primary_key = db.primary_key_for(native_contact_idx)
        df = df[primary_key + ['native_contact']]
        df = table(df).apply_rest('native_contact',rest)

        if self.position is None:
            self.position = df 
        else: 
            self.position = self.position & df 

        return self

    def norm_affinity(self, affinity_idx, rest):
        # select available ligand with norm affinity value in restriction: rest
        
        self.affinity_key = 'norm_affinity'
        _, _, df = db.get_success_data(affinity_idx, dataframe=True)
        primary_key = db.primary_key_for(affinity_idx)
        df = df[primary_key+[self.affinity_key]]
        df = table(df).apply_rest(self.affinity_key,rest)
        
        if self.ligand is None:
            self.ligand = df
        else:
            self.ligand = self.ligand & df 

        return self

    def log_affinity(self, affinity_idx, rest):
        # select available receptor with log affinity value in restriction: rest

        self.affinity_key = 'log_affinity'
        _, _, df = db.get_success_data(affinity_idx, dataframe=True)
        primary_key = db.primary_key_for(affinity_idx)
        df = df[primary_key+[self.affinity_key]]
        df = table(df).apply_rest(self.affinity_key,rest)

        if self.ligand is None:
            self.ligand = df
        else:
            self.ligand = self.ligand & df 

        return self

    def export_table(self):
        if self.position is None:
            return self.ligand
        else:
            return self.ligand and self.position

    def export_data_to(self, folder_name, d_format):
        # export data to the folder : folder_name
        # export data format
        #   'pkl': python pikle
        #   'av4': affinity build-in binary format
        #   'tfr': tensorflow record format
        
        if not d_format in ['pkl','av4','tfr', 'tfr_one']:
            raise Exception("Unexpected format {}, available format: {}".\
                                format(d_format, ['pkl','av4','tfr']))
        
        export_dir = os.path.join(config.export_dir, folder_name)
        
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
        export_table_dir = os.path.join(export_dir,'index')
        if not os.path.exists(export_table_dir):
            os.makedirs(export_table_dir)

        df = pd.DataFrame(collection,columns=['receptor','ligand','docked','position','affinity'])
        df.to_csv(os.path.join(export_table_dir,'raw.csv'), index=False, sep='\t')

        data_export_dir = os.path.join(export_dir,'data')

        index = []
        for receptor, ligand, docked, position, aff in collection:
            rec_path, lig_path = convert_and_save_data(data_export_dir,
                                                     receptor,
                                                     ligand,
                                                     docked,
                                                     position,
                                                     aff,
                                                     d_format)
            if rec_path is None:
                continue
            index.append([rec_path, lig_path, affinity])

        df = pd.DataFrame(index, columns=['receptor','ligand','affinity'])
        df.to_csv(os.path.join(export_table_dir,'index.csv'), index=False)




def example1():
    
    ra = retrive_data() # output's filder name
    ra.receptor(2) # splited receptor table sn
    ra.crystal(3) # splited ligand table sn
    ra.log_affinity(4, None) # affinity table idx , [minimum, maximum]
    
    ra.export_data_to('test_tfr_one','tfr') # convert file into av4 format
    
    #table = ra.export_table()

def example2():

    rb = retrive_data().recpeotr(2).crystal(3).norm_affinity(4,None)
    rc = rb.same().overlap(5,[None,0.5])
    rd = rb.same().overlap(5,(0.5,None)).rmsd(6,[None,2])
    re = rc | rd 
    table = re.export_table()


if __name__ == '__main__':
    print 'retrive_v2'
    example1()
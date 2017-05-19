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
    '''
    save the labels, elemetns and coords into as d_format file
    Args:
        filepath: path for the output file
        labels:  [0] for receptor [affinity] for crystal ligand
        elements: int64::list
        multiframe_coords:  
        d_format: output format ['tfr','pkl','av4']

    Returns: None

    '''
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
    '''
    record receptor and ligand into one TFRecords
    
    Args:
        save_path: path for output file
        rec_labels: always [0] for receptor
        rec_elements: int64::list
        rec_coords:  numpy.array 
        lig_labels:  numpy.array
        lig_elements: numpy.array
        lig_coords: numpy.array

    Returns: None

    '''

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
    """
    convert the receptor and ligand into given format
    
    
    
    in the database position is start with 1
    but when select coordinates be care that python list start with 0
    so we need np.asarray(position).sort()-1 to make it work
    Args:
        base_dir: directory for output file
        rec_path: path for receptor
        lig_path: path for ligand
        doc_path: path for docked ligand
        position: list of position for docked liangd, start with 1
        affinity: affinity value ( log affinity or norm faffinity
        d_format: output format [ pdb, pkl, av4, tfr, tfr_one ]

    Returns: (save_rec_path, save_lig_path) path of the output file
             (None, None) when failed to parse PDB file
             (save_path, None) when output format is tfr_one

    """
   
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
        docked_elem = prody_docked.getElements()

        assert all(np.asarray(docked_elem) == np.asarray(ligand_elem))

        docked_coords = prody_docked.getCoordsets()[np.asarray(position).sort() -1]
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

    if d_format == 'pdb':
        save_rec_name= os.path.basename(rec_path)
        save_rec_path = os.path.join(dest_dir, save_rec_name)
        #os.system('cp {} {}'.format(rec_path, save_rec_path))

        save_lig_name = os.path.basename(lig_path)
        save_lig_path = os.path.join(dest_dir, save_lig_name)
        if len(position):
            try:
                lig = prody.parsePDB(lig_path)
                doc = prody.parsePDB(doc_path)
                lig.addCoordset(doc.getCoordsets()[np.asarray(position).sort()-1])
            except Exception as e:
                print e
                exit(1)

            prody.writePDB(save_lig_path, lig)
        else:
            os.system('cp {} {}'.format(lig_path, save_lig_path))
        os.system('cp {} {}'.format(rec_path, save_rec_path))

        return save_rec_path, save_lig_path



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
        if len(self) == 0:
            return self
        elif len(other) == 0:
            return other
        else:
            new = self
            new = new.merge(other).drop_duplicates().dropna()
            return self.wrap(new)

    def __or__(self, other):
        if len(self) == 0:
            return other
        elif len(other) == 0:
            return self
        else:
            new = self
            new = new.merge(other, how='outer').drop_duplicates().dropna()
            return self.wrap(new)

    def __sub__(self, other):

        new = self
        intersec = new & other
        #union = new | other

        i = set(map(tuple, list(intersec.values)))
        u = set(map(tuple, list(new.values)))

        diff = u - i
        columns = self.columns

        if len(diff):
            new = self.wrap(pd.DataFrame(list(diff), columns=columns))
        else:
            new = self.wrap(pd.DataFrame())

        return new

class retrive_data(object):
    export_fmt = ['pdb','pkl','av4','tfr', 'tfr_one']

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

        # table for ligand that is invalid
        # columns = ['resname']
        self.exclude = table(pd.DataFrame())

    def __and__(self, other):
        '''
        and operation between two retive_data obj
        new = self & other
        
        Args:
            other: retrive_data obj

        Returns: retrive_data 

        '''
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

        if new.exclude is None:
            new.exclude = other.exclude
        elif other.exclude is not None:
            new.exclude = new.exclude | other.exclude

        if new.docked_folder is None:
            new.docked_folder = other.docked_folder

        return new 

    def __or__(self, other):
        '''
        or operation between data_retrive obj
        new = self | other
        
        Args:
            other: data_retrive obj

        Returns: data_retrive obj

        '''
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

        if new.exclude is None:
            new.exclude = other.exclude
        elif other.exclude is not None:
            new.exclude = new.exclude | other.exclude
        
        if new.docked_folder is None:
            new.docked_folder = other.docked_folder

        return new
      

    def same(self):
        '''
        Greate and return a new object with same attribute
        
        Returns: retrive_data object

        '''
        new = retrive_data()
        new.receptor_folder = self.receptor_folder
        new.ligand_folder = self.ligand_folder
        new.docked_folder = self.docked_folder
        new.ligand = self.ligand
        new.position = self.position
        new.affinity_key = self.affinity_key
        new.exclude = self.exclude

        return new


    def receptor(self, receptor_idx):
        '''
        load available receptor from table with idx: receptor_idx
        set receptor_folder
        
        Args:
            receptor_idx: 

        Returns:

        '''

        _, _, df = db.get_success_data(receptor_idx, dataframe=True)
        primary_key = db.primary_key_for(receptor_idx)
        df = df[primary_key]
        df = table(df) 
        if self.ligand is None:
            self.ligand  = df 
        else:
            self.ligand = self.ligand & df

        folder_name = db.get_folder(receptor_idx)
        self.receptor_folder = '{}_{}'.format(receptor_idx, folder_name)
        
        return self

    def crystal(self, crystal_idx):
        '''
        load available ligand from table with idx: crystal_idx
        set ligand_folder
        
        Args:
            crystal_idx: int 

        Returns: self

        '''

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

    def exclusion(self, ex_idx):
        '''
        load the ligand that should be exclude from table with idx: ex_idx
        Args:
            ex_idx: int

        Returns: self

        '''

        _, _, df = db.get_success_data(ex_idx, dataframe=True)
        primary_key = db.primary_key_for(ex_idx)
        df = df[primary_key]
        df = table(df)

        if self.exclude is None:
            self.exclude = df
        else:
            self.exclude = self.exclude | df

        return self

    def docked(self, docked_idx):
        '''
        load available docked ligand from table with idx: docked_idx
        set docked_folder
    
        Args:
            docked_idx: int

        Returns: self

        '''

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
        '''
        # select position with overlap value in restriction: rest
        # e.g. rest=[0.1,0.5] overlap ratio : 0.1 <= value <= 0.5
        Args:
            overlap_idx: int index for overlap table
            rest: typle: (a,b) , restriction a < value < b 
                  list: [a,b] , restriction a <= value <= b
                  None: no restriction

        Returns: self

        '''

        _, _, df = db.get_success_data(overlap_idx, dataframe=True)
        primary_key = db.primary_key_for(overlap_idx)
        df = df[primary_key+['overlap_ratio']]
        df = table(df).apply_rest('overlap_ratio',rest)

        if self.position is None:
            self.position = df 
        else:
            self.position = self.docked & df 

        return self

    def rmsd(self, rmsd_idx, rest):
        '''
        select position with rmsd value in restriction: rest
        e.g. rest=[None, 2] rmsd ration : minimum <= value <= 2
        
        Args:
            rmsd_idx: 
            rest: typle: (a,b) , restriction a < value < b 
                  list: [a,b] , restriction a <= value <= b
                  None: no restriction

        Returns: self

        '''
        
        _, _, df = db.get_success_data(rmsd_idx, dataframe=True)
        primary_key = db.primary_key_for(rmsd_idx)
        df= df[primary_key + ['rmsd']]
        df = table(df).apply_rest('rmsd',rest)
        
        if self.position is None:
            self.position = df
        else:
            self.position = self.position & df

        return self

    def native_contact(self, native_contact_idx, rest):
        '''
        select position with native contact ration in restriction: rest
        e.g. rest=None  no restriction on native contact
        Args:
            native_contact_idx:  int
            rest: typle: (a,b) , restriction a < value < b 
                  list: [a,b] , restriction a <= value <= b
                  None: no restriction


        Returns:

        '''


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
        '''
        select available ligand with norm affinity value in restriction: rest
        Args:
            affinity_idx: int
            rest: typle: (a,b) , restriction a < value < b 
                  list: [a,b] , restriction a <= value <= b
                  None: no restriction

        Returns:

        '''
        
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
        '''
        select available receptor with log affinity value in restriction: rest
        Args:
            affinity_idx: int
            rest: typle: (a,b) , restriction a < value < b 
                  list: [a,b] , restriction a <= value <= b
                  None: no restriction

        Returns:

        '''

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
        '''
        export the retrive result as a dataframe table
        Returns: table

        '''
        if self.position is None:
            return self.ligand - self.exclude
        else:
            return self.ligand & self.position - self.exclude

    def export_data_to(self, folder_name, d_format):
        '''
        retrive data to the folder named by [folder_name] and convert them to [d_format]
           
        Args:
            folder_name: the data to export the folder
            d_format: str
                'pkl': python pikle
                'av4': affinity build-in binary format
                'tfr': tensorflow record format
                'pdb': save format as input
                'tfr_one': save ligand and receptor in one TFRecord

        Returns: None

        '''
        
        if not d_format in self.export_fmt:
            raise Exception("Unexpected format {}, available format: {}".\
                                format(d_format, self.export_fmt))
        
        export_dir = os.path.join(config.export_dir, folder_name)
        
        if self.position is None:
            valid = self.ligand - self.exclude
        
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
            valid = self.ligand & self.position - self.exclude
            collection =[]
            
            for keys, group in valid.groupby(['receptor','chain','resnum','resname', self.affinity_key]):
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
            index.append([rec_path, lig_path, aff])

        df = pd.DataFrame(index, columns=['receptor','ligand','affinity'])
        df.to_csv(os.path.join(export_table_dir,'index.csv'), index=False)




def example1():
    '''
    simple example for retrive data

    '''

    # create retrive_data class
    ra = retrive_data()

    # get the info for receptor from table: 2
    ra.receptor(2)

    # get the info for ligand from table: 3
    ra.crystal(3)

    # get the info for affinity from table: 4
    ra.log_affinity(4, None)

    # export the data to the folder named 'test_tfr_one' as TFRecord
    ra.export_data_to('test_tfr_one','tfr')
    
    #table = ra.export_table()

def example2():
    '''
    example to show how to combine result coming from diffrernt way
    '''

    # create retrive_data object
    r  = retrive_data()

    # set the table to get ligand
    # receptor from table:2
    # reordered ligand from table:3
    # docked ligand from table:4
    # affinity information from table:5
    # select the record which have norm affinity value
    rb = r.recpeotr(2).crystal(3).docked(4).norm_affinity(5,None)

    # overlap info from table 6
    # select the position with overlap ratio value <= 0.5
    rc = rb.same().overlap(6,[None,0.5])

    # overlap info from table 6 rmsd info from table 7
    # select teh position with overlap ratio value > 0.5 and rmas value <= 2
    rd = rb.same().overlap(6,(0.5,None)).rmsd(7,[None,2])

    #
    re = rc | rd 
    table = re.export_table()


if __name__ == '__main__':
    print 'retrive_v2'
    example1()

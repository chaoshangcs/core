import os
import sys
from collections import namedtuple
import multiprocessing

"""
variable shared between file
"""

manager = multiprocessing.Manager()
lock = manager.Lock()

"""
Parameter
"""
db_name='affinity.db'

# number of process running at the same time
process_num = 8


# RENAMES:
# [ ] tanimoto_cutoff    # minimum Tanimoto similarity score to be considered 
# [ ] clash_size_cutoff  # ls # no cutoff here!
# [x] base               # database_root
# [x] lig_download_path  # why ?

# [x] downloads          # raw_pdbs
# [x] receptors          # raw_receptors
# [x] raw_ligands
# [x] docked_ligand_path # SMINA_DOCKED ?? 
# [x] docking parameters

# [x] target_list_file     list_of_PDBs_to_download
# [x] lig_target_list_file (Not needed)
 
# [x] add folder for repaired ligands and proteins (with H)
# [x] add folder for minimized hydrogens on ligands and proteins (How does the hydrogen addition happen)
# [x] think how we can deal with multiple docking parameters


# _________________________ Review 2 _________________________________________________


# Bugs
# [ ] 0) Tanimoto similarity cutoff does not work
# [x] 1) Fraction of native contacts does not work
# [ ] 2) similarity cutoff, fraction of native contacts, clash -- all should have a similar simple format
# [x] 3) reorder should become a part of every docking algorithm
# [x] 4) get_same_ligands becomes standalone
# [x] 5) ligand_name_of becomes _ligand_name_of same for receptor_of, receptor_path_of, mkdir, run ---> run_multiptocessor
# [-] 6) calculate_clash becomes a part of detect_overlap
# [x] 7) dock merges with VInardo_do_docking
# [x] 8) Empty files + broken file check (have a log file to do the stop so as to be able to resume from any point )
# [x] broken file is unless "success is written in log"

# Enhancements
# [ ] 0) Dependencies into README.md
# [x] 1) rename into 1_download, 2_split, 3_docking
# [x] 2) split_structure needs medhod (IE: NMR,X-RAY)

# 3) More splitting statistics
# [x] let's also write a number of receptor atoms when splitting
# [x] and the number of receptor chains when splitting
# [x] number of receptor residues when splitting

# 4) write a script that will aether all of the logs together and create many plots about the data in python notebook
# Statistics about the PDB in README.md with all of the plots
# def crystal_ligand_for_same_receptor

# 5) Write the script that will do a retrieval based on 4 and write the structures in av4 format

# 6) How do we do this on Orchestra or Bridges -- how do we launch many separate jobs ?

"""
overlap detaction constant
"""
tanimoto_cutoff = 0.75  # an exact Tanimoto similarity score should be recorded in the file
clash_cutoff_A = 4      # 
clash_size_cutoff = 0.3 # __ an exact value should be recorded


"""
Folders
"""
# the path for this script config.py
script_path = sys.path[0]
#base folder for all the output
#database_root = os.path.join(script_path, '..', 'AffinityDB')
database_root = '/home/xander/affinityDB/test_v2'

db_path =os.path.join(database_root, db_name)

data_dir = os.path.join(database_root,'data')
# log files
log_dir = os.path.join(database_root, 'log')
# csv files
table_dir = os.path.join(database_root, 'table')


"""
File Path 
"""

# path of smina binary file
#smina = 'smina.static'
smina = '/home/xander/Program/smina/smina.static'

# pdb_target_list
#list_of_PDBs_to_download = os.path.join(sys.path[0],'target_list','main_pdb_target_list.txt')
list_of_PDBs_to_download = '/home/xander/affinityDB/target_list/small_list.txt'

# example scoring
scoring_terms = os.path.join(sys.path[0], 'scoring', 'smina.score')


"""
docking para
"""
smina_dock_pm = {
    'vinardo':{
        'args': [],
        'kwargs' : {
            'autobox_add':12,
            'num_modes':400,
            'exhaustiveness':64,
            'scoring':'vinardo',
            'cpu':1
        }
    },
    'smina_default':{
        'args':[],
        'kwargs':{
            'num_modes':400,
            'cpu':1
        }
    },
    'reorder':{
        'args':['score_only'],
        'kwargs':{
            'custom_scoring':scoring_terms 
        }
    }

}

vinardo_pm = {
    'args': [],
    'kwargs' : {
    'autobox_add':12,
    'num_modes':400,
    'exhaustiveness':64,
    'scoring':'vinardo',
    'cpu':1
    }
}

smina_pm = {
    'args':[],
    'kwargs':{
    'num_modes':400,
    'cpu':1
    }
}

reorder_pm = {
    'args':['score_only'],
    'kwargs':{
    'custom_scoring':scoring_terms 
    }
}

overlap_pm = {
    'default':{
        'clash_cutoff_A':4,
        'clash_size_cutoff':0.3
    }
}


overlap_default = {
    'clash_cutoff_A':4,
    'clash_size_cutoff':0.3
}

native_contact_pm = {
    'default':{
        'distance_threshold':4.0
    }
}

native_contace_default = {
    'distance_threshold':4.0
}
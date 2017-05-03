# AffinityDB

## Requirement

Install using `pip install -r requirements.txt`

## Edit config

- `database_root` : dir to store the generated data of AffinityDB
- `smina` : path of (smina)[https://sourceforge.net/projects/smina/] executable file
- `list_of_PDBs_to_download` : path of txt file, the PDB is is written in one line and split by ', '

## Operation

### create

Create a new task. It will generate an unique sn number, create table and folder for this task.

```shell
# create a task to download pdb files
# they're stored under the folder named by
# [sn]_[folder_name] e.g. 1_download
python database_create_v2.py --create --action=download --folder_name=download

# create a tak to calculate native contact
python database_create_v2.py --create --action=native_contact --receptor_sn=[int] --crystal_sn=[int] --docked_sn=[int]
```

### continue

Continue a task.

```shell
# [int] is an unique sn number for a task
python database_create_v2.py --continue --table_sn=[int]
```

### delete

Delete all the relative data for a task. Including the table and folder for this task and all the other task depend on it.

```shell
# [int] is an unique sn number for a task
python database_create_v2.py --continue --table_sn=[int]
```

## Args

Arguments for different kinds of task

**action** : type of task. `download, split_receptor, split_ligand, reorder, smina_dock, rmsd, overlap, native_contact`

**folder_name** : name of folder to store the generated data. ( When create the folder the sn number for the task will be the prefix ) . Required when action in `doanlowd, split_receptor, split_ligand, reorder, smina_dock`

**table_sn** : sn number for a task. Required when `continue, delete` task

**download_sn** : sn for the task which download pdb. Required when action in `split_receptor, split_ligand`

**receptor_sn** : sn for the task which generate receptor. Required when action in `smina_dock, reorder, native_contact`

**ligand_sn** : sn for the task which generate ligand. Required when action in `reorder, smina_dock`

**crystal_sn** : sn for the task which generate ligand. Required when action in `rmsd, overlap, native_contact`

**docked_sn** : sn for the task which docking ligand. Required when action in `rmsd, overlap, native_contact`

**dock_param** : parameter used for docking

## Example

```bash
# download pdb
python database_create_v2.py --create --action=download --folder_name=download

# split the receptor from pdb
python database_create_v2.py --create --action=split_receptor --folder_name=splite_receptor --download_idx=1

# split the ligand from pdb
python database_create_v2.py --create --action=split_ligand --folder_name=splite_ligand --download_idx=1

# reorder the ligand
python database_create_v2.py --create --action=reorder --folder_name=reorder --ligand_idx=3 --receptor_idx=2

# docking
python database_create_v2.py --create --action=smina_dock --folder_name=vinardo --ligand_idx=4 --receptor_idx=2 --param=vinardo

# calculate rmsd
python database_create_v2.py --create --action=rmsd --crystal_idx=4 --docked_idx=5

# calculate overlap
python database_create_v2.py --create --action=overlap --crystal_idx=4 --docked_idx=5

# calculate native_contact
python database_create_v2.py --create --action=native_contact --receptor_idx=2 --crystal_idx=4 --docked_idx=5
```

## Retrive av4

### Install requirement
Install using `pip install -r requirements.txt`

### Edit config.py
- database_root : dir to store the generated data of AffinityDB
- smina : path of (smina)[https://sourceforge.net/projects/smina/] executable file

### Run code

```bash
# download pdb
python database_create_v2.py --create --action=download --folder_name=download

# split the receptor from pdb
python database_create_v2.py --create --action=split_receptor --folder_name=splite_receptor --download_idx=1

# split the ligand from pdb
python database_create_v2.py --create --action=split_ligand --folder_name=splite_ligand --download_idx=1

# parse binding affinity
python database_create_v2.py --create --binding_affinity --param=pdbbind

# retrive av4 file
python retrive_v2.py
```

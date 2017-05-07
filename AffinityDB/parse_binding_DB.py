import numpy as np
import time
import re,math
import csv
from collections import Counter
def read_binding_moad(binding_moad_index):
    class PDB_moad:
        num_records = 0
        num_exception = 0
        pdb_names = []
        ligand_names = []
        binding_affinityes = []
        log_affinities = []
        normalized_affinities = []
        exceptions = []
        states = []
        comments = []


    def parse_entry(entry):
        receptor, res, attr, measure, op, value, unit = entry
        if not attr == 'valid':
            return
        if not measure in ['Kd', 'Ki', 'ic50']:
            return
        if not op in ['=','~']:
            return

        resnames, chain, resnum = res.split(':')
        resnames = resnames.split(' ')

        try:
            value = float(value)
        

            if unit.lower() == 'fm':
                log_affinity = np.log(value) - np.log(10.0**15)
            elif unit.lower() == 'pm':
                log_affinity = np.log(value) - np.log(10.0**12)
            elif unit.lower() == 'nm':
                log_affinity = np.log(value) - np.log(10.0**9)
            elif unit.lower() == 'um':
                log_affinity = np.log(value) - np.log(10.0**6)
            elif unit.lower() == 'mm':
                log_affinity = np.log(value) - np.log(10.0**3)
            elif unit.lower() == 'm':
                log_affinity = np.log(value)
            else:
                raise Exception("unexpected unit {}".format(ligand))

            state = 1
            comment = 'success'
        
        except Exception as e:
            #PDB_moad.exceptions.append(e)
            log_affinity = 0
            state = 0
            comment = str(e)


       

        for resname in resnames:
            PDB_moad.pdb_names.append(receptor.upper())
            PDB_moad.ligand_names.append(resname.upper())
            PDB_moad.log_affinities.append(log_affinity)
            PDB_moad.states.append(state)
            PDB_moad.comments.append(comment)
            PDB_moad.num_records +=1
            if state ==0:
                PDB_moad.num_exception +=1
                PDB_moad.exceptions.append(e)
            

    with open(binding_moad_index) as fin:
        while(not fin.readline() == '"========================="\n'):
            continue
    
        csv_reader = csv.reader(fin)
        receptor = []
        for row in csv_reader:

            if len(row) == 2:
                # smile string and rest
                pass
            elif not row[0]== '':
                # first columns like '2.6.1.62'
                pass
            elif not row[2] == '':
                # get receptor
                receptor = row[2]
            else:
                try:
                    parse_entry([receptor.upper()] + row[3:9])
                except Exception as e:
                    PDB_moad.num_exception +=1
                    PDB_moad.exceptions.append(e)
                PDB_moad.num_records +=1

    max_log_affinity = np.log(10.**0)
    min_log_affinity = np.log(10.**-18)

    PDB_moad.normalized_affinities = (PDB_moad.log_affinities - min_log_affinity)\
    /(max_log_affinity - min_log_affinity)

    print "parsing finished. num records: {:<8d} num exceptions: {:<8d}"\
    .format(PDB_moad.num_records, PDB_moad.num_exception)

    return PDB_moad

def read_PDB_bind(pdb_bind_index = "/home/maksym/PyCharmProjects/datasets/pdbbind/INDEX_general_PL.2016"):
    class PDB_bind:
        num_records = 0
        num_exceptions = 0
        cant_split_lines = []
        cant_split_affinity_and_moilarity = []
        pdb_names = []
        ligand_names = []
        binding_affinities = []
        log_affinities = []
        normalized_affinities = []
        exceptions = []
        states = []
        comments = []

    with open(pdb_bind_index) as f:
        [f.readline() for _ in range(6)]
        file_text = f.readlines()

    for line in file_text:
        pdb_name = re.split("[\s]+", line)[0]
        ligand_name = re.split("[\s]+", line)[6]
        
        try:

            # sanity check
            if re.compile(".*incomplete ligand.*").match(line):
                raise Exception('missing atoms in ligand')

            if re.compile(".*covalent complex.*").match(line):
                raise Exception('forms covalent complex')

            if re.compile(".*Nonstandard assay.*").match(line):
                raise Exception('not standard assay')

            

            if not re.compile("^[a-zA-Z0-9][a-zA-Z0-9][a-zA-Z0-9][a-zA-Z0-9]$").match(pdb_name):
                raise Exception('PDB name in the record is impossible')

            
            if not re.compile("\([a-zA-Z0-9][a-zA-Z0-9][a-zA-Z0-9]\)").match(ligand_name):
                if not re.compile("\([a-zA-Z0-9][a-zA-Z0-9][a-zA-Z0-9]-[a-zA-Z0-9][a-zA-Z0-9][a-zA-Z0-9]\)").match(ligand_name):
                    raise Exception('ligand name is impossible')


            
            affinity_record = re.split("[\s]+", line)[3]

            # split the lines of the PDB bind file record
            molar_affinity_record = re.sub(re.compile('(IC50=|Ki=|Kd=|IC50~|Ki~|Kd~)'), "", affinity_record)
            affinity_number_and_molarity = re.split(re.compile("([0-9]+\.[0-9]+|[0-9]+)"), molar_affinity_record)

            if len(affinity_number_and_molarity) != 3:
                raise Exception('can not split record'), len(affinity_number_and_molarity)

            affinity_number = affinity_number_and_molarity[1]
            if not re.compile("[0-9]").match(affinity_number):
                raise Exception('can not split record')

            molarity = affinity_number_and_molarity[2]
            if not re.compile("(fM|pM|nM|uM|mM)").match(molarity):
                raise Exception('can not split record')

            # convert affinities into normalized affinities
            # fm = -15; pm = -12, nm = -9; uM = -6; mM = -3;
            
            if re.compile("fm|fM|Fm").match(molarity):
                log_affinity = np.log(float(affinity_number)) - np.log(10.0 ** 15)
            elif re.compile("pm|Pm|pM").match(molarity):
                log_affinity = np.log(float(affinity_number)) - np.log(10.0 ** 12)
            elif re.compile("nm|Nm|nM").match(molarity):
                log_affinity = np.log(float(affinity_number)) - np.log(10.0 ** 9)
            elif re.compile("um|uM|Um"):
                log_affinity = np.log(float(affinity_number)) - np.log(10.0 ** 6)
            elif re.compile("mM|Mm|mm"):
                log_affinity = np.log(float(affinity_number)) - np.log(10.0 ** 3)
            
            state = 1
            comment = 'success'    

        except Exception as e:
            PDB_bind.num_exceptions += 1
            PDB_bind.exceptions.append(e)

            state = 0
            comment = str(e)
            log_affinity = 0


        PDB_bind.pdb_names.append(pdb_name)
        PDB_bind.ligand_names.append(ligand_name.strip("(").strip(")"))
        PDB_bind.log_affinities.append(log_affinity)
        PDB_bind.states.append(state)
        PDB_bind.comments.append(comment)

        PDB_bind.num_records += 1

    max_log_affinity = np.log(10.**0)
    min_log_affinity = np.log(10.**-18)
    PDB_bind.normalized_affinities = (PDB_bind.log_affinities - min_log_affinity) / (max_log_affinity - min_log_affinity)
    print "parsing finished. num records:",PDB_bind.num_records,"num exceptions:",PDB_bind.num_exceptions
    return PDB_bind


def read_binding_db(binding_db_index):
    class binding_db:
        num_records = 0
        num_exceptions = 0
        cant_split_lines = []
        cant_split_affinity_and_moilarity = []
        pdb_names = []
        ligand_names = []
        binding_affinities = []
        log_affinities = []
        normalized_affinities = []
        exceptions = []
        states = []
        comments = []

    with open(binding_db_index) as fin:
        reader = csv.reader(fin, delimiter='\t')
        head = reader.next()
        # select the important columns
        #     Ki(nM) 
        #     IC50(nM) 
        #     Kd(nM)
        #     pH
        #     Temp (C)
        #     Ligand HET ID
        #     PDB ID(s) for Ligand-Target Complex
        collections = map(lambda row:[row[8],row[9],row[10],row[14],
                                    row[15],row[26],row[27]],reader)
        
        # remove record if the ligand or receptor ID missing
        valid = filter(lambda row: 
                            not(row[-1]=='' or row[-2]==''), collections)
        
        # remove ligand-receptor record if it appears more than once
        pairs = map(lambda row:(row[-2],row[-1]),valid)
        counter = Counter(pairs)
        unique_pair = [k for k,v in counter.items() if v==1]
        unique_entry = filter(lambda row: 
                                (row[-2],row[-1]) in unique_pair, valid)
        
        # Every record only have one PDB ID
        single_pairs = []
        for entry in unique_entry:
            for pdb in entry[-1].split(','):
                single_pairs.append(entry[:-1]+[pdb])
    
        # remove records which have more than one kind of measure value
        better = []
        for entry in single_pairs:
            # only 25 entry have more than one kind of measure
            c = 0
            pos = 0
            for i in range(3):
                if not entry[i] == '':
                    c +=1
                    pos = i
            if c==1:
                better.append([entry[pos]]+entry[3:])
        
        # remove the record if its affinity value not precise
        temp = filter(lambda x:x[0][0] not in ['<', '>'], better)
        records = map(lambda x:[float(x[0])]+x[1:], temp)

        for affinity, ph, temp, lig, rec in records:
            binding_db.pdb_names.append(rec)
            binding_db.ligand_names.append(lig)
            binding_db.log_affinities.append(np.log(float(affinity)) - np.log(10.0 ** 9))
            binding_db.states.append(1)
            binding_db.comments.append('success')

            binding_db.num_records += 1

        max_log_affinity = np.log(10.**0)
        min_log_affinity = np.log(10.**-18)
        binding_db.normalized_affinities = (binding_db.log_affinities - min_log_affinity) / (max_log_affinity - min_log_affinity)
        print "parsing finished. num records:",binding_db.num_records,"num exceptions:",binding_db.num_exceptions
        return binding_db

            

parse_bind_func = {
    'pdbbind':read_PDB_bind,
    'bindmoad':read_binding_moad,
    'binidngdb':read_binding_db
}



if __name__ == '__main__':
    PDB_bind = read_PDB_bind()

    for i in range(len(PDB_bind.pdb_names)):
        print PDB_bind.pdb_names[i],PDB_bind.ligand_names[i],PDB_bind.normalized_affinities[i]
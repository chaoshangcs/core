import numpy as np
import time
import re,math


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

    with open(pdb_bind_index) as f:
        [f.readline() for _ in range(6)]
        file_text = f.readlines()

    for line in file_text:
        try:

            # sanity check
            if re.compile(".*incomplete ligand.*").match(line):
                raise Exception('missing atoms in ligand')

            if re.compile(".*covalent complex.*").match(line):
                raise Exception('forms covalent complex')

            if re.compile(".*Nonstandard assay.*").match(line):
                raise Exception('not standard assay')

            pdb_name = re.split("[\s]+", line)[0]

            if not re.compile("^[a-zA-Z0-9][a-zA-Z0-9][a-zA-Z0-9][a-zA-Z0-9]$").match(pdb_name):
                raise Exception('PDB name in the record is impossible')

            ligand_name = re.split("[\s]+", line)[6]
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
            PDB_bind.pdb_names.append(pdb_name)
            PDB_bind.ligand_names.append(ligand_name.strip("(").strip(")"))
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
            PDB_bind.log_affinities.append(log_affinity)


        except Exception as e:
            PDB_bind.num_exceptions += 1
            PDB_bind.exceptions.append(e)
        PDB_bind.num_records += 1

    max_log_affinity = np.log(10.**0)
    min_log_affinity = np.log(10.**-18)
    PDB_bind.normalized_affinities = (PDB_bind.log_affinities - min_log_affinity) / (max_log_affinity - min_log_affinity)
    print "parsing finished. num records:",PDB_bind.num_records,"num exceptions:",PDB_bind.num_exceptions
    return PDB_bind


if __name__ == '__main__':
    PDB_bind = read_PDB_bind()

    for i in range(len(PDB_bind.pdb_names)):
        print PDB_bind.pdb_names[i],PDB_bind.ligand_names[i],PDB_bind.normalized_affinities[i]
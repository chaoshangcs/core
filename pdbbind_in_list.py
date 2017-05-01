import math

def get_pdbdata(path_to_text, protein_name = [],binding_energy = []):

    #path_to_textfile = ~/common/data/general-set-except-refined/index/INDEX_core_data.2016
    with open(path_to_text) as f:
        protein_list = f.readlines()
        for protein in protein_list:
            try:
                unit = protein[protein.index("M")-1:protein.index("M")+1]

                if unit == "mM": # 1mM = 1e+9nm
                    binding_energy.append(float(protein[protein.index("=")+1:protein.index("M")-1]) * 1000000000) 
                elif unit == "uM": # 1uM = 1000000pm
                    binding_energy.append(float(protein[protein.index("=")+1:protein.index("M")-1]) * 1000000)
                elif unit == "nM": # 1nM = 1000pm
                    binding_energy.append(float(protein[protein.index("=")+1:protein.index("M")-1]) * 1000)
                elif unit == "pM": # 1pM = 0.001nm
                    binding_energy.append(float(protein[protein.index("=")+1:protein.index("M")-1]))
                else:
                    raise ValueError

                protein_name.append(protein[:4])
            except:
                break
        return protein_name,binding_energy

def get_protein_name():
    return get_pdbdata()[0]

def get_binding_energy(protein_energy = get_pdbdata()[1]):
    log_energy = [1]
    max_energy = math.log(protein_energy[0])
    for data in protein_energy[1:]:
        log_energy.append(math.log(data)/max_energy)
    return log_energy

#path_to_textfile = ~/common/data/general-set-except-refined/index/INDEX_core_data.2016
#database_path = ~/christine/data/pdbbind_av4
def convert_database_to_av4(database_path, pdb_folder):
    """Goes through the list with protein names and protein energy 
    and converts it to av4 """

    #make a directory where the av4 form of the output will be written
    output_path = "~/christine/data/pdbbind_av4"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    def save_av4(filepath,labels,elements):
        labels = np.asarray(labels,dtype=np.int32)
        elements = np.asarray(elements,dtype=np.int32)

        number_of_examples = np.array([len(labels)], dtype=np.int32)
        av4_record = number_of_examples.tobytes()
        av4_record += labels.tobytes()
        av4_record += elements.tobytes()

        f = open(filepath + ".av4", 'w')
        f.write(av4_record)
        f.close()

    path_to_text = "~/common/data/general-set-except-refined/index/INDEX_core_data.2016"
    protein_name, protein_energy = get_pdbdata(path_to_text, protein_name = [],binding_energy = []):

    pdbbind_energy = [1]
    max_energy = math.log(protein_energy[0])
    for data in protein_energy[1:]:
        pdbbind_energy.append(math.log(data)/max_energy)

    for i in range(len(protein_name)):

        #pdbbind_energy, protein_name
        pdb_output_file = output_path + "/" + protein_name[i] + ".av4"
        if os.path.exists(pdb_output_file):
            continue

        if not os.path.exists(pdb_output_file):
            os.makedirs(path_to_pdb_subfolder)

                # convert atomnames to tags and write the data to disk
                def atom_to_number(atomname):
                    atomic_tag_number = atom_dictionary.ATM[atomname.lower()]
                    return atomic_tag_number

                receptor_elements = map(atom_to_number,prody_receptor.getElements())
                ligand_elements = map(atom_to_number,prody_positive.getElements())

                receptor_output_path = path_to_pdb_subfolder + "/" +str(os.path.abspath(dirpath)).split("/")[-1]
                save_av4(receptor_output_path,[0],receptor_elements,prody_receptor.getCoords())
                ligand_output_path = path_to_pdb_subfolder + "/" + path_to_positive.split("/")[-1].split(".")[0]
                save_av4(ligand_output_path,labels,ligand_elements,multiframe_ligand_coords)























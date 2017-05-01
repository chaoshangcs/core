import math

def get_pdbdata(protein_name = [],binding_energy = []):

    #path_to_textfile = ~/common/data/general-set-except-refined/index/INDEX_core_data.2016
    with open("INDEX_core_data.2016") as f:
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

print(get_pdbdata())

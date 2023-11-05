import json
from protein_symmetry import *
import os 

# Open and read the JSON file
with open('current_file_holdings.json', 'r') as json_file:
    data = json.load(json_file)
    
pdb_ids = list(data.keys())
pdb_ids = [x.lower() for x in pdb_ids]

batch_size = 100
counter = 0

for i in tqdm(range(int(len(pdb_ids) / batch_size))):
    pdb_ids_batch = pdb_ids[i*batch_size:(i+1)*batch_size]
    download_cif_files(pdb_ids_batch, cache_dir='cache_dir/pdb_cache/', verbose=False)
    for pdb_id in pdb_ids_batch:
        try:
            coord_array = get_coordinates_from_file(pdb_id, cache_dir='cache_dir/pdb_cache/')
            if coord_array.shape[0] != 0:
                try:
                    chain, ss = get_secondary_structure(pdb_id, cache_dir='cache_dir/pdb_cache/', model_no='A')
                except:
                    break
                if len(chain) > 0 and len(ss) > 0:
                    np.savez(f'cache_dir/np_cache/{pdb_id}.npz', coord_array=coord_array, chain=chain, ss=ss)
                    counter += 1
            os.remove('cache_dir/pdb_cache/' + pdb_id + '.cif')
        except:
            pass
    
pdb_ids_batch = pdb_ids[(i+1)*batch_size:]
download_cif_files(pdb_ids_batch, cache_dir='cache_dir/pdb_cache/', verbose=False)
for pdb_id in pdb_ids_batch:
    try:
        coord_array = get_coordinates_from_file(pdb_id, cache_dir='cache_dir/pdb_cache/')
        if coord_array.shape[0] != 0:
            chain, ss = get_secondary_structure(pdb_id, cache_dir='cache_dir/pdb_cache/', model_no='A')
            np.savez(f'cache_dir/np_cache/{pdb_id}.npz', coord_array=coord_array, chain=chain, ss=ss)
            counter += 1
        os.remove('cache_dir/pdb_cache/' + pdb_id + '.cif')
    except:
        pass
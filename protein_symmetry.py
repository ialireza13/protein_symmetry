from Bio.PDB import FastMMCIFParser, PDBList, Polypeptide, Residue
import numpy as np
from tqdm import tqdm
import re
from scipy.spatial.distance import squareform, pdist
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.graph_objects as go


def get_coordinates_from_file(name, cache_dir, model_no=''):

    parser = FastMMCIFParser(QUIET=True)
    struct = parser.get_structure(structure_id=name.upper(), filename=cache_dir + name.lower()+'.cif')

    model = list(struct.get_models())[0]
    model_nos = list(model.child_dict.keys())
    
    if model_no == '':
        model_no = model_nos[0]
        
    c = model[model_no]
    
    d = []
    for res in c:
            try:
                d.append(res['CA'].coord)
            except KeyError:
                pass

    return np.array(d)


def download_cif_files(list_of_names, cache_dir, file_format='mmCif', verbose=False):

    pdbl = PDBList(verbose=False)
    for name in tqdm(list_of_names, disable=not verbose):
        pdbl.retrieve_pdb_file(name, pdir=cache_dir, file_format=file_format)


def get_secondary_structure_midpoints(secondary_struct_sequence, coordinates):
    res = []
    prev_char = ''
    counter = 0
    ss_locs = [[0,0,0]]
    assert len(secondary_struct_sequence) == coordinates.shape[0]

    for idx, char in enumerate(secondary_struct_sequence):
        if char != prev_char:
            res.append([str(counter) + prev_char, np.mean(ss_locs, axis=0).tolist()])
            counter = 1
            ss_locs = [coordinates[idx]]
            prev_char = char
        else:
            counter += 1
            ss_locs.append(coordinates[idx])
            
    res.append([str(counter) + char, np.mean(ss_locs, axis=0).tolist()])
    
    return res[1:], secondary_struct_sequence


def get_secondary_structure(name, cache_dir, model_no='A'):
    def extract_blocks(cif_path, start_markers, end_marker='#'):
        blocks = {marker: "" for marker in start_markers}
        current_marker = None

        try:
            with open(cif_path, 'r') as file:
                for line in file:
                    # Check if the line matches any of the start markers
                    if any(line.startswith(marker) for marker in start_markers):
                        current_marker = next(marker for marker in start_markers if line.startswith(marker))

                    # If we are inside a block, append the line to the block text
                    if current_marker:
                        blocks[current_marker] += line

                    # Check for the end of a block
                    if line.startswith(end_marker) and current_marker:
                        current_marker = None

        except IOError as e:
            print(f"Error opening file: {e}")
            return None

        return [blocks[marker] for marker in start_markers]

    def process_helix_block(helix_block):
        helices = []
        for line in helix_block.strip().splitlines():
            if line.startswith('_struct_conf') or line.startswith('loop_') or line.startswith('#'):
                continue
            tokens = line.split()
            if len(tokens) < 16 or tokens[4] != 'A':  # Not enough tokens, possibly an invalid line
                continue
            helix_data = (int(tokens[5]), int(tokens[9]))
            helices.append(helix_data)
        return helices

    def process_sheet_block(sheet_block):
        sheets = []
        for line in sheet_block.strip().splitlines():
            if line.startswith('_struct_sheet_range') or line.startswith('loop_') or line.startswith('#'):
                continue
            tokens = line.split()
            if len(tokens) < 10 or tokens[3] != 'A':  # Not enough tokens, possibly an invalid line
                continue
            sheet_data = (int(tokens[4]), int(tokens[8]))
            sheets.append(sheet_data)
        return sheets

    def process_seq_block(sheet_block):
        sequence = []
        for line in sheet_block.strip().splitlines():
            if line.startswith('_entity_poly.pdbx_seq_one_letter_code') or line.startswith('_'):
                continue
            elif (len(line)<10 and line.startswith(';')):
                break
            sequence.append(line.strip(';'))
        return ''.join(sequence)

    def extract_residue_indices(cif_file_path):
        start_markers = ['_struct_sheet_range.sheet_id', '_struct_conf.conf_type_id', '_entity_poly.entity_id']
        sheet_block, conf_block, seq_block = extract_blocks(cif_file_path, start_markers)

        helices = list(set(process_helix_block(conf_block)))
        sheets = list(set(process_sheet_block(sheet_block)))
        seq = process_seq_block(seq_block)

        return helices, sheets, seq
    helices, sheets, seq = extract_residue_indices(cache_dir + name + '.cif')
    
    ss = ['C'] * len(seq)
    for item in helices:
        for i in range(min(item)-1, max(item)):
            ss[i] = 'H'
    for item in sheets:
        for i in range(min(item)-1, max(item)):
            ss[i] = 'S'
    
    return seq, ss

def get_contact_map(coordinates, threshold=8):
    contact_map = squareform(pdist(coordinates))
    contact_map = np.logical_and(contact_map > 0, contact_map <= threshold) * 1
    
    return contact_map


def collapse_secondary_structures(contact_map, secondary_struct_sequence, coordinates):
    res, _ = get_secondary_structure_midpoints(secondary_struct_sequence, coordinates)
    G = nx.from_numpy_array(contact_map)

    idx = 0

    collapsed_coords = []
    collapsed_ss = []

    for r in res:
        n = int(r[0][:-1])
        t = r[0][-1]

        for j in range(1, n):
            G = nx.contracted_nodes(G, idx, idx+j, self_loops=False)

        collapsed_coords.append(coordinates[idx:idx+n, :].mean(axis=0).tolist())
        collapsed_ss.append(t)

        idx = idx + n

    collapsed_coords = np.array(collapsed_coords)
    collapsed_contact_map = nx.adjacency_matrix(G).todense()
    
    return collapsed_coords, collapsed_ss, collapsed_contact_map


def plot_contact_map(contact_map, secondary_structure_sequence, title=None):
    fig, ax = plt.subplots(figsize=(5,5))

    ax.imshow(contact_map)

    color_residue = {'C':'none', 'S':'blue', 'H':'red'}
    w = max(2, len(secondary_structure_sequence) / 20)

    for idx, residue in enumerate(secondary_structure_sequence):
        ax.add_patch(Rectangle((idx-0.5, -w), 1, w-1,
                            alpha=1, facecolor=color_residue[residue]))
        ax.add_patch(Rectangle((-w, idx-0.5), w-1, 1,
                            alpha=1, facecolor=color_residue[residue]))

    plt.ylim(len(contact_map), -w)
    plt.xlim(-w, len(contact_map))
    plt.title(title)
    

def plot_struct(coordinates, secondary_structure_sequence, contact_map=None, chain=None):
    color_map = {'C':'black', 'H':'red', 'S':'blue'}

    fig = go.Figure(data=[go.Scatter3d(x=coordinates[:,0], y=coordinates[:,1], z=coordinates[:,2], showlegend=False, 
                                        marker=dict(size=5, color=[color_map[x] for x in secondary_structure_sequence]), 
                                        mode='markers',
                                        hoverinfo='none')])

    fig.update_layout(scene=dict(xaxis_title='', yaxis_title='', zaxis_title=''))

    if contact_map is not None:
        for i in range(coordinates.shape[0]):
            for j in range(i + 1, coordinates.shape[0]):
                if contact_map[i,j] == 1:
                    fig.add_trace(go.Scatter3d(x=[coordinates[i,0], coordinates[j,0]], y=[coordinates[i,1], coordinates[j,1]], z=[coordinates[i,2], coordinates[j,2]],
                                            mode='lines', showlegend=False, line=dict(width=2, color='black'), 
                                            hoverinfo='none', opacity=0.5))
    fig.update_layout(
        autosize=False,
        width=600,
        height=500
    )
    fig.show()
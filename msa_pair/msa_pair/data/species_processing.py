import os
import json
import time
from tqdm import tqdm
from typing import Sequence, Dict
from collections import defaultdict

import numpy as np
import pandas as pd

from alphafold.data import parsers, pipeline, msa_identifiers, msa_pairing
from alphafold.common import residue_constants

from msa_pair.data import msa_processing



def get_uniref_species(description):
    temp = description[1:]
    if temp.find('TaxID=')!=-1:
        n_index = temp.index('n=')
        temp = temp[n_index:].split(maxsplit=1)[-1]
        TaxID_index = temp.index('TaxID=')
        Tax,[TaxID,RepID] = temp[:TaxID_index].strip(),temp[TaxID_index:].split()
        Tax = Tax.split('=')[1]
        TaxID = int(TaxID.split('=')[1])
        RepID = RepID.split('=')[1]
        return str(TaxID)
    else:
        return ''

def make_msa_features(msas: Sequence[parsers.Msa]):
    """Constructs a feature dict of MSA features."""
    if not msas:
      raise ValueError('At least one MSA must be provided.')

    int_msa = []
    deletion_matrix = []
    species_ids = []
    seen_sequences = set()
    sequences = []
    descriptions = []
    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(
                f'MSA {msa_index} must contain at least one sequence.'
            )
        for sequence_index, sequence in enumerate(msa.sequences):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append(
                [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence]
            )
            
            #--------------------------------------------------------------------
            # # for MSA searched from uniprotKB
            # identifiers = msa_identifiers.get_identifiers(
            #     msa.descriptions[sequence_index]
            # )
            # species_ids.append(identifiers.species_id.encode('utf-8'))

            
            # for MSA seached from uniref
            species_idf = get_uniref_species(msa.descriptions[sequence_index]) 
            species_ids.append(species_idf.encode('utf-8'))
            
            #--------------------------------------------------------------------
            
            deletion_matrix.append(msa.deletion_matrix[sequence_index])
            sequences.append(sequence)
            descriptions.append(msa.descriptions[sequence_index])

    processed_msa = parsers.Msa(
        sequences=sequences,
        descriptions=descriptions,
        deletion_matrix=deletion_matrix,
    )

    num_res = len(msas[0].sequences[0])
    num_alignments = len(int_msa)
    features = {}
    features['deletion_matrix_int'] = np.array(
        deletion_matrix, dtype=np.int32
    )
    features['msa'] = np.array(int_msa, dtype=np.int32)
    features['num_alignments'] = np.array(
        [num_alignments] * num_res, dtype=np.int32)
    features['msa_species_identifiers'] = np.array(
        species_ids, dtype=np.object_
    )

    return features, processed_msa


def make_msa_df(chain_features):
    """Construct DataFrame for species processing
    """
    chain_msa = chain_features['msa']
    # print(chain_msa)
    # exit()
    query_seq = chain_msa[0]
    per_seq_similarity = np.sum(
        query_seq[None] == chain_msa, axis=-1
    ) / float(len(query_seq))
    per_seq_gap = np.sum(chain_msa == 21, axis=-1) / float(len(query_seq))
    msa_df = pd.DataFrame({
        'msa_species_identifiers':
            chain_features['msa_species_identifiers'],
        'msa_row':
            np.arange(len(chain_features['msa_species_identifiers'])),
        'msa_similarity': per_seq_similarity,
        'gap': per_seq_gap
    })
    return msa_df


def create_species_dict(msa_df: pd.DataFrame) -> Dict[bytes, pd.DataFrame]:
    species_lookup = {}
    for species, species_df in msa_df.groupby('msa_species_identifiers'):
        species_lookup[species] = species_df
    return species_lookup


def parse(
    input_files_dict,
    pair_species=False,
):

    msas_dict = {}
    msa_feats_dict = {}
    if pair_species:
        all_species_dict = defaultdict(dict)
    for chain_id, path in input_files_dict.items():
        msas = []
        #for path in paths:
        with open(path) as fh:
            a3m_str = fh.read()
            msa = parsers.parse_a3m(a3m_str)
        msas.append(msa)
        msa_feat, processed_msa = make_msa_features(msas)

        if pair_species:
            msa_df = make_msa_df(msa_feat)
            species_dict = create_species_dict(msa_df)
            for spec, df in species_dict.items():
                if spec == b'':
                    continue
                all_species_dict[spec][chain_id] = df

        msas_dict[chain_id] = processed_msa
        msa_feats_dict[chain_id] = msa_feat
    
    if pair_species:
        return all_species_dict, msas_dict, msa_feats_dict
    else:
        return msas_dict, msa_feats_dict


def parse_pairs(
    pairs: list,
    pair_species=False,
):
    msas_dict = {}
    msa_feats_dict = {}
    chain_ids = ["A", "B"]

    if pair_species:
        all_species_dict = defaultdict(dict)
    for i in range(2): # 0 or 1
        chain_id = chain_ids[i]
        msas = []
        descriptions = [pair[i][0] for pair in pairs]
        sequences = [pair[i][1] for pair in pairs]

        # for desc, seq in zip(descriptions, sequences):
        msas.append(parsers.convert_seq_desc_to_Msa(sequences, descriptions))
        msa_feat, processed_msa = make_msa_features(msas)

        if pair_species:
            msa_df = make_msa_df(msa_feat)
            species_dict = create_species_dict(msa_df)
            for spec, df in species_dict.items():
                if spec == b'':
                    continue
                all_species_dict[spec][chain_id] = df

        msas_dict[chain_id] = processed_msa
        msa_feats_dict[chain_id] = msa_feat
    
    if pair_species:
        return all_species_dict, msas_dict, msa_feats_dict
    else:
        return msas_dict, msa_feats_dict


def pair_species(
    input_files_dict
):
    all_species_dict, msas_dict, msa_feats_dict = parse(
        input_files_dict,
        pair_species=True,
    )

    matched_species_dict = {}
    for spec, dfs in all_species_dict.items():
        if len(dfs) < 2:
            continue
        matched_species_dict[spec] = dfs
    num_residues = {}
    for chain_id, msa_feat in msa_feats_dict.items():
        num_residues[chain_id] = msa_feat['msa'].shape[1]

    fixed = {}
    unfixed = {}
    for spec, dfs in matched_species_dict.items():
        num_seqs = [(chain_id, len(df)) for chain_id, df in dfs.items()]
        if max(_[1] for _ in num_seqs) == 1:
            fixed[spec.decode()] = dict(num_seqs)
        else:
            unfixed[spec.decode()] = dict(num_seqs)

    species_stats = {
        'num_species': {'fixed': len(fixed), 'unfixed': len(unfixed)},
        'num_sequences': {'fixed': fixed, 'unfixed': unfixed},
        'num_residues': num_residues,
    }
    print(f"matched_species: {len(matched_species_dict)} fixed: {len(fixed)} unfixed: {len(unfixed)}")
    return matched_species_dict, msas_dict, msa_feats_dict, species_stats


def process_paired_msas(matched_species_dict, msas_dict):
    processed_msas_dict = {}
    for chain_id, msa in msas_dict.items():
        rows = []
        for spec, dfs in matched_species_dict.items():
            rows += list(dfs[chain_id].msa_row.values)
        rows = sorted(rows)
        if 0 not in rows:
            rows = [0] + rows
        msa = msa_processing._build_msa(msa, rows)
        processed_msas_dict[chain_id] = msa

    return processed_msas_dict

def dump_msa_to_a3m(dst_path, msa):
    with open(dst_path, 'wt') as fh:
        for desc, seq in zip(msa.descriptions, msa.sequences):
            fh.write(f'>{desc}\n{seq}\n')

def read_include_file(include_file):
    included_names = set()
    with open(include_file) as fh:
        for line in fh:
            line = line.strip()
            if line:
                name = line.split()[0]
                included_names.add(name)

    return included_names


def process_batch(src_root, dst_root, include_file=None):
    names = sorted(os.listdir(src_root))
    if include_file is not None:
        included_names = read_include_file(include_file)
        names = [name for name in names if name in included_names]

    for name in tqdm(names):
        input_dir = os.path.join(src_root, name)
        output_dir = os.path.join(dst_root, name)
        output_stats_path = os.path.join(output_dir, 'species_stats.json')
        if os.path.exists(output_stats_path):
            continue 
        result = pair_species(input_dir)
        if result is None:
            continue

        os.makedirs(output_dir, exist_ok=True)

        processed_msas_dict = process_paired_msas(result[0], result[1])
        for chain_id, msa in processed_msas_dict.items():
            dst_chain_dir = os.path.join(output_dir, chain_id)
            os.makedirs(dst_chain_dir, exist_ok=True)
            dump_msa_to_a3m(os.path.join(dst_chain_dir, 'uniprot.a3m'), msa)

        with open(output_stats_path, 'wt') as fh:
            json.dump(result[-1], fh, indent=2, sort_keys=True)


if __name__ == '__main__':
    import sys
    include_file = None if len(sys.argv) <= 3 else sys.argv[3]
    process_batch(sys.argv[1], sys.argv[2], include_file)

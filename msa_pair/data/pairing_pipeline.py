import os
import copy
import json
import logging
import argparse

from typing import Mapping, List
from sympy import sequence

from tqdm import tqdm

import tree
import numpy as np
from alphafold.data import (
    msa_pairing, pipeline, pipeline_multimer, feature_processing
)
from alphafold.common import residue_constants

from msa_pair.data import species_processing
from copy import deepcopy

logger = logging.getLogger(__file__)

def _make_all_seq_msa_features(msa_feats):
    return {
        f'{k}_all_seq': v for k, v in msa_feats.items() if k in (
            msa_pairing.MSA_FEATURES + ('msa_species_identifiers',)
        )
    }

def _make_empty_templates_features(num_res):
    """Construct a default template with all zeros."""                           
    return {
        'template_aatype':
            np.zeros(
                (1, num_res, len(residue_constants.restypes_with_x_and_gap)),
                np.float32
            ),
        'template_all_atom_masks':
            np.zeros(
                (1, num_res, residue_constants.atom_type_num), np.float32
            ),
        'template_all_atom_positions':
            np.zeros(
                (1, num_res, residue_constants.atom_type_num, 3), np.float32
            ),
        'template_domain_names': np.array([''.encode()], dtype=object),
        'template_sequence': np.array([''.encode()], dtype=object),
        'template_sum_probs': np.array([0], dtype=np.float32),
    }

def build_paired_rows(auth_chain_ids, paired_rows_dict):
    paired_rows = []
    all_rows = [paired_rows_dict[chain_id] for chain_id in auth_chain_ids]
    num_rows = len(all_rows[0])
    for i in range(num_rows):
        paired_rows.append([rows[i] for rows in all_rows])
    paired_rows = np.array(paired_rows)
    assert paired_rows.shape == (num_rows, len(auth_chain_ids)), paired_rows.shape
    return paired_rows



# def make_fixed_length(np_example, min_seq_l=1024):
#     if np_example['seq_length'] >= min_seq_l:
#         return np_example
    
#     np_example = deepcopy(np_example)
#     for k, v in np_example.items():
#         if not isinstance(v, int):
#             v_shape = v.shape

#         if k in ['seq_length']:
#             np_example[k] = np.array(min_seq_l)
#         elif k in ['msa', 'deletion_matrix', 'bert_mask', 'msa_mask']:
#             np_example[k] = np.pad(v, ((0, 0), (0, min_seq_l - v_shape[-1])))
#         elif k in ['residue_index', 'asym_id', 'sym_id', 'entity_id', 'entity_mask', 'seq_mask', 'aatype']:
#             np_example[k] = np.pad(v, (0, min_seq_l - v_shape[-1]))
#         elif k in ['template_aatype']:
#             np_example[k] = np.pad(v, ((0, 0), (0, min_seq_l - v_shape[-1])))
#         elif k in ['template_all_atom_mask']:
#             np_example[k] = np.pad(v, ((0, 0), (0, min_seq_l - v_shape[1]), (0, 0)))
#         elif k in ['template_all_atom_positions']:
#             np_example[k] = np.pad(v, ((0, 0), (0, min_seq_l - v_shape[1]), (0, 0), (0, 0)))
#         elif k in ['all_atom_mask']:
#             np_example[k] = np.pad(v, ((0, min_seq_l - v_shape[0]), (0, 0)))
#         elif k in ['all_atom_positions']:
#             np_example[k] = np.pad(v, ((0, min_seq_l - v_shape[0]), (0, 0), (0, 0)))

#     return np_example



class PairingPipeline:
    def process(self, input_dir, paired_rows_dict, padding = True):
        logger.info(f'Build into {input_dir}')
        # load paired msas
        msas_all_seq_dict, msa_all_seq_feats_dict = species_processing.parse(
            input_dir, names=['uniprot.a3m'], pair_species=False
        )

        # print('%s',
        #         tree.map_structure(lambda x: x.shape, msa_all_seq_feats_dict))
        # exit()
        # load unpaired msas
        msas_dict, msa_feats_dict = species_processing.parse( 
            input_dir, names=['uniclust30.a3m'], pair_species=False
        )
        # msas_dict, msa_feats_dict = species_processing.parse( 
        #     input_dir, names=['uniref90.a3m'], pair_species=False
        # )

        # msas_dict: {A/B:{sequences:[n_seq, seq_length], descriptions:[n_seq, -1]}}
        # print('%s',tree.map_structure(lambda x: x.shape, msa_feats_dict))
        # build merged feature
        chains_dict = {}
        for chain_id, msa in msas_dict.items():
            seq, desc = msa.sequences[0], msa.descriptions[0]  # primary seq and desc
            num_res = len(seq)  # seq length
            chain = {
                **_make_all_seq_msa_features(msa_all_seq_feats_dict[chain_id]),  # 'deletion_matrix_int_all_seq', 'msa_all_seq', 'num_alignments_all_seq', 'msa_species_identifiers_all_seq'
                **msa_feats_dict[chain_id],   # 'deletion_matrix_int', 'msa_all', 'num_alignments', 'msa_species_identifiers'
                **pipeline.make_sequence_features(seq, desc, num_res),  # 'aatype', 'between_segment_residues', 'domain_name', 'residue_index', 'seq_length', 'sequence'
                **_make_empty_templates_features(num_res),  # 'template_aatype', 'template_all_atom_masks', 'template_all_atom_positions', 'template_domain_names', 'template_sequence', 'template_sum_probs'
            }

            chains_dict[chain_id] = pipeline_multimer.convert_monomer_features(
                chain, chain_id=chain_id
            )

        chains_dict = pipeline_multimer.add_assembly_features(chains_dict)

        np_example = self.pair_and_merge(chains_dict, paired_rows_dict)
        np_example = pipeline_multimer.pad_msa(np_example, 512)
        logger.info(f"Done building {input_dir}")
        # if padding:
        #     np_example = make_fixed_length(np_example)
        return np_example

    def pair_and_merge(
        self,
        chains_dict: Mapping[str, pipeline.FeatureDict],
        paired_rows_dict: Mapping[str, List[int]],
    ) -> pipeline.FeatureDict:

        feature_processing.process_unmerged_features(chains_dict)
        chains = list(chains_dict.values())

        auth_chain_ids = [str(chain['auth_chain_id']) for chain in chains]
        paired_rows = build_paired_rows(auth_chain_ids, paired_rows_dict)  # [n_pairs, 2]
        # num_alignments_all_seq
        chains = self.create_paired_features(paired_rows, chains)
        chains = msa_pairing.deduplicate_unpaired_sequences(chains)
        chains = feature_processing.crop_chains(
            chains,
            msa_crop_size=feature_processing.MSA_CROP_SIZE,
            pair_msa_sequences=True,
            max_templates=feature_processing.MAX_TEMPLATES,
        )
        # print(feature_processing.MSA_CROP_SIZE)
        # print('-------------',tree.map_structure(lambda x: x.shape, chains))
        # exit()
        np_example = msa_pairing.merge_chain_features(
            np_chains_list=chains,
            pair_msa_sequences=True,
            max_templates=feature_processing.MAX_TEMPLATES,
        )
        np_example = feature_processing.process_final(np_example)
        logger.info(tree.map_structure(lambda x: x.shape, np_example))
        return np_example

    def create_paired_features(self, paired_rows, chains):
        # print(paired_rows)
        # print(chains)
        # print('%s',tree.map_structure(lambda x: x.shape, chains))
        # exit()
        
        chain_keys = chains[0].keys()
        updated_chains = []

        for chain_num, chain in enumerate(chains):
            new_chain = {k: v for k, v in chain.items() if '_all_seq' not in k}
            for feature_name in chain_keys:
                if feature_name.endswith('_all_seq'):
                    feats_padded = msa_pairing.pad_features(
                        chain[feature_name], feature_name
                    )
                    new_chain[feature_name] = \
                        feats_padded[paired_rows[:, chain_num]]  # only take paired seq
            new_chain['num_alignments_all_seq'] = np.asarray(
                len(paired_rows[:, chain_num])
            )
            updated_chains.append(new_chain)
        # print("****************************")
        # print('%s',tree.map_structure(lambda x: x.shape, updated_chains))
        # exit()
        return updated_chains

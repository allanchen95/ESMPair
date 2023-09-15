import os
import copy
import json
import logging
import argparse

from typing import Mapping, List

from tqdm import tqdm

import tree
import numpy as np
from alphafold.data import (
    msa_pairing, pipeline, pipeline_multimer, feature_processing
)
from alphafold.common import residue_constants

from msa_pair.data import species_processing

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


class PairingPipeline:
    def process(self, pairs):
        logger.info(f'Start building')
        
        # load paired msas
        # msa_feats_dict: dict_keys(['deletion_matrix_int', 'msa', 'num_alignments', 'msa_species_identifiers'])
        msas_all_seq_dict, msa_all_seq_feats_dict = species_processing.parse_pairs(
            pairs
        )

        # msas_dict: {A/B:{sequences:[n_seq, seq_length], descriptions:[n_seq, -1]}}
        # print('%s',tree.map_structure(lambda x: x.shape, msa_feats_dict))
        # build merged feature
        chains_dict = {}
        for chain_id, msa in msas_all_seq_dict.items():
            seq, desc = msa.sequences[0], msa.descriptions[0]  # parimary seq
            num_res = len(seq)
            chain = {
                **_make_all_seq_msa_features(msa_all_seq_feats_dict[chain_id]),  # 'deletion_matrix_int_all_seq', 'msa_all_seq', 'num_alignments_all_seq', 'msa_species_identifiers_all_seq'
                # **msa_feats_dict[chain_id],   # 'deletion_matrix_int', 'msa_all', 'num_alignments', 'msa_species_identifiers'
                **pipeline.make_sequence_features(seq, desc, num_res),  # 'aatype', 'between_segment_residues', 'domain_name', 'residue_index', 'seq_length', 'sequence'
                **_make_empty_templates_features(num_res),  # 'template_aatype', 'template_all_atom_masks', 'template_all_atom_positions', 'template_domain_names', 'template_sequence', 'template_sum_probs'
            }
            # sequence key is the primary sequence

            # 'auth_chain_id'
            # only change the feature 'aatype' 'template_aatype' 'template_all_atom_masks' 'template_all_atom_mask
            chains_dict[chain_id] = pipeline_multimer.convert_monomer_features(
                chain, chain_id=chain_id
            )

        # for heterodiamer (A["sequence"] != B["sequence"]), A,B -> A_1,B_1
        chains_dict = pipeline_multimer.add_assembly_features(chains_dict)
        np_example = self.pair_and_merge(chains_dict)
        np_example = pipeline_multimer.pad_msa(np_example, 512)
        logger.info(f"Done building")
        return np_example

    def pair_and_merge(
        self,
        chains_dict: Mapping[str, pipeline.FeatureDict],
        # paired_rows_dict: Mapping[str, List[int]],
    ) -> pipeline.FeatureDict:

        feature_processing.process_unmerged_features(chains_dict)
        chains = list(chains_dict.values())

        # auth_chain_ids = [str(chain['auth_chain_id']) for chain in chains]
        # paired_rows = build_paired_rows(auth_chain_ids, paired_rows_dict)
        # chains = self.create_paired_features(paired_rows, chains)
        for k in range(len(chains)):
            chains[k]['num_alignments_all_seq'] = np.asarray(
                chains[k]["msa_all_seq"].shape[0]
            )

        # chains = msa_pairing.deduplicate_unpaired_sequences(chains)
        chains = feature_processing.crop_chains_v2(
            chains,
            msa_crop_size=feature_processing.MSA_CROP_SIZE,
            max_templates=feature_processing.MAX_TEMPLATES,
        )

        np_example = msa_pairing.merge_chain_features(
            np_chains_list=chains,
            pair_msa_sequences=True,
            max_templates=feature_processing.MAX_TEMPLATES,
        )

        np_example = feature_processing.process_final(np_example)
        logger.info(tree.map_structure(lambda x: x.shape, np_example))

        return np_example

    def create_paired_features(self, paired_rows, chains):
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
                        feats_padded[paired_rows[:, chain_num]]
            new_chain['num_alignments_all_seq'] = np.asarray(
                len(paired_rows[:, chain_num])
            )
            updated_chains.append(new_chain)
        # print("****************************")
        # print('%s',tree.map_structure(lambda x: x.shape, updated_chains))
        # exit()
        return updated_chains

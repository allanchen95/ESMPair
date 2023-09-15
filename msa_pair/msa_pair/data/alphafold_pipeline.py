import os
import copy
import json
import logging
import dataclasses

import numpy as np
from alphafold.data import (
    pipeline, pipeline_multimer, parsers, msa_pairing, feature_processing
)

from msa_pair.data import pairing_pipeline

logger = logging.getLogger(__file__)

class AlphaFoldPipeline:
    def _all_seq_msa_features(self, hit_path, sequence):
        with open(hit_path, 'rt') as fh:
            a3m_str = fh.read()
        msa = parsers.parse_a3m(a3m_str)
        assert msa.sequences[0] == sequence
        all_seq_features = pipeline.make_msa_features([msa])
        valid_feats = msa_pairing.MSA_FEATURES + (
            'msa_species_identifiers',
        )
        feats = {
            f'{k}_all_seq': v for k, v in all_seq_features.items()
            if k in valid_feats
        }

        return feats

    def process(self, input_dir, chain_ids=['A', 'B']):
        input_fasta_path = os.path.join(input_dir, 'multimer.fasta')
        with open(input_fasta_path) as f:
            input_fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)

        chain_id_map = pipeline_multimer._make_chain_id_map(
            sequences=input_seqs,
            descriptions=input_descs
        )
        chain_id_map_path = os.path.join(input_dir, 'chain_id_map.json')
        with open(chain_id_map_path, 'w') as f:
            chain_id_map_dict = {
                chain_id: dataclasses.asdict(fasta_chain)
                for chain_id, fasta_chain in chain_id_map.items()
            }
            json.dump(chain_id_map_dict, f, indent=4, sort_keys=True)

        all_chain_features = {}
        sequence_features = {}
        is_homomer_or_monomer = len(set(input_seqs)) == 1
        for chain_id, fasta_chain in chain_id_map.items():
            if fasta_chain.sequence in sequence_features:
                all_chain_features[chain_id] = copy.deepcopy(
                    sequence_features[fasta_chain.sequence]
                )
                continue
            num_res = len(fasta_chain.sequence)
            sequence_features = pipeline.make_sequence_features(
                sequence=fasta_chain.sequence,
                description=fasta_chain.description,
                num_res=num_res
            )
            uniclust_path = os.path.join(input_dir, chain_id, 'uniclust30.a3m')
            # uniclust_path = os.path.join(input_dir, chain_id, 'uniref90.a3m')
            with open(uniclust_path) as fh:
                a3m_str = fh.read()
                uniclust_msa = parsers.parse_a3m(a3m_str)
            assert uniclust_msa.sequences[0] == fasta_chain.sequence
            msa_features = pipeline.make_msa_features((uniclust_msa,))
            # uniprot_hits.a3m
            uniprot_path = os.path.join(input_dir, chain_id, 'uniprot.a3m')
            # uniprot_path = os.path.join(input_dir, chain_id, 'uniprot_hits.a3m')
            msa_all_seq_features = self._all_seq_msa_features(
                uniprot_path, fasta_chain.sequence
            )

            chain_features = {
                **msa_all_seq_features,
                **sequence_features,
                **msa_features,
                **pairing_pipeline._make_empty_templates_features(num_res),
            }
            chain_features = pipeline_multimer.convert_monomer_features(
                chain_features, chain_id=chain_id
            )
            all_chain_features[chain_id] = chain_features
            sequence_features[fasta_chain.sequence] = chain_features

        all_chain_features = pipeline_multimer.add_assembly_features(
            all_chain_features
        )

        np_example = feature_processing.pair_and_merge(
            all_chain_features=all_chain_features
        )

        # Pad MSA to avoid zero-sized extra_msa.
        np_example = pipeline_multimer.pad_msa(np_example, 512)

        return np_example

import os
import json
from typing import List, Mapping

import numpy as np

from alphafold.data import parsers, msa_pairing, feature_processing
from alphafold.common import residue_constants

def _upgrade_sequences_scores(sequence_scores):
    processed_scores = {}
    for result in sequence_scores.values():
        desc = result['description']
        score = result['score']
        processed_scores[desc] = score

    return processed_scores

def create_paired_rows_dict(
    species_dict,
    msas_dict,
    sequences_scores,
    chain_ids: List[str] = ['A', 'B'],
) -> Mapping[str, List[int]]:

    # upgrade sequences scores 
    for chain_id in chain_ids:
        sequences_scores[chain_id] = _upgrade_sequences_scores(
            sequences_scores[chain_id]
        )

    paired_rows_dict = {chain_id: [0] for chain_id in chain_ids}
    pair_scores = [ np.full((1, len(chain_ids)), 1e6) ]
    for spec, dfs in species_dict.items():
        if spec == b'':
            continue

        # find and sort scores
        scores_ = {}
        for chain_id, df in dfs.items():
            chain_scores_ = []
            rows = df.msa_row.values
            for r in rows:
                desc = msas_dict[chain_id].descriptions[r].split()[0]
                if desc in sequences_scores[chain_id]:
                    s = sequences_scores[chain_id][desc]
                    chain_scores_.append((int(r), s))

            if len(chain_scores_) >= 1:
                scores_[chain_id] = sorted(
                    chain_scores_, key=lambda v: v[-1], reverse=True,
                )

        # match rows within the species
        if len(scores_) > 1:
            num_seqs = min(len(v) for v in scores_.values())
            pair_scores_ = []
            for chain_id in chain_ids:
                if chain_id in scores_:
                    chain_rows_ = [
                        _[0] for _ in scores_[chain_id][:num_seqs]
                    ]
                    chain_scores_ = [
                        _[-1] for _ in scores_[chain_id][:num_seqs]
                    ]
                else:
                    chain_rows_ = [-1] * num_seqs
                    chain_scores_ = [0.] * num_seqs

                paired_rows_dict[chain_id] += chain_rows_
                pair_scores_ += [np.array(chain_scores_)[:,None]]

            pair_scores.append(np.concatenate(pair_scores_, axis=-1))

    # sort rows by their pair scores
    pair_scores = np.concatenate(pair_scores, axis=0)
    # following gives in descending order
    inds_sorted = np.argsort(np.sum(pair_scores, axis=-1))[::-1]
    # acending order     
    #inds_sorted = np.argsort(np.sum(pair_scores, axis=-1))
    paired_rows_dict  = {
        chain_id: [rows[i] for i in inds_sorted] for chain_id, rows in 
        paired_rows_dict.items()
    }
    assert len(pair_scores) == len(paired_rows_dict[chain_ids[0]])
    # check the first row is always zero (that is, the first sequence is 
    # always the query sequence)
    assert all(rows[0] == 0 for rows in paired_rows_dict.values())

    return paired_rows_dict

from contextlib import suppress
import os
import json
from typing import List, Mapping
from collections import defaultdict
import numpy as np
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress = True)

from alphafold.data import parsers, msa_pairing, feature_processing
from alphafold.common import residue_constants
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from operator import itemgetter

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
    # pair_scores = [ np.full((1, len(chain_ids)), 1e6) ]
    pair_scores = [ np.full((1, len(chain_ids)), -1e6) ] # reverse
    

    spec2ind2score = defaultdict(list)
    for spec, dfs in species_dict.items():
        if spec == b'':
            continue
        # spec = b'CERCE'
        # dfs = species_dict[spec]

        # find and sort scores
        scores_ = {}
        for chain_id, df in dfs.items():
            chain_scores_ = []
            rows = df.msa_row.values
            # print(rows)
            for r in rows:
                desc = msas_dict[chain_id].descriptions[r].split()[0]
                if desc in sequences_scores[chain_id]:
                    s = sequences_scores[chain_id][desc]
                    chain_scores_.append((int(r), s))
                # if int(r) == 90:
                #     print(dfs['A'])
                #     print(dfs['B'])
                #     print(desc)
                #     print(chain_scores_)
                    # exit()
            # print(chain_scores_)
            if len(chain_scores_) >= 1:
                scores_[chain_id] = sorted(
                    chain_scores_, key=lambda v: v[-1], reverse=True,
                )
        # print(scores_)
        
        # exit()

        # match rows within the species
        if len(scores_) > 1:
            num_seqs = min(len(v) for v in scores_.values())
            pair_scores_ = []
            for chain_id in chain_ids:
                tmp = {}
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
                tmp[chain_id] = scores_[chain_id][:num_seqs]
                spec2ind2score[spec].append(tmp)
                paired_rows_dict[chain_id] += chain_rows_
                pair_scores_ += [np.array(chain_scores_)[:,None]]
            pair_scores.append(np.concatenate(pair_scores_, axis=-1))
    # sort rows by their pair scores
    pair_scores = np.concatenate(pair_scores, axis=0)
    # inds_sorted = np.argsort(np.sum(pair_scores, axis=-1))[::-1]
    inds_sorted = np.argsort(np.sum(pair_scores, axis=-1)) # reverse
    # print(np.sum(pair_scores, axis=-1)[inds_sorted])
    # print(inds_sorted[:10])
    # # print(np.sum(pair_scores, axis=-1)[inds_sorted][:10])
    
    paired_rows_dict  = {
        chain_id: [rows[i] for i in inds_sorted] for chain_id, rows in 
        paired_rows_dict.items()
    }
    assert len(pair_scores) == len(paired_rows_dict[chain_ids[0]])
    # check the first row is always zero (that is, the first sequence is 
    # always the query sequence)
    assert all(rows[0] == 0 for rows in paired_rows_dict.values())

    return paired_rows_dict



def find_alignment(rowsA, rowsB, sims, tag):
    ori_sims = sims
    lenx, leny = sims.shape
    # print(sims.shape, len(rowsA), len(rowsB))
    assert (lenx == len(rowsA)) and (leny == len(rowsB))
    num_seqs = min(len(rowsA), len(rowsB))
    alignments = []
    if tag == 'local':
        sims = sims.reshape(-1)
        index = np.argsort(-sims)
        det_x = set()
        det_y = set()
        for each in index:
            row, col = int(each/leny), each % leny
            if row in det_x or col in det_y:
                continue
            det_x.add(row)
            det_y.add(col)
            alignments.append((rowsA[row], rowsB[col], sims[each]))
        assert len(alignments) == num_seqs
        # print(alignments)
        # exit()
    elif tag =='global':
        tmp_align = []
        sims = csr_matrix(sims)
        rows, cols = min_weight_full_bipartite_matching(sims, maximize = True)
        sims = sims.toarray()
        assert len(rows) == len(cols) == num_seqs
        for x, y in zip(rows, cols):
            tmp_align.append((rowsA[x], rowsB[y], sims[x][y]))
        alignments = sorted(tmp_align, key=itemgetter(2), reverse=True)
    else:
        raise ValueError(f"No such strategy: {tag}!")
    
    return alignments, num_seqs
            



def create_inter_paired_rows_dict(
    sequences_scores,
    tag,
    chain_ids: List[str] = ['A', 'B'],
) -> Mapping[str, List[int]]:

    paired_rows_dict = {chain_id: [0] for chain_id in chain_ids}
    pair_scores = [1e6]
    for spec, info in sequences_scores.items():
        if spec == b'':
            continue
        rowsA, rowsB, sims = info['rowsA'], info['rowsB'], info['sims']

        alignments, num_seqs = find_alignment(rowsA, rowsB, sims, tag)        

        paired_rows_dict['A'] += [_[0] for _ in alignments]
        paired_rows_dict['B'] += [_[1] for _ in alignments]
        pair_scores += [_[2] for _ in alignments]
    inds_sorted = np.argsort(-np.array(pair_scores))
    paired_rows_dict  = {
        chain_id: [int(rows[i]) for i in inds_sorted] for chain_id, rows in 
        paired_rows_dict.items()
    }
    assert len(pair_scores) == len(paired_rows_dict[chain_ids[0]])
    # check the first row is always zero (that is, the first sequence is 
    # always the query sequence)
    assert all(rows[0] == 0 for rows in paired_rows_dict.values())

    return paired_rows_dict
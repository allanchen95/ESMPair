import copy

from typing import Sequence, Mapping, List

from alphafold.data import parsers, msa_pairing, feature_processing
from alphafold.common import residue_constants


def _build_msa(input_msa, rows):
    return parsers.Msa(
        sequences = [input_msa.sequences[i] for i in rows],
        deletion_matrix = [input_msa.deletion_matrix[i] for i in rows],
        descriptions = [input_msa.descriptions[i] for i in rows],
    )

class _MsaSubset:
    def __init__(self, main_msa: parsers.Msa, rows: Sequence[int]):
        self.msa_subset = {}
        for r in sorted(set(rows)):
            self.msa_subset[r] = {
                key: getattr(main_msa, key)[r] for key in [
                    'sequences', 'descriptions', 'deletion_matrix'
                ]
            }


    def __add__(self, next_subset):
        this_subset = copy.deepcopy(self)
        for r, v in next_subset.msa_subset.items():
            if r in this_subset.msa_subset:
                assert v['sequences'] == this_subset.msa_subset[r]['sequences']
            else:
                this_subset.msa_subset[r] = v

        return this_subset

    def build_msa(self):
        rows_sorted = self.rows
        return parsers.Msa(
            sequences=[self.msa_subset[r]['sequences'] for r in rows_sorted],
            deletion_matrix=[
                self.msa_subset[r]['deletion_matrix'] for r in rows_sorted
            ],
            descriptions=[
                self.msa_subset[r]['descriptions'] for r in rows_sorted
            ],
        )

    @property
    def rows(self):
        return sorted(self.msa_subset)

    def __len__(self):
        return len(self.msa_subset)

class MsaBlock:
    def __init__(
        self,
        msas_dict: Mapping[str, parsers.Msa],
        rows_dict: Mapping[str, List[int]],
    ):
        assert len(set(msas_dict) ^ set(rows_dict)) == 0
        self.msa_subsets = {
            chain_id: _MsaSubset(msa, rows_dict[chain_id]) for chain_id, msa
            in msas_dict.items()
        }

    def __add__(self, next_block):
        this_block = copy.deepcopy(self)
        for chain_id, msa_subset in next_block.msa_subsets.items():
            if chain_id in self.msa_subsets:
                this_block.msa_subsets[chain_id] += msa_subset
            else:
                this_block.msa_subsets[chain_id] = msa_subset

        return this_block

    def get_msas(self):
        return {
            chain_id: msa_subset.build_msa() for chain_id, msa_subset in 
            self.msa_subsets.items()
        }

    def get_rows(self):
        return {
            chain_id: msa_subset.rows for chain_id, msa_subset in 
            self.msa_subsets.items()
        }

    def get_lengths(self):
        return {
            chain_id: len(msa_subset) for chain_id, msa_subset in 
            self.msa_subsets.items()
        }


def create_paired_rows(chains, rows_dict):
    paired_rows = []
    for chain in chains:
        chain_id = str(chain['auth_chain_id'])
        paired_rows.append(rows_dict[chain_id])
    
    paired_rows = np.concatenate(paired_rows, axis=0).T

    return paired_rows

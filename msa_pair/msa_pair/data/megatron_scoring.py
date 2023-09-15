from asyncio import FastChildWatcher
from importlib.machinery import all_suffixes
import string
from tkinter.tix import Tree
from typing import List, Tuple, Mapping

from tqdm import tqdm
import torch
import pandas as pd
from alphafold.data import parsers
import torch.nn.functional as F

from msa_pair.data.msa_processing import MsaBlock

class EsmScoring:
    def __init__(self, msa_transformer, msa_batch_converter):
        self.msa_transformer = msa_transformer.eval()
        self.msa_batch_converter = msa_batch_converter
        deletekeys = dict.fromkeys(string.ascii_lowercase)
        deletekeys["."] = None
        deletekeys["*"] = None
        self.translation = str.maketrans(deletekeys)
        # self.inter_tag = inter_tag
        self.is_cpu = False
        # self.tag = tag

    def set_device(self, is_cpu):
        self.is_cpu = is_cpu
        return self.is_cpu


    def _read_msa(
        self,
        input_msa: parsers.Msa,
        nseq: int
    ) -> List[Tuple[str, str]]:
        def remove_insertions(sequence: str) -> str:
            """ Removes any insertions into the sequence. Needed to load 
                aligned sequences in an MSA.
            """
            return sequence.translate(self.translation)
        input_msa = [
            (desc, seq) for desc, seq in zip(
                input_msa.descriptions[:nseq], input_msa.sequences[:nseq]
            )
        ]
        return [ (desc, remove_insertions(seq)) for desc, seq in input_msa ]

    def score(self, input_msa: parsers.Msa, max_num_msas: int, is_cpu = False, repr_layers=[]):
        """Compute the ColAttn scores
        """
        assert max_num_msas <= 1024
        msa_data = [ self._read_msa(input_msa, max_num_msas) ]
        msa_batch_labels, msa_batch_strs, msa_batch_tokens = \
            self.msa_batch_converter(msa_data)
        if not is_cpu:
            msa_transformer = self.msa_transformer.cuda()
            msa_batch_tokens = msa_batch_tokens.cuda()
        else:
            msa_transformer = self.msa_transformer.cpu()
            msa_batch_tokens = msa_batch_tokens.cpu()
        # print(repr_layers)
        # exit()
        with torch.no_grad():
            all_info = msa_transformer(
                msa_batch_tokens, repr_layers = repr_layers, need_head_weights=True
            )
            col_attention = all_info['col_attentions']
            B, C, R, R = col_attention.size()
            # B, C, R
            col_attention = col_attention[:, :, 0, :]
            col_attention = col_attention.mean(-2).cpu().numpy()

        return col_attention[0][1:] # R - 1

    def score_sequences(
        self,
        species_dict: Mapping[str, Mapping[str, pd.DataFrame]],
        msas_dict: Mapping[str, parsers.Msa],
        take_num_seqs: int = 128,
        max_num_msas: int = 512,
        repr_layers: list = [],
        max_num_species: int = -1,
        show_progress: bool = True,
    ):
        """Compute the scores of sequences that are paired by species
        Args:
            take_num_seqs: max number of sequence per species
            max_num_msas: max number of sequence in an MsaBlock
            max_num_species: max number of species to be processed
        """
        msa_blocks = self._build_msa_blocks(
            species_dict, msas_dict, take_num_seqs=take_num_seqs
        )

        cur_msa_block = MsaBlock(
            msas_dict, {chain_id: [0] for chain_id in msas_dict}
        )
        sequences_scores = {
            chain_id: {
                '0': {
                    'description': str(msa.descriptions[0]),
                    'score': 1e6,
                    'block_num': -1,
                }
            } for chain_id, msa in msas_dict.items()
        }
        # print(repr_layers)
        # exit()
        def _score_cur_block():
            if not hasattr(_score_cur_block, 'block_num'):
                _score_cur_block.block_num = 0
            _score_cur_block.block_num += 1
            chain_msas = cur_msa_block.get_msas()
            rows = cur_msa_block.get_rows()
            for chain_id, chain_msa in chain_msas.items():   
                try:
                    scores_ = self.score(chain_msa, max_num_msas, is_cpu = False, repr_layers=repr_layers)
                except:
                    scores_ = self.score(chain_msa, max_num_msas, is_cpu = True, repr_layers=repr_layers)
                rows_ = rows[chain_id]
                # print(len(scores_), len(rows_))
                assert len(scores_) + 1 == len(rows_)
                for i, (r, s) in enumerate(zip(rows_[1:], scores_), start=1):
                    r = str(int(r))
                    assert r not in sequences_scores[chain_id]
                    # if r in sequences_scores[chain_id]:
                    #     print("orig:", sequences_scores[chain_id][r])
                    #     print("cur: ",  {
                    #     'description': 
                    #         str(chain_msa.descriptions[i]).split()[0],
                    #     'score': float(s),
                    #     'block_num': _score_cur_block.block_num,
                    #     })
                    #     print(r)

                    sequences_scores[chain_id][r] = {
                        'description': 
                            str(chain_msa.descriptions[i]).split()[0],
                        'score': float(s),
                        'block_num': _score_cur_block.block_num,
                    }

        all_msa_blocks = list(msa_blocks.items())
        if max_num_species > 0:
            all_msa_blocks = all_msa_blocks[:max_num_species]
        if show_progress:
            progress = tqdm(all_msa_blocks)
        else:
            progress = all_msa_blocks

        for spec, msa_block in progress:
            cur_lengths = cur_msa_block.get_lengths()
            next_lengths = {
                chain_id:
                this_length if chain_id not in cur_lengths else
                this_length + cur_lengths[chain_id]
                for chain_id, this_length in msa_block.get_lengths().items()
            }
            # print(next_lengths, max_num_msas)
            if max(next_lengths.values()) >= max_num_msas:
                _score_cur_block()
                cur_msa_block = MsaBlock(
                    msas_dict, {chain_id: [0] for chain_id in msas_dict}
                )
            cur_msa_block += msa_block

        if max(cur_msa_block.get_lengths().values()) > 1:
            _score_cur_block()

        return sequences_scores

    def _build_msa_blocks(
        self, species_dict, msas_dict, gap_cutoff=0.4, take_num_seqs=128,
    ):
        def _filter_rows(df):
            df = df.sort_values('msa_similarity', axis=0, ascending=False)
            rows = df.msa_row[df.gap <= gap_cutoff].iloc[:take_num_seqs].values
            return rows.astype(int)

        msa_blocks = {}
        for spec, dfs in species_dict.items():
            rows_dict = {}
            if spec == b'':
                continue

            for chain_id, df in dfs.items():
                rows = _filter_rows(df)
                if len(rows) < 1:
                    continue
                rows_dict[chain_id] = rows

            if len(rows_dict) < 2:
                continue

            msa_blocks[spec] = MsaBlock(msas_dict, rows_dict)

        return msa_blocks

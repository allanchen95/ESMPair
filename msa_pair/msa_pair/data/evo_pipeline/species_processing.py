import os
import string
from collections import defaultdict
from msa_pair.data import species_processing
from alphafold.data import parsers
from Bio import SeqIO
from typing import List, Tuple
import itertools
import string
import numpy as np
import json

deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)


def convert_seq_desc_to_Msa(sequences, descriptions):
    deletion_matrix = []
    for msa_sequence in sequences:
        deletion_vec = []
        deletion_count = 0
        for j in msa_sequence:
            if j.islower():
                deletion_count += 1
            else:
                deletion_vec.append(deletion_count)
                deletion_count = 0
        deletion_matrix.append(deletion_vec)

    # Make the MSA matrix out of aligned (deletion-free) sequences.
    deletion_table = str.maketrans('', '', string.ascii_lowercase)
    aligned_sequences = [s.translate(deletion_table) for s in sequences]
    return parsers.Msa(sequences=aligned_sequences,
                        deletion_matrix=deletion_matrix,
                        descriptions=descriptions)


def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

def parse_pairs(
    input_dir: str,
    paired_row_dict: list,
    chain_ids=['A', 'B'], 
    pair_species=False,
):

    grouped_paths = {
        chain_id: 
            os.path.join(input_dir, chain_id, 'uniprot.a3m')
        for chain_id in chain_ids
    }
    msas_dict = {}
    msa_feats_dict = {}
    chain_ids = ["A", "B"]

    if pair_species:
        all_species_dict = defaultdict(dict)
    for chain_id, paths in grouped_paths.items():
        msas = []
        msa = read_msa(paths, nseq=100000) 
        msa = [msa[_] for _ in paired_row_dict[chain_id]]
        descriptions = [_[0] for _ in msa]
        sequences = [_[1] for _ in msa]
        msas.append(convert_seq_desc_to_Msa(sequences, descriptions))
        msa_feat, processed_msa = species_processing.make_msa_features(msas)

        if pair_species:
            msa_df = species_processing.make_msa_df(msa_feat)
            species_dict = species_processing.create_species_dict(msa_df)
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


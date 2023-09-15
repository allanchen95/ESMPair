import pair_process

"""AlphaFold protein structure prediction."""
import os
import sys
import time
import random
import logging
import argparse
from tqdm import tqdm

import torch
import numpy as np
from alphafold.common import residue_constants
from alphafold.model import model as alphafold_model

from msa_pair.runner import utils
import model_preset_runner

logger = logging.getLogger(__file__)


@torch.no_grad()
def run_alphafold(
    processed_feature_dict,
    model_runner,
    random_seed: int,
    max_num_res: int = -1,
    num_predictions_per_model: int = 1,
):
    num_res = len(processed_feature_dict['aatype'])
    if max_num_res >= 0 and num_res > max_num_res:
        logger.warning(f'Too many residues {num_res}\n')
        return

    all_outputs = model_runner.predict(
        processed_feature_dict,
        rng_seed=random_seed,
        num_predictions_per_model=num_predictions_per_model,
    )

    return all_outputs


"""for testing"""
from Bio import SeqIO
from typing import List, Tuple
import itertools
import string
import json

deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

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

def read_pair(filename: str):
    with open(filename) as file:
        json_dict = json.load(file)
        return json_dict["A"], json_dict["B"]

if __name__ == "__main__":
    input_dir = "/root/dataset/Old_PLM_MSA/data/5xeq"
    msa_A = read_msa(input_dir + "/A/uniprot.a3m", nseq=5000)
    msa_B = read_msa(input_dir + "/B/uniprot.a3m", nseq=5000)

    msa_A, msa_B = msa_A[:100], msa_B[:100]
    msa_pair = [(A, B) for A, B in zip(msa_A, msa_B)]
    cur_pipeline = pair_process.PairingPipeline()
    np_example = cur_pipeline.process(msa_pair)
    print(np_example.keys())

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['CUDA_VISIBLE_DEVICES'] = "7"

    model_runner = model_preset_runner.ModelPresetRunner(
        "/root/dataset/af2_database/",
        ignore_unpaired_sequences=False,
    )

    num_predictions_per_model = 1
    seed = random.randrange(1, 2**24)
    prediction = run_alphafold(np_example, model_runner, seed, num_predictions_per_model)

import os
import json
from tqdm import tqdm
import numpy as np
from msa_pair.data import (
    species_processing, row_processing, pairing_pipeline,
)
import esm


msa_transformer, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_batch_converter = msa_alphabet.get_batch_converter()

def compute_scores(input_dir, dst_path, tag, max_num_msas, is_cpu=False):
    from msa_pair.data import esm_scoring

    species_dict, msas_dict, _, _ = species_processing.pair_species(
        input_dir, names=['uniref90.a3m'], chain_ids=['A', 'B']
        # input_dir, names=['uniprot.a3m'], chain_ids=['A', 'B']
        )
    esm_scorer = esm_scoring.EsmScoring(msa_transformer, msa_batch_converter, tag)
    sequences_scores = esm_scorer.score_sequences(species_dict, msas_dict, max_num_msas=max_num_msas)
    with open(dst_path, 'wt') as fh:
        json.dump(sequences_scores, fh, indent=4, sort_keys=True)


def pair_rows(input_dir, src_score_path, dst_pr_path, tag, overwrite=False):


    with open(src_score_path) as fh:
        sequences_scores = json.load(fh)

    species_dict, msas_dict, _, _ = species_processing.pair_species(
        input_dir, names=['uniref90.a3m'], chain_ids=['A', 'B']
        # input_dir, names=['uniprot.a3m'], chain_ids=['A', 'B']
    )
    paired_rows_dict = row_processing.create_paired_rows_dict(
        species_dict, msas_dict, sequences_scores
    )
    # print(paired_rows_dict)
    # print(paired_rows_dict["A"])
    # print(paired_rows_dict["A"][:10])
    # print(paired_rows_dict["B"][:10])
    # exit()

    with open(dst_pr_path, 'wt') as fh:
        json.dump(paired_rows_dict, fh, indent=4)


def process(input_dir, src_pr_path, dst_path, overwrite=False):
    if not overwrite and os.path.exists(dst_path):
        return

    pipeline = pairing_pipeline.PairingPipeline()

    with open(src_pr_path) as fh:
        paired_rows_dict = json.load(fh)

    try:
        np_example = pipeline.process(input_dir, paired_rows_dict)
    except IOError as e:
        print(e)
        return

    np.savez(dst_path, **np_example)

if __name__ == '__main__':
    # python colattn_pair.py ./dataset/ 4 512 
    import sys
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # use column attention for pairing
    tag = 'col'
    
    # .a3m path
    input_root = sys.argv[1]
    
    # device id
    device_id = sys.argv[2]
    
    # max_msa for each batch: default 512
    max_per_msa = 512
    total_dir_list = os.listdir(input_root)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    err_dirs = []
    for name in tqdm(total_dir_list):
        input_dir = os.path.join(input_root, name)
        
        # calculate and save the column attention score
        score_path = os.path.join(input_dir, f'{tag}_scores_{max_per_msa}.json')
        if not os.path.exists(score_path):
            compute_scores(input_dir, score_path, tag, int(max_per_msa))
        
        # 
        pr_path = os.path.join(input_dir, f'{tag}_pr_{max_per_msa}.json')
        if not os.path.exists(pr_path):
            pair_rows(input_dir, score_path, pr_path, tag)

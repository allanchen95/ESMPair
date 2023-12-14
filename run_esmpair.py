import os, sys
import json
from argparse import ArgumentParser
from pathlib import Path
import zipfile
from tqdm import tqdm
import numpy as np
from msa_pair.data import (
    species_processing, row_processing, pairing_pipeline,
)
import esm
import dataclasses, copy

msa_transformer, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_batch_converter = msa_alphabet.get_batch_converter()

def compute_scores(input_files_dict, out_score_file_path, tag, max_num_msas, is_cpu=False):
    from msa_pair.data import esm_scoring

    species_dict, msas_dict, _, _ = species_processing.pair_species(
        input_files_dict
        # input_dir, names=['uniprot.a3m'], chain_ids=['A', 'B']
        )
    esm_scorer = esm_scoring.EsmScoring(msa_transformer, msa_batch_converter, tag)
    sequences_scores = esm_scorer.score_sequences(species_dict, msas_dict, max_num_msas=max_num_msas)
    with open(out_score_file_path, 'wt') as fh:
        json.dump(sequences_scores, fh, indent=4, sort_keys=True)


def pair_rows(input_files_dict, src_score_path, dst_pr_path, tag, overwrite=False):


    with open(src_score_path) as fh:
        sequences_scores = json.load(fh)
    num_key_seq_scores = copy.deepcopy(sequences_scores)
    species_dict, msas_dict, _, _ = species_processing.pair_species(
        input_files_dict
        # input_dir, names=['uniprot.a3m'], chain_ids=['A', 'B']
    )
    paired_rows_dict = row_processing.create_paired_rows_dict(
        species_dict, msas_dict, sequences_scores
    )

    msa_a_dict = dataclasses.asdict(msas_dict["A"])
    msa_b_dict = dataclasses.asdict(msas_dict["B"])
    pr_idx_a_list = paired_rows_dict["A"]
    pr_idx_b_list = paired_rows_dict["B"]
    """
    for i in range(5):
        print (msa_a_dict['descriptions'][pr_idx_a_list[i]])
        print (msa_a_dict['sequences'][pr_idx_a_list[i]])
        print (msa_b_dict['descriptions'][pr_idx_b_list[i]])
        print (msa_b_dict['sequences'][pr_idx_b_list[i]])
        print ("____", pr_idx_a_list[i], str(pr_idx_a_list[i]))
        print (num_key_seq_scores["A"][str(pr_idx_a_list[i])])
        print (num_key_seq_scores["B"][str(pr_idx_b_list[i])])
    """
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
    import logging
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser (
        description="Script for running ESMPair"
    )

    

    parser.add_argument("files", nargs="+", help="Names of the two MSA files.")
    parser.add_argument("-o", "--outdir", help="Output directory for saving results.")
    
    args = parser.parse_args()
    in_files = args.files

    if len(in_files) == 2:
        file_name_a = in_files[0]
        file_name_b = in_files[1]
    else:
        sys.exit("Please input the names of the two MSA files")

    if (not (Path(file_name_a).is_file())) or (not (Path(file_name_a).is_file())):
        sys.exit("MSA files do not exist.")

    file_path_a = Path(file_name_a)
    file_path_b = Path(file_name_b)     

    # following for compatibility with esmpair code from the research group
    in_files_dict = {
        "A" : file_path_a,
        "B" : file_path_b
    }

    if args.outdir is not None:
        out_dir = Path(args.outdir)
    else:
        out_dir = Path.cwd()

    # following are candidates for input options
    # use column attention for pairing
    # TO DO: make this an option 
    tag = 'col'
    
    # device id - eventually modify for multiple GPUs, etc.
    device_id = 0 
    
    # max_msa for each batch: default 512
    # TO DO: make this an option that can be specified.
    max_per_msa = 512
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    err_dirs = []

    #issue - we are using outdir to store results
    # same name. need to change this for processing multiple pairs of files
    # if we relax the option for files, we need to change this.

 
        
    # calculate and save the column attention score
    score_path = out_dir.joinpath(f'{tag}_scores_{max_per_msa}.json')
    if not os.path.exists(score_path):
        compute_scores(in_files_dict, score_path, tag, int(max_per_msa))
        
    # 
    pr_path = out_dir.joinpath(f'{tag}_pr_{max_per_msa}.json')
    if not os.path.exists(pr_path):
            pair_rows(in_files_dict, score_path, pr_path, tag)

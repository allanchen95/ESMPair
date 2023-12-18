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
import dataclasses, copy, glob

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


def pair_rows(input_files_dict, src_score_path, dst_pr_path, tag, a3m_file, overwrite=False):


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
    pr_idx_a_list.reverse()
    pr_idx_b_list.reverse()
   
    fh1 = open(a3m_file, "w")
    a_len = len(msa_a_dict['sequences'][0])
    b_len = len(msa_b_dict['sequences'][0])
    zero_hdr = f'> {msa_a_dict["descriptions"][0]}  a_seq_n={a_len} {msa_b_dict["descriptions"][0]}  b_seq_len={b_len}' 
    zero_seq = msa_a_dict['sequences'][0] + msa_b_dict['sequences'][0]
    fh1.write(zero_hdr+"\n")
    fh1.write(zero_seq+"\n")
    for i in range(len(pr_idx_a_list)):
        msa_a_hdr_dict = num_key_seq_scores["A"][str(pr_idx_a_list[i])]
        msa_b_hdr_dict = num_key_seq_scores["B"][str(pr_idx_b_list[i])]
        comb_hdr_str = "> "+msa_a_hdr_dict["description"] +f'  a_scr = {msa_a_hdr_dict["score"]:.4f} '
        comb_hdr_str = comb_hdr_str + f'{ msa_b_hdr_dict["description"]}  b_scr = {msa_b_hdr_dict["score"]:.4f} '
        comb_hdr_str = comb_hdr_str + msa_a_dict['descriptions'][pr_idx_a_list[i]] + '  ' + msa_b_dict['descriptions'][pr_idx_b_list[i]] 
        comb_seq = msa_a_dict['sequences'][pr_idx_a_list[i]] + msa_b_dict['sequences'][pr_idx_b_list[i]]
        fh1.write(comb_hdr_str+"\n")
        fh1.write(comb_seq+"\n")
    fh1.close()

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

def get_all_file_prefixes (in_files_list, in_dir_path):
    unique_files_list = []
    filtered_files_list = []
    for file in in_files_list:
        fn = file.stem
        fn = fn.split("_")[0]
        if fn not in unique_files_list:
            unique_files_list.append(fn)
    for file in unique_files_list:
        f1_str = file+"_1.a3m"
        f2_str = file+"_2.a3m"
        fn1 = in_dir_path / f1_str
        fn2 = in_dir_path / f2_str
        if fn1.exists() and fn2.exists():
            filtered_files_list.append(file)
    
    return filtered_files_list

if __name__ == '__main__':
    # python colattn_pair.py ./dataset/ 4 512 
    import logging
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser (
        description="Script for running ESMPair"
    )

    

    parser.add_argument("in_dir",  help="Name of directory containing MSA files")
    parser.add_argument("-o", "--outdir", help="Output directory for saving results.")
    parser.add_argument("-n", "--num_threads", help="For parallel processing")

    args = parser.parse_args()
    in_dir = args.in_dir
    in_dir_path = Path(args.in_dir)

    
    if args.outdir is not None:
        out_dir = Path(args.outdir)
    else:
        out_dir = in_dir_path

    # input file processing
        
    files_list = Path(in_dir).glob('*.a3m')

    if not files_list:
        sys.exit("No MSA files found in the input directory.")
    
    file_prefix_list = get_all_file_prefixes(files_list, in_dir_path)
    if not file_prefix_list:
        sys.exit("Not matched MSA files found in the input directory.")


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

    for file_p in file_prefix_list:
        p1, p2 = file_p+"_1.a3m", file_p+"_2.a3m"
        p_out = file_p+"_paralogs.a3m"
        a3m_fn = out_dir.joinpath(p_out)   
        f1_path = in_dir.joinpath(p1)
        f2_path = in_dir.joinpath(p2)
        in_files_dict = {
            "A" : f1_path,
            "B" : f2_path
        }

        # calculate and save the column attention score
        score_path = out_dir.joinpath(f'{p_prefix}_{tag}_scores_{max_per_msa}.json')
        if not os.path.exists(score_path):
            compute_scores(in_files_dict, score_path, tag, int(max_per_msa))
        
        # 
        pr_path = out_dir.joinpath(f'{p_prefix}_{tag}_pr_{max_per_msa}.json')
        if not os.path.exists(pr_path):
            pair_rows(in_files_dict, score_path, pr_path, tag, a3m_fn)

# ESMPair


## Requirement
+ Installing the lastest version of [AlphaFold](https://github.com/deepmind/alphafold).

## Installation
+ pip install fair-esm
+ cd msa_pair; pip install -e .

## ESMPair pipeline
+ For saving GPU memory, you need to replace the *esm/model/msa_transformer.py* in the directory of {your python path}/site-packages/ with the provided *msa_transformer.py*. In the new *msa_transformer.py*, we remove redundant operations.
+ Run the ColAttn MSA Pairing,  take the  *2p01A* in the *dataset* for example, let's build a paired MSA on heterodimers.
    + Making the subdirectory *A* and *B* in the directory of   *2p01A*;
    + Renaming the *2p01A_domain_0_start_21_end_91.a3m* with *uniref90.a3m* and moving it to the subdirectory *A*. Similarly, renaming *2p01A_domain_1_start_111_end_215.a3m* with *uniref90.a3m* and moving it to *B*;
    + Running: python colattn_pair.py ./dataset/ {device_id} to get the scoring output: *col_scores_512.json* and the final paired output *col_pr_512.json*


## Output format
+ **Scoring output**:  *col_scores_512.json*
{
    'A':{
        "{msa_index}":{
            "blocknum": xxx,
            "description": msa description,
            "score": colattn score.
        }
    }
    'B':{
        "{msa_index}":{
            "blocknum": xxx,
            "description": msa description,
            "score": colattn score.
        }
    }
}

+ **Paired output**:  *col_pr_512.json*
{
    "A":[
        0, # 0 is the index of the primary sequence.
        3, # index of other msas.
        ...
    ],
    "B":[
        0,  
        5,
        ...
    ]
}

Notaly, msas with the same rank from the two chain lists should be paired, such as the (3+1)th sequence from chainA and the (5+1)th sequence from chainB should be paired.
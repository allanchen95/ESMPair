import json
from time import time

from alphafold.common import protein
from alphafold.common import confidence

def move_to_tensor(feat):
    import torch

    new_feat = {}
    for key, value in feat.items():
        new_feat[key] = torch.from_numpy(value).cuda()
    return new_feat

def write_pdb(
    output_pdb: str, processed_feature_dict, ret, plddt=None, is_multimer=True,
):
    unrelaxed_protein = protein.from_prediction(
        features=processed_feature_dict,
        result=ret,
        b_factors=plddt,
        remove_leading_feature_dimension=not is_multimer,
    )

    pdb_string = protein.to_pdb(unrelaxed_protein)

    with open(output_pdb, 'wt') as fh:
        fh.write(pdb_string)

def write_ranking(output_ranking: str, ret, timing, random_seed=None):
    stat = {
        'confidence': float(ret['ranking_confidence']),
        'timing': timing,
    }
    if 'iptm' in ret:
        stat['iptm'] = float(ret['iptm'])
    if 'ptm' in ret:
        stat['ptm'] = float(ret['ptm'])
    if random_seed is not None:
        stat['seed'] = float(random_seed)

    with open(output_ranking, 'wt') as fh:
        json.dump(stat, fh, indent=2, sort_keys=True)

def get_iptm(prediction_result, asym_id):
    confidence_metrics = {}
    if 'predicted_aligned_error' in prediction_result:
        # Compute the ipTM only for the multimer model.
        confidence_metrics['iptm'] = confidence.predicted_tm_score(
            logits=prediction_result['predicted_aligned_error']['logits'],
            breaks=prediction_result['predicted_aligned_error']['breaks'],
            asym_id=asym_id,
            interface=True,
        )

    return confidence_metrics

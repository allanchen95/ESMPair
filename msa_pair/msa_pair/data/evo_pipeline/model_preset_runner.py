import os
import time
import random
import logging
from typing import Mapping, Optional, Sequence, List

import jax
import tree
import numpy as np

from alphafold.data import pipeline_multimer, msa_pairing
from alphafold.model import config, data, model
# from msa_pair.runner import model, modules_multimer

logger = logging.getLogger(__file__)


def clear_mem(device=None):
    '''remove all data from device'''
    backend = jax.lib.xla_bridge.get_backend(device)
    if hasattr(backend,'live_buffers'):
        for buf in backend.live_buffers():
            buf.delete()


def _is_params_transferable(old_params, next_params):
    def _get_all_keys(params):
        all_keys = []
        for path in params.keys():
            for k in params[path].keys():
                all_keys.append(f'{path}/{k}')
        return tuple(sorted(all_keys))

    return _get_all_keys(old_params) == _get_all_keys(next_params)


class ModelPresetRunner:
    def __init__(
        self,
        database: str,
        model_preset: str = 'multimer',
        allow_params_transfer: bool = True,
        cache_params: bool = True,
        ignore_unpaired_sequences: bool = True,
    ):
        assert model_preset == 'multimer', 'Only supports preset multimer'

        self.model_preset = model_preset
        self.database = database
        self.allow_params_transfer = allow_params_transfer
        self.cache_params = cache_params

        self.params_dict = {}
        # Check model parameters
        first_params, first_model_name = None, None
        first_config = None

        self.avail_model_names = []
        for ind, model_name in enumerate(config.MODEL_PRESETS[model_preset]):
            try:
                params = data.get_model_haiku_params(
                    model_name, self.database
                )
                if first_params is None:
                    first_model_name = model_name
                    first_params = params
                    first_config = config.model_config(model_name)
                    print(ind)
                elif self.allow_params_transfer:
                    if not _is_params_transferable(first_params, params):
                        continue
                    if first_config != config.model_config(model_name):
                        continue
                self.avail_model_names.append(model_name)
            except FileNotFoundError as e:
                continue

        assert len(self.avail_model_names) >= 1

        print(type(first_config))
        # first_config["num_recycle"] = 1

        print(f'{len(self.avail_model_names)} models are available')
        print(f'Build RunModel {first_model_name}')
        self.running_model = model.RunModel(first_config, first_params)

        self.ignore_unpaired_sequences = ignore_unpaired_sequences

    def predict(
        self,
        feat: Mapping[str, np.ndarray],
        rng_seed: Optional[int] = None,
        model_names: Optional[List[str]] = None,
        num_predictions_per_model: int = 5,
    ):
        if rng_seed is not None:
            rng = np.random.default_rng(rng_seed)
        else:
            rng = np.random.default_rng(random.randrange(1, 2**24))

        if model_names is None:
            model_names = self.avail_model_names
        else:
            model_names = [
                model_name for model_name in model_names if model_name in 
                self.avail_model_names
            ]

        outputs = {}
        for model_name in model_names:
            for pred_num in range(num_predictions_per_model):
                seed = int(rng.integers(1, 2**24))
                result, timing = self.run_model(
                    feat, model_name, seed
                )
                outputs[f'{model_name}_{pred_num}'] = {
                    'result': result,
                    'timing': timing,
                    'seed': seed,
                }
            logger.info(tree.map_structure(lambda x: x.shape, result))

        return outputs

    def run_model(
        self,
        feat: Mapping[str, np.ndarray],
        model_name: str,
        random_seed: int,
    ):
        t0 = time.time()

        if not self.cache_params or model_name not in self.params_dict:
            params = data.get_model_haiku_params(model_name, self.database)
            if self.cache_params:
                self.params_dict[model_name] = params
        else:
            params = self.params_dict[model_name]

        if self.running_model is not None and self.allow_params_transfer:
            logger.info(f'Transfer {model_name} params to the running model')
            for path in params.keys():
                self.running_model.params[path] = params[path]

        logger.info(
            f'Start prediction using {model_name} with seed {random_seed}'
        )
        if self.ignore_unpaired_sequences:
            feat = self._remove_unpaired_sequences(feat)
        result = self.running_model.predict(feat, random_seed)
        t1 = time.time()

        logger.info(
            f'Finish prediction using {model_name} in {t1 - t0} seconds'
        )
        # exit()
        timing = t1 - t0

        return result, timing

    def _remove_unpaired_sequences(
        self,
        feat: Mapping[str, np.ndarray],
    ):
        paired_sequence_mask = 1
        for chain_id in np.unique(feat['asym_id']):
            if chain_id:
                chain_mask = feat['asym_id'] == chain_id
                paired_sequence_mask *= np.sum(
                    feat['msa'][:,chain_mask] != msa_pairing.MSA_GAP_IDX,
                    axis=-1,
                ) > 0
        paired_sequence_mask = paired_sequence_mask.astype(bool)
        for key, value in feat.items():
            if key in (
                'msa', 
                'deletion_matrix',
                'bert_mask',
                'msa_mask', 
                'cluster_bias_mask',
            ):
                feat[key] = value[paired_sequence_mask]
            elif key in ('num_alignments',):
                feat[key] = np.sum(paired_sequence_mask)

        feat = pipeline_multimer.pad_msa(feat, 512)

        return feat

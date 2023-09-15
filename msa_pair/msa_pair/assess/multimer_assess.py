import os
import io
import sys
import copy
import json
import math
import logging
import argparse
import tempfile
import itertools
import subprocess
from typing import Mapping, Any
from collections import defaultdict

import numpy as np
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBParser import PDBParser
from alphafold.common import residue_constants, protein

from msa_pair.common import sequence_utils, pdb_utils

logger = logging.getLogger(__file__)


def _decode_binary_output(output):
    msg = []
    for line in output.decode().split('\n'):
        msg.append(line)
    return '\n'.join(msg)

class MultimerAssess:
    def __init__(self, dockq_binary_path):
        self.dockq_binary_path = dockq_binary_path
        assert os.path.exists(self.dockq_binary_path)

    @classmethod
    def add_args(cls, parser):
        group = parser.add_argument_group('AlphaFold Evaluation')
        group.add_argument('--query-pdb', type=str)
        group.add_argument('--truth-pdb', type=str)

    def _dockq(
        self, query_pdb: str, truth_pdb: str, receptor_chain: str
    ) -> Mapping[str, Any]:
        cmd_paths = [query_pdb, truth_pdb]
        cmd = (
            [self.dockq_binary_path] + 
            cmd_paths +
            [
                '-native_chain1', receptor_chain, 
                '-model_chain1', receptor_chain, '-no_needle'
            ]
        )
        # print(cmd)
        # exit()
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, stderr = process.communicate()
        retcode = process.wait()
        if retcode:
            msg = _decode_binary_output(stderr)
            if 'length of native is zero' in msg:
                logger.info(
                   "###########################                           \n"
                   "Error during running DockQ:                           \n"
                   f"{msg.strip()}                                        \n"
                   "--------------------------                            \n"
                   "the reason may be that the common_interface defined in\n" 
                   "DockQ is empty                                        \n"
                   "##########################                              "
                )
            else:
                raise RuntimeError(msg)

        def _get_score(line):
            return float(line.split()[1])

        result = {}
        lines = []
        # print(output)
        for line in output.decode().split('\n'):
            if line.startswith('Fnat'):
                result['fnat'] = _get_score(line)
            elif line.startswith('Fnonnat'):
                result['fnonnat'] = _get_score(line)
            elif line.startswith('iRMS'):
                result['irms'] = _get_score(line)
            elif line.startswith('LRMS'):
                result['lrms'] = _get_score(line)
            elif line.startswith('DockQ'):
                result['dockq'] = _get_score(line)
            lines.append(line)

        if 'dockq' not in result:
            result = None
            logger.warning(cmd + lines)

        return result


    def align_residue_index(
        self, query_pdb: str, truth_pdb: str, output_pdb_dir: str, suffix: str
    ):
        query_chains = pdb_utils.from_pdb_string(
            pdb_utils.read_pdb_string(query_pdb)
        )
        if len(query_chains) <= 1:
            raise ValueError(query_pdb)

        truth_chains = pdb_utils.from_pdb_string(
            pdb_utils.read_pdb_string(truth_pdb)
        )
        if len(truth_chains) <= 1:
            return

        # Align every chains
        alns = defaultdict(dict)
        alns_score = defaultdict(dict)
        for qchain in query_chains.keys():
            query_seq = sequence_utils.aatype_to_sequence(
                query_chains[qchain].aatype
            )
            for tchain in truth_chains.keys():
                truth_seq = sequence_utils.aatype_to_sequence(
                    truth_chains[tchain].aatype
                )
                aln, qidx, tidx = sequence_utils.global_align(
                    query_seq, truth_seq
                )
                if aln is None or aln.score < 5:
                    continue
                alns[qchain][tchain] = (aln.score, qidx, tidx)
                alns_score[qchain][tchain] = aln.score

        if len(alns_score) < 2:
            logger.info(
                f'Cannot align {query_pdb} to {truth_pdb}: {alns_score}'
            )
            return

        # Find unambiguous assignments
        final_assignment = defaultdict(dict)
        for qchain in list(alns):
            if len(alns[qchain]) == 1:
                for tchain in alns[qchain]:
                    if tchain in final_assignment:
                        logger.info(
                            f'Cannot align {query_pdb} to {truth_pdb}: '
                            f'{alns_score}'
                        )
                        return
                    final_assignment[tchain][qchain] = alns[qchain][tchain]
                del alns[qchain]

        # Resolve ambiguous assignments
        results = []
        if len(alns) > 0:
            # try all possible permutations
            for qchain in list(alns):
                for tchain in final_assignment:
                    if tchain in alns[qchain]:
                        del alns[qchain][tchain]
            qchains = [qchain for qchain in alns]
            tuples = [list(alns[qchain]) for qchain in qchains]
            n = 0
            for tchains in itertools.product(*tuples):
                if len(set(tchains)) < len(tchains): # ignore this combination
                    continue
                final_assignment_ = copy.deepcopy(final_assignment)
                for t, q in zip(tchains, qchains):
                    final_assignment_[t][q] = alns[q][t]
                output_pdb = os.path.join(
                    output_pdb_dir, f'model_{n}_to_{suffix}.pdb'
                )
                receptor_chain = self._build_new_pdb(
                    final_assignment_, query_chains, truth_chains, output_pdb
                )
                results.append((receptor_chain, output_pdb))
                n += 1
        else:
            output_pdb = os.path.join(
                output_pdb_dir, f'model_0_to_{suffix}.pdb'
            )
            receptor_chain = self._build_new_pdb(
                final_assignment, query_chains, truth_chains, output_pdb
            )
            results.append((receptor_chain, output_pdb))

        if len(results) < 1:
            logger.warning(f'Empty results, the alns_score is {alns_score}')
            return

        return results

    def _build_new_pdb(
        self,
        final_assignment, query_chains, truth_chains,
        output_pdb: str,
        only_backbone: bool = False
    ):
        assigning = []
        receptor_chain = None
        max_score = 0
        for tid in final_assignment:
            for qid, (s, qidx, tidx) in final_assignment[tid].items():
                if s > max_score:
                    max_score = s
                    receptor_chain = tid
                assigning.append((qid, tid, qidx, tidx))

        new_query_chains = pdb_utils.assign_residue_index(
            assigning, query_chains, truth_chains
        )

        if len(new_query_chains) < 2:
            raise RuntimeError(output_pdb, len(query_chains), len(truth_chains))

        new_query_structure = pdb_utils.merge_chains(
            new_query_chains, only_backbone=only_backbone
        )

        pdb_string = protein.to_pdb(new_query_structure)
        p = protein.from_pdb_string(pdb_string)
        with open(output_pdb, 'wt') as fh:
            fh.write(pdb_string)

        return receptor_chain

    def build_models(self, truth_pdb: str, output_prefix: str):
        pdb_str = pdb_utils.read_pdb_string(truth_pdb)
        pdb_fh = io.StringIO(pdb_str)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('none', pdb_fh)
        models = list(structure.get_models())
        # print(models)

        build_full_model =  len(models) > 1
        if build_full_model:
            full_model = Model(0)

        output_models = {}
        num_chains = 0
        for model in models:
            pdb_io = PDBIO()
            pdb_io.set_structure(model)
            output_path = f'{output_prefix}_{model.id}.pdb'
            # print(list(pdb_io))
            # exit()
            with open(output_path, 'wt') as fh:
                pdb_io.save(fh)
            output_models[model.id] = output_path
            # build a model that includes all chains
            if build_full_model:
                chains = sorted(model, key=lambda chain: chain.id)
                PDB_CHAIN_IDS = sorted(protein.PDB_CHAIN_IDS)
                for chain in model:
                    chain.id = PDB_CHAIN_IDS[num_chains]
                    num_chains += 1
                    full_model.add(chain)

        if build_full_model:
            pdb_io = PDBIO()
            pdb_io.set_structure(full_model)
            output_path = f'{output_prefix}_full.pdb'
            with open(output_path, 'wt') as fh:
                pdb_io.save(fh)
            output_models['full'] = output_path

        return output_models


    def assess(
        self,
        query_pdb: str, truth_pdb: str, receptor_chain: str
    ):
        # print(query_pdb, truth_pdb)
        query_model, gt_model = trim_common_residues(query_pdb, truth_pdb)
        with tempfile.NamedTemporaryFile('w+t') as gt_fp:
            pdb_io = PDBIO()
            pdb_io.set_structure(gt_model)
            pdb_io.save(gt_fp)

            with tempfile.NamedTemporaryFile('w+t') as query_fp:
                pdb_io = PDBIO()
                pdb_io.set_structure(query_model)
                pdb_io.save(query_fp)
                try:
                    result = self._dockq(
                        query_fp.name, gt_fp.name, receptor_chain
                    )
                except RuntimeError as e:
                    raise RuntimeError(query_pdb, truth_pdb)
        return result

def get_residues(pdb_str):
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('none', pdb_fh)
    models = list(structure.get_models())
    assert len(models) == 1
    model = models[0]
    residues = {}
    for chain in model:
        for res in chain:
            if res.id[2] != ' ' or res.id[0] != ' ':
                continue
            name = f'{chain.id}_{res.id[1]}'
            residues[name] = res

    return residues

def build_model_from_residues(residues):
    new_model = Model(0)
    chains = {}
    for res_id, res in residues.items():
        chain_id, res_id = res_id.split('_')
        if chain_id not in chains:
            chains[chain_id] = Chain(chain_id)
        chains[chain_id].add(res)

    for chain in chains.values():
        new_model.add(chain)

    return new_model


def trim_common_residues(query_pdb: str, truth_pdb: str):
    truth_residues = get_residues(pdb_utils.read_pdb_string(truth_pdb))
    query_residues = get_residues(pdb_utils.read_pdb_string(query_pdb))
    common_res_ids = set(query_residues.keys()) & set(truth_residues.keys())

    truth_model = build_model_from_residues(dict(
        (res_id, res) for res_id, res in truth_residues.items() if res_id in 
        common_res_ids
    ))
    query_model = build_model_from_residues(dict(
        (res_id, res) for res_id, res in query_residues.items() if res_id in 
        common_res_ids
    ))

    return query_model, truth_model

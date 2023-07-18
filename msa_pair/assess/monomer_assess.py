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

logger = logging.getLogger(__file__)


def _decode_binary_output(output):
    msg = []
    for line in output.decode().split('\n'):
        msg.append(line)
    return '\n'.join(msg)

class MonomerAssess:
    def __init__(self, tmalign_binary_path):
        self.tmalign_binary_path = tmalign_binary_path

    @classmethod
    def add_args(cls, parser):
        group = parser.add_argument_group('AlphaFold Evaluation')
        group.add_argument('--query-pdb', type=str)
        group.add_argument('--truth-pdb', type=str)

    def _run_tmalign(
        self, query_pdb: str, truth_pdb: str
    ) -> Mapping[str, Any]:
        cmd_paths = [query_pdb, truth_pdb]
        cmd = (
            [self.tmalign_binary_path] + 
            cmd_paths
        )
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, stderr = process.communicate()
        retcode = process.wait()
        if retcode:
            msg = _decode_binary_output(stderr)
            if 'Warning! Cannot parse file' in msg:
                logger.warning(msg)
                return
            raise RuntimeError(msg)

        def _get_score(line):
            return float(line.split()[1])

        result = {}
        for line in output.decode().split('\n'):
            if line.startswith('TM-score='):
                result['TMscore'] = float(line.split()[1])
                break

        return result


    def assess(
        self,
        query_pdb: str,
        truth_pdb: str,
    ):
        parser = PDBParser(QUIET=True)
        def _get_chains(pdb_file):
            chains = {}
            structure = parser.get_structure('', pdb_file)
            assert len(structure) == 1
            for chain in structure[0]:
                chains[chain.id] = chain
            return chains

        query_chains = _get_chains(query_pdb)
        truth_chains = _get_chains(truth_pdb)
        common_chains = set(list(query_chains)) & set(list(truth_chains))

        results = {}
        for chain_id in common_chains:
            with tempfile.NamedTemporaryFile('w+t') as gt_fp:
                pdb_io = PDBIO()
                gt_model = truth_chains[chain_id]
                pdb_io.set_structure(gt_model)
                pdb_io.save(gt_fp)

                with tempfile.NamedTemporaryFile('w+t') as query_fp:
                    pdb_io = PDBIO()
                    query_model = query_chains[chain_id]
                    pdb_io.set_structure(query_model)
                    pdb_io.save(query_fp)
                    try:
                        result = self._run_tmalign(query_fp.name, gt_fp.name)
                    except RuntimeError as e:
                        raise RuntimeError(query_pdb, truth_pdb)
                if result:
                    results[chain_id] = result

        return results

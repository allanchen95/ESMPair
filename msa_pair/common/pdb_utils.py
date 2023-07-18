import io
import gzip
import dataclasses
from typing import Mapping, Set

import torch
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from alphafold.common import protein, residue_constants


def trim_pdb_by_residue_index(query_pdb: str, target_pdb: str) -> str:
    tp = build_from_pdb_string(
        read_pdb_string(query_pdb),
        chain_residue_index=build_chain_residue_index(
            read_pdb_string(target_pdb)
        ),
        output_model=True
    )
    return tp

def assign_residue_index(assigning, query_chains, target_chains):
    new_query_chains = {}
    for qi, ti, qidx, tidx in assigning:
        qchain = dataclasses.asdict(query_chains[qi])
        tchain = dataclasses.asdict(target_chains[ti])
        for k in qchain:
            if k == 'residue_index':
                qchain[k] = tchain[k][tidx]
            elif k == 'chain_index':
                t = protein.PDB_CHAIN_IDS.index(ti)
                qchain[k] = np.full(len(tchain[k][tidx]), t)
            else:
                qchain[k] = qchain[k][qidx]
        new_query_chains[ti] = protein.Protein(**qchain)
    return new_query_chains

def merge_chains(
    chains: Mapping[str, protein.Protein],
    only_backbone: bool = False,
) -> protein.Protein:
    ca_atom_order = residue_constants.atom_order['CA']
    p = None
    for _, chain in chains.items():
        chain = dataclasses.asdict(chain)
        if p is not None:
            for k in p:
                p[k] = np.concatenate([p[k], chain[k]], axis=0)
        else:
            p = chain
    if only_backbone:
        ca_mask = p['atom_mask'][:,ca_atom_order]
        atom_mask = np.zeros(p['atom_mask'].shape)
        for atom_type in ('CA', 'N', 'C'):
            atom_order = residue_constants.atom_order[atom_type]
            atom_mask[:,atom_order] = p['atom_mask'][:,atom_order]
        p['atom_mask'] = atom_mask

    return protein.Protein(**p)

def read_pdb_string(path):
    if path.endswith('.gz'):
        with gzip.open(path) as fh:
            pdb_string = fh.read().decode()
    else:
        with open(path) as fh:
            pdb_string = fh.read()
    return pdb_string

def from_pdb_string(pdb_string):
    pdb_fh = io.StringIO(pdb_string)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('none', pdb_fh)
    models = list(structure.get_models())
    model = models[0]
    chains_result = {}
    chain_ids = []
    for chain in model:
        chains_result[chain.id] = from_chain(chain)
        chain_ids += chains_result[chain.id]['chain_ids']
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}

    chains = {}
    for chain_id, chain_result in chains_result.items():
        chains[chain_id] = protein.Protein(
            atom_positions=np.array(chain_result['atom_positions']),
            atom_mask=np.array(chain_result['atom_mask']),
            aatype=np.array(chain_result['aatype']),
            residue_index=np.array(chain_result['residue_index']),
            chain_index=np.array([
                chain_id_mapping[cid] for cid in chain_result['chain_ids']
            ]),
            b_factors=np.array(chain_result['b_factors'])
        )

    return chains

def build_chain_residue_index(pdb_string):
    pdb_fh = io.StringIO(pdb_string)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('none', pdb_fh)
    models = list(structure.get_models())
    model = models[0]
    chain_residue_index = {}
    for chain in model:
        if (chain_residue_index is not None and chain.id in chain_residue_index):
            continue
        chain_residue_index[chain.id] = set()
        for res in chain:
            if res.id[0] != ' ': # remove hetatom
                continue
            if res.id[2] != ' ':
                continue
            chain_residue_index[chain.id].add(res.id)
    return chain_residue_index

def build_from_pdb_string(
    pdb_string,
    chain_residue_index: Mapping[str, Set[int]] = None,
    min_length_cutoff=0,
    output_model=False,
):
    pdb_fh = io.StringIO(pdb_string)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('none', pdb_fh)
    models = list(structure.get_models())
    chains = {}
    chains_counter = {}
    if output_model:
        new_model = Model(0)
    for model in models:
        for chain in model:
            if (
                chain_residue_index is not None 
                and chain.id not in chain_residue_index
            ):
                continue
            new_chain = Chain(chain.id)
            for res in chain:
                if res.id[0] != ' ': # remove hetatom
                    continue
                if res.id[2] != ' ':
                    continue
                res_id = res.id
                if (
                    chain_residue_index is not None
                    and res_id not in chain_residue_index[chain.id]
                ):
                    continue
                if new_chain.has_id(res_id):
                    continue
                new_res = Residue(res.id, res.resname, res.segid)
                for atom in res:
                    new_atom = Atom(
                        atom.name,
                        atom.coord,
                        atom.bfactor,
                        atom.occupancy,
                        atom.altloc,
                        atom.fullname,
                        atom.serial_number,
                        atom.element
                    )
                    new_res.add(new_atom)
                new_chain.add(new_res)

            if len(new_chain) < min_length_cutoff:
                # too few experimentally resolved residues
                continue

            if output_model:
                new_model.add(new_chain)
            else:
                if chain.id not in chains_counter:
                    chains_counter[chain.id] = 0
                chains_counter[chain.id] += 1
                chains[f'{chain.id}{chains_counter[chain.id]}'] = new_chain

    if output_model:
        return new_model
    else:
        return chains

def from_chain(chain):
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []
    for res in chain:
        if res.id[0] != ' ': # remove hetatom
            continue
        if res.id[2] != ' ':
            print(
                f'PDB contains an insertion code at chain {chain.id} and res'
                f' index {res.id[1]}. This insertion is skipped.'
            )
            continue
        res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num
        )
        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for atom in res:
            if atom.name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom.name]] = atom.coord
            mask[residue_constants.atom_order[atom.name]] = 1.
            res_b_factors[residue_constants.atom_order[atom.name]] = \
                atom.bfactor
        if np.sum(mask) < 0.5:
            # If no known atom positions are reported for the residue then skip
            # it
            continue
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        chain_ids.append(chain.id)
        b_factors.append(res_b_factors)

    return {
      'atom_positions': atom_positions,
      'aatype': aatype,
      'atom_mask': atom_mask,
      'residue_index': residue_index,
      'chain_ids': chain_ids,
      'b_factors': b_factors,
    }

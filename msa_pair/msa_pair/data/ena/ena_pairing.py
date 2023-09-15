import os
import json
import math
import string
import logging
import functools
import itertools
from typing import List, Any, Callable, Mapping, Set
from collections import defaultdict

from tqdm import tqdm
from alphafold.data import msa_identifiers, parsers

# from repair.msa import msa_df

logger = logging.getLogger(__file__)

def get_wgs_loci_path(ena_repo, wgs_name,):
    for name in [wgs_name, wgs_name[:6]]:
        path = os.path.join(ena_repo, f'{name}.json')
        if os.path.exists(path):
            return path
     
class EnaPairing:
    def __init__(self, targets_dirs: List[str], ena_repo=None):
        self.targets_dict = self._get_targets_lookup(targets_dirs)
        self.targets_lookup = {}
        self.ena_repo = ena_repo

    def lookup_loci(self, set_name):
        path = get_wgs_loci_path(self.ena_repo, set_name)
        if path is not None:
            with open(path) as fh:
                all_locus = json.load(fh)
            for name, locus in all_locus.items():
                name = name.split('.')[0]
                if name == set_name:
                    return {
                        loci_name.split('.')[0]: loc for loci_name, loc in 
                        locus.items()
                    }

    def _get_targets_lookup(self, targets_dirs):
        targets_dict = defaultdict(list)
        for targets_dir in targets_dirs:
            for filename in os.listdir(targets_dir):
                shortname = filename.split('.')[0].split('/')[-1]
                targets_dict[shortname].append(
                    os.path.join(targets_dir, filename)
                )
        return targets_dict


    def lookup_target(self, ac):
        shortname = ac[3:6]
        if shortname not in self.targets_dict:
            return

        # load target
        if shortname not in self.targets_lookup:
            self.targets_lookup[shortname] = {}
            for p in self.targets_dict[shortname]:
                with open(p) as fh:
                    targets_ = json.load(fh)
            self.targets_lookup[shortname].update(targets_)
        else:
            targets_ = self.targets_lookup[shortname]

        if ac not in targets_:
            return

        return targets_[ac]


    def read_msa(self, src_path):
        with open(src_path) as fh:
            a3m_str = fh.read()
        msa = parsers.parse_a3m(a3m_str)
        return msa


    def parse_paired_accessions(self, input_dir, chain_ids):
        msas_dict = {}
        for chain_id in chain_ids:
            msa = self.read_msa(
                os.path.join(input_dir, chain_id, 'uniprot.a3m')
            )
            msas_dict[chain_id] = msa

        grouped_paired_acs = defaultdict(lambda :defaultdict(set))
        ac_to_row = defaultdict(dict)
        ac_to_cds = {}
        for chain_id, msa in msas_dict.items():
            for r, desc in tqdm(list(enumerate(msa.descriptions))):
            # for r, desc in enumerate(msa.descriptions):
                if r == 0:
                    continue
                identifiers = msa_identifiers._extract_sequence_identifier(
                    desc
                ).split('|')
                ac = identifiers[1]
                ac_to_row[chain_id][ac] = r
                targets = self.lookup_target(ac)
                if targets is None:
                    continue
                set_names = targets['EMBL'].split(',')
                for set_name in set_names:
                    grouped_paired_acs[set_name][chain_id].add(ac)
                cds_names = targets['EMBL-CDS']
                cds_names = [_.split('.')[0] for _ in cds_names.split(',')]
                ac_to_cds[ac] = cds_names
        # clear targets
        self.targets_lookup = {}
        # filter unpaired
        for target in list(grouped_paired_acs.keys()):
            paired_acs = grouped_paired_acs[target]
            if len(paired_acs) <= 1:
                del grouped_paired_acs[target]

        return grouped_paired_acs, ac_to_row, ac_to_cds


    def find_paired_accessions(
        self,
        grouped_paired_acs: Mapping[str, Mapping[str, Set[str]]],
        ac_to_row: Mapping[str, Mapping[str, int]],
        ac_to_cds: Mapping[str, str],
        chain_ids: List[str],
        find_matches: Callable,
    ):
        # find rows
        paired_rows = {chain_id: [] for chain_id in chain_ids}
        request_paired_acs = {}
        request_ac_to_cds = {}
        for target, paired_acs in grouped_paired_acs.items():
            accession_id_lists = [
                list(paired_acs[chain_id]) for chain_id in chain_ids
            ]
            accession_tuples = find_matches(accession_id_lists)
            # update paired rows
            for paired_ac in accession_tuples:
                for i, (chain_id, ac) in enumerate(zip(chain_ids, paired_ac)):
                    paired_rows[chain_id].append(ac_to_row[chain_id][ac])
                    if ac in accession_id_lists[i]:
                        accession_id_lists[i].remove(ac)
            # update paired acs
            paired_acs = {
                chain_id: ids for chain_id, ids in
                zip(chain_ids, accession_id_lists) if len(ids) > 0
            }
            if len(paired_acs) > 1:
                request_paired_acs[target] = paired_acs
                for acs in paired_acs.values():
                    for ac in acs:
                        request_ac_to_cds[ac] = ac_to_cds[ac]

        return request_paired_acs, request_ac_to_cds, paired_rows


    def pair_rows(self, input_dir):
        chain_ids = ['A', 'B']
        grouped_paired_acs, ac_to_row, ac_to_cds = self.parse_paired_accessions(
            input_dir, chain_ids,
        )

        request_paired_acs, request_ac_to_cds, paired_rows = \
            self.find_paired_accessions(
                grouped_paired_acs,
                ac_to_row,
                ac_to_cds,
                chain_ids,
                _find_all_exact_accession_matches,
                # _find_all_accession_matches,
            )

        def get_num_rows(chain_id=chain_ids[0]) -> int:
            return len(paired_rows[chain_id])

        logger.info(f'Before loci pairing: {input_dir} {get_num_rows()}')

        def calc_locus_diff(id_a, id_b, wgs) -> int:
            v_a = [wgs[_] for _ in ac_to_cds[id_a] if _ in wgs]
            v_b = [wgs[_] for _ in ac_to_cds[id_b] if _ in wgs]
            if len(v_a) == 0 or len(v_b) == 0: 
                return math.inf
            return min(abs(a[0] - b[0]) for a, b in itertools.product(v_a, v_b))


        # wgs_names = tqdm(list(request_paired_acs.keys()))
        wgs_names = request_paired_acs.keys()
        for wgs_name in wgs_names:
            paired_acs = request_paired_acs[wgs_name]
            wgs = self.lookup_loci(wgs_name)
            if not wgs:
                continue
            wgs_paired_acs = {} 
            for chain_id, acs in paired_acs.items():
                acs = [ac for ac in acs if ac in ac_to_cds]
                if len(acs) > 0:
                    wgs_paired_acs[chain_id] = acs
            if len(wgs_paired_acs) <= 1:
                continue

            wgs_paired_acs = {wgs_name: wgs_paired_acs}
            _find_all_locus_matches = functools.partial(
                _find_all_accession_matches,
                diff_cutoff=20,
                _calc_id_diff=lambda a, b: calc_locus_diff(a, b, wgs),
            )
            _, _, wgs_paired_rows = self.find_paired_accessions(
                wgs_paired_acs,
                ac_to_row,
                ac_to_cds,
                chain_ids,
                _find_all_locus_matches,
            )
            for chain_id in paired_rows:
                paired_rows[chain_id] += wgs_paired_rows[chain_id]

        assert all(
            get_num_rows(chain_ids[i]) == get_num_rows() for i in 
            range(1,len(paired_rows))
        )

        logger.info(f'After loci pairing: {input_dir} {get_num_rows()}')

        return paired_rows

    def export_paired_accessions(self, input_dir):
        chain_ids = ['A', 'B']
        grouped_paired_acs, ac_to_row, ac_to_cds = self.parse_paired_accessions(
            input_dir, chain_ids
        )
        request_paired_acs, request_ac_to_cds, paired_rows = \
            self.find_paired_accessions(
                grouped_paired_acs,
                ac_to_row,
                ac_to_cds,
                chain_ids,
                _find_all_exact_accession_matches,
            )

        def get_num_rows(chain_id=chain_ids[0]) -> int:
            return len(paired_rows[chain_id])

        logger.info(f'{get_num_rows()}')

        format_paired_acs = {}
        for target, paired_acs in request_paired_acs.items():
            format_paired_acs[target] = {
                chain_id: ','.join(acs) for chain_id, acs in paired_acs.items()
            }
        logger.warning(len(request_paired_acs))

        return {
            'uniprot_accessions': format_paired_acs,
            'embl_cds': request_ac_to_cds,
        }


"""From AlphaFold-v2.1.2
"""
ALPHA_ACCESSION_ID_MAP = {x: y for y, x in enumerate(string.ascii_uppercase)}
ALPHANUM_ACCESSION_ID_MAP = {
    chr: num for num, chr in enumerate(string.ascii_uppercase + string.digits)
}  # A-Z,0-9
NUM_ACCESSION_ID_MAP = {str(x): x for x in range(10)}

@functools.lru_cache(maxsize=65536)
def encode_accession(accession_id: str) -> int:
    """Map accession codes to the serial order in which they were assigned."""
    alpha = ALPHA_ACCESSION_ID_MAP        # A-Z
    alphanum = ALPHANUM_ACCESSION_ID_MAP  # A-Z,0-9
    num = NUM_ACCESSION_ID_MAP            # 0-9

    coding = 0

    # This is based on the uniprot accession id format
    # https://www.uniprot.org/help/accession_numbers
    if accession_id[0] in {'O', 'P', 'Q'}:
        bases = (alpha, num, alphanum, alphanum, alphanum, num)
    elif len(accession_id) == 6:
        bases = (alpha, num, alpha, alphanum, alphanum, num)
    elif len(accession_id) == 10:
        bases = (
            alpha, num, alpha, alphanum, alphanum, num, alpha, alphanum,
            alphanum, num
        )

    product = 1
    for place, base in zip(reversed(accession_id), reversed(bases)):
        coding += base[place] * product
        product *= len(base)

    return coding


def calc_accession_diff(id_a, id_b) -> int:
    return abs(encode_accession(id_a) - encode_accession(id_b))

def _find_all_accession_matches(
    accession_id_lists: List[List[str]],
    diff_cutoff: int = 20, # **strictly** less than diff_cutoff
    _calc_id_diff: Callable = calc_accession_diff,
) -> List[List[Any]]:
    """Finds accession id matches across the chains based on their difference.
    """
    all_accession_tuples = []
    current_tuple = []
    tokens_used_in_answer = set()

    def _matches_all_in_current_tuple(inp, diff_cutoff) -> bool:
        return all((_calc_id_diff(s, inp) < diff_cutoff for s in current_tuple))

    def _all_tokens_not_used_before() -> bool:
        return all((s not in tokens_used_in_answer for s in current_tuple))

    def dfs(level, accession_id, diff_cutoff=diff_cutoff) -> None:
        if level == len(accession_id_lists) - 1:
            if _all_tokens_not_used_before():
                all_accession_tuples.append(list(current_tuple))
                for s in current_tuple:
                    tokens_used_in_answer.add(s)
            return

        if level == -1:
            new_list = accession_id_lists[level+1]
        else:
            new_list = [
                (_calc_id_diff(accession_id, s), s) for s in 
                accession_id_lists[level+1]
            ]
            new_list = sorted(new_list)
            new_list = [s for d, s in new_list]

        for s in new_list:
            if (
                _matches_all_in_current_tuple(s, diff_cutoff)
                and s not in tokens_used_in_answer
            ):
                current_tuple.append(s)
                dfs(level + 1, s)
                current_tuple.pop()

    dfs(-1, '')

    return all_accession_tuples

def _find_all_exact_accession_matches(accession_id_lists):
    return _find_all_accession_matches(
        accession_id_lists,
        diff_cutoff=1,
        _calc_id_diff=calc_accession_diff,
    )

def filter_names(names, include_file):
    included_names = set()
    with open(include_file) as fh:
        for line in fh:
            line = line.strip()
            if line:
                name = line.split()[0]
                included_names.add(name)

    return [name for name in names if name in included_names]


def batch_process(names, input_root, use_tqdm):
    if use_tqdm:
        names = tqdm(names)
    for name in names:
        ena_pairing = EnaPairing(['idmappings'], 'ena_repo')
        # logger.info(f'Start pairing {name}')
        sub_dir = os.path.join(input_root, name)
        pr_path = os.path.join(sub_dir, 'ena_pr.json')
        if os.path.exists(pr_path):
            continue 
        paired_rows = ena_pairing.pair_rows(sub_dir)
        with open(pr_path, 'wt') as fh:
            json.dump(paired_rows, fh, indent=2)

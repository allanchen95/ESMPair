import os
import json
import mmap
import time
import logging
from collections import defaultdict

from tqdm import tqdm

logger = logging.getLogger(__file__)


def parse_idmapping_file(
    idmapping_file: str,
    dst_file: str = None,
    use_mmap: bool = False,
):
    accessions = {}
    t0 = time.time()
    # Read accessions
    with open(idmapping_file) as fh:
        if use_mmap:
            mm = mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ)
            lines = [line.decode() for line in mm.read().split(b'\n')]
            progress = tqdm(lines)
        else:
            progress = tqdm(fh)
        for line in iterator:
            line = line.strip()
            fields = line.split()
            accession = fields[0]
            key = fields[1]
            target = fields[2]
            if key in ('EMBL', 'EMBL-CDS',):
                if accession not in accessions:
                    accessions[accession] = {
                        'EMBL': [],
                        'EMBL-CDS': [],
                    }
                accessions[accession][key].append(target)

    t1 = time.time()

    if t1 - t0 > 3*60:
        logger.warning(
            f'[W] Read {len(accessions)} accessions in {t1-t0:.2f}s'
        )

    return accessions


def split_src_file(accessions, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    split_accessions = defaultdict(dict)
    ac_keys = list(accessions.keys())
    for ac in tqdm(ac_keys):
        ac_dict = accessions[ac]
        split_accessions[ac[3:6]][ac] = {
            k: ','.join(vs) for k, vs in ac_dict.items()
        }

    logger.warning(len(split_accessions))
    prefices = list(split_accessions.keys())
    for prefix in tqdm(prefices):
        sub_accessions = split_accessions[prefix]
        output_path = os.path.join(dst_dir, f'{prefix}.json')
        with open(output_path, 'wt') as fh:
            json.dump(sub_accessions, fh, indent=2)


def update_legacy(src_dir: str):
    for src_filename in tqdm(os.listdir(src_dir)):
        src_path = os.path.join(src_dir, src_filename)
        if not src_filename.endswith('.json'):
            continue
        with open(src_path, 'rt') as fh:
            src_data = json.load(fh)
        dst_data = {}
        for key, values in src_data.items():
            if isinstance(values, str):
                dst_data[key] = values
                continue
            values = sorted([v['EMBL'] for v in values if 'EMBL' in v])
            if len(values) > 0:
                dst_data[key] = ','.join(values)
        with open(src_path, 'wt') as fh:
            src_data = json.dump(dst_data, fh, indent=2)

if __name__ == '__main__':
    import sys
    parser = IdMappingParser()
    accessions = parse_idmapping_file(sys.argv[1])
    split_src_file(accessions, sys.argv[2])

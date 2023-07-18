import os
import re
import gzip
import json
import time
import ftplib
import logging
import functools
import itertools
import urllib.request
import multiprocessing as mp
from typing import Mapping, List, Any
from collections import defaultdict, OrderedDict

from tqdm import tqdm

from repair.ena import ena_pairing

EMBL_EXT = '.cds.gz'
FTP_BASE = 'ftp.ebi.ac.uk'
EMBL_VIEW_URL_BASE = 'https://www.ebi.ac.uk/ena/browser/api/embl'
BASE = '/pub/databases/ena/coding'
SET_TYPES = ['wgs', 'tsa', 'tls']
STATUSES = ['public', 'suppressed']

EMBL_WORDS = { 'ID', 'PA', 'OS', 'FT', '//' }

logger = logging.getLogger(__file__)


def get_basename(filename, stem=False):
    filename = filename.split('/')[-1]
    if stem:
        filename = filename.split('.')[0]
    return filename

def format_wgs_url(
    accession: str,
    prefices: List[str] = ['wgs/public', 'wgs/suppressed'],
):
    sub_dir = accession[:3].lower()
    filename = f'{accession[:6]}{EMBL_EXT}'
    for prefix in prefices:
        url = (
            f'https://{FTP_BASE}/{BASE.strip("/")}/{prefix}/{sub_dir}/{filename}'
        )
        try:
            response = urllib.request.urlopen(url)
        except Exception as e:
            continue
        return url

def format_sequence_url(accession):
    url = f'{EMBL_VIEW_URL_BASE}/{accession}'
    try:
        response = urllib.request.urlopen(url)
        if url == response.url: # "sequence" target
            return url
        elif not response.url.endswith('dat.gz'):
            print(url, response.url)
    except Exception as e:
        return

def batch_get_urls(is_master, acs):
    # if is_master:
    #     acs = tqdm(acs)
    urls = []
    for ac in acs:
        for g in [format_wgs_url, format_sequence_url]:
            url = g(ac)
            if url is not None:
                break
        else:
            logger.info(f'[W] Cannot find exact match to {ac} to download.')
        if url is not None:
            urls.append(url)
    return urls


def retrieve_url(url, dst_file):
    try:
        urllib.request.urlretrieve(url, dst_file)
        ret_code = 0
    except Exception as e:
        logger.error(f'[E] {e}')
        ret_code = -1

    return ret_code

def download_record(url: str, dst_dir: str):
    dst_dat_file = os.path.join(dst_dir, get_basename(url))
    ret_code = retrieve_url(url, dst_dat_file)
    if ret_code == 0:
        return dst_dat_file


START_SEGMENT_PATTERN = re.compile("<(\d+)\.\.>?\d+")
END_SEGMENT_PATTERN = re.compile("<?\d+\.\.>(\d+)")
SEGMENT_PATTERN = re.compile("<?(\d+)\.\.>?(\d+)")


def batch_download_and_parse(
    is_master: bool, urls: List[Any], dst_dir: str
):
    # if is_master:
    #     urls = tqdm(urls)

    for url in urls:
        _download_and_parse(url, dst_dir, remove_dat_file=True)


def _download_and_parse(
    url: str,
    dst_dir: str,
    remove_dat_file: bool = False,
):
    dst_file = os.path.join(
        dst_dir,
        os.path.basename(url).split('.')[0] + '.json'
    )
    if os.path.exists(dst_file):
        return
    dst_dat_file = download_record(url, dst_dir,)
    if dst_dat_file is None:
        logger.warning(f'[I] No exact match to {url} can be downloaded')
        return

    if os.path.exists(dst_dat_file):
        if dst_dat_file.endswith('.gz'):
            try:
                with gzip.open(dst_dat_file, 'rb') as fh:
                    src_str = fh.read().decode()
            except Exception as e:
                print(url, e)
                return -1
            cds_sorted = parse_wgs_cds(src_str)
        else:
            with open(dst_dat_file, 'rt') as fh:
                src_str = fh.read()
            cds_sorted = parse_sequence_cds(src_str)

        with open(dst_file, 'wt') as fh:
            json.dump(cds_sorted, fh, indent=2, sort_keys=False)

    if remove_dat_file:
        if os.path.exists(dst_dat_file):
            os.remove(dst_dat_file)


def parse_cds(value):
    start, end = None, None
    m = re.search(START_SEGMENT_PATTERN, value['CDS'])
    if m is not None:
        start = int(m.group(1))

    m = re.search(END_SEGMENT_PATTERN, value['CDS'])
    if m is not None:
        end = int(m.group(1))

    if start is None or end is None:
        m = re.findall(SEGMENT_PATTERN, value['CDS'])
        start = int(m[0][0])
        end = int(m[-1][-1])

    return start, end


def parse_sequence_cds(src_str: str):
    cds_data = defaultdict(dict)
    all_cds = []
    cds_flag = False
    sequence_id = None
    for line in src_str.split('\n'):
        line = line.strip()
        prefix = line[:2]
        fields = line.split()
        if prefix == 'ID':
            sequence_id = fields[1].split(';')[0].strip()
        elif prefix == 'FT':
            if fields[1] == 'CDS' and line.startswith("FT   CDS"):
                all_cds.append({})
                all_cds[-1]['CDS'] = ''
                cds_flag = True
            elif fields[1].startswith('/'):
                cds_flag = False
                if fields[1].startswith('/protein_id'):
                    cds_id = fields[1].split('=')[-1].strip('"')
                    all_cds[-1]['ID'] = cds_id

            if cds_flag:
                all_cds[-1]['CDS'] += fields[-1]

    assert sequence_id is not None
    grouped_cds = {sequence_id: []}
    for value in all_cds:
        try:
            if 'ID' not in value:
                continue
            cds_id = value['ID']
            start, end = parse_cds(value)
            grouped_cds[sequence_id].append((cds_id, start, end))
        except Exception:
            raise ValueError(f'Cannot parse {m}\n : {value}\n')

    cds_sorted = OrderedDict()
    cds_list = grouped_cds[sequence_id]
    if len(cds_list) > 1:
        cds_sorted[sequence_id] = {
            v[0]: (i,) + v[1:] for i, v in 
            enumerate(sorted(cds_list, key=lambda v: v[1]))
        }

    return cds_sorted


def parse_wgs_cds(src_str: str, dst_file: str = None):
    cds_data = defaultdict(dict)
    cds_flag = False
    cds_id = None
    for line in src_str.split('\n'):
        line = line.strip()
        prefix = line[:2]
        fields = line.split()
        if prefix == 'ID':
            assert not cds_flag
            cds_id = fields[1].split(';')[0].strip()
            cds_data[cds_id] = {}
        elif prefix == 'PA':
            assert cds_id is not None
            cds_data[cds_id]['PA'] = fields[1].strip(';')
        elif prefix == 'FT':
            assert cds_id is not None
            if fields[1] == 'CDS' and line.startswith("FT   CDS"):
                cds_flag = True
                cds_data[cds_id]['CDS'] = ''
            elif fields[1].startswith('/'):
                cds_flag = False

            if cds_flag:
                cds_data[cds_id]['CDS'] += fields[-1]
        elif prefix == '//':
            assert cds_id is not None
            cds_id = None

    grouped_cds = defaultdict(list)
    for cds_id, value in cds_data.items():
        try:
            set_id = value['PA']
            start, end = parse_cds(value)
            grouped_cds[set_id].append((cds_id, start, end))
        except Exception:
            raise ValueError(f'Cannot parse {m}\n{set_id}, {cds_id}: {value}\n')

    cds_sorted = OrderedDict()
    for set_id in sorted(grouped_cds.keys()):
        cds_list = grouped_cds[set_id]
        if len(cds_list) > 1:
            cds_sorted[set_id] = {
                v[0]: (i,) + v[1:] for i, v in 
                enumerate(sorted(cds_list, key=lambda v: v[1]))
            }

    return cds_sorted


def download_requests(src_file: str, dst_repo: str, max_ncpu: int = 20):
    with open(src_file) as fh:
        wgs_acs = json.load(fh)['uniprot_accessions']

    acs = []
    for ac in wgs_acs:
        path = ena_pairing.get_wgs_loci_path(dst_repo, ac)
        if path is None:
            acs.append(ac)

    ncpu = min(len(acs), max_ncpu)
    if ncpu > 1:
        urls = set()
        with mp.Pool(ncpu) as pool:
            results = []
            for i in range(ncpu):
                acs_ = [acs[j] for j in range(i, len(acs), ncpu)]
                result = pool.apply_async(
                    batch_get_urls, (i==0, acs_),
                )
                results.append(result)

            for result in results:
                urls.update(result.get())
    else:
        urls = batch_get_urls(True, acs)

    logger.warning(f'{len(urls)} to be downloaded')
    ncpu = min(len(urls), max_ncpu)
    if ncpu > 1:
        with mp.Pool(ncpu) as pool:
            results = []
            for i in range(ncpu):
                urls_ = [urls[j] for j in range(i, len(urls), ncpu)]
                result = pool.apply_async(
                    batch_download_and_parse, (i==0, urls_, dst_repo),
                )
                results.append(result)
            [ result.get() for result in results ]
    else:
        batch_download_and_parse(True, urls, dst_repo)

import numpy as np

from Bio import pairwise2
from alphafold.common import protein, residue_constants

def aatype_to_sequence(aatype):
    sequence = ''.join(
        residue_constants.restypes_with_x_and_gap[aa] for aa in aatype)
    return sequence


x_order = residue_constants.restype_order_with_x['X']
def sequence_to_aatype(sequence):
    aatype = np.array(
        [residue_constants.restype_order_with_x.get(w, x_order) for w in 
         sequence],
        dtype=np.int64)
    return aatype


def global_align(srcseq, tgtseq, gap_penalty=-2.):
    def _get_alnidx(seqA, seqB):
        srcidx, tgtidx = [], []
        ai, bj = 0, 0
        for a, b in zip(seqA, seqB):
            if a != '-' and b != '-':
                srcidx.append(ai)
                tgtidx.append(bj)
            if a != '-':
                ai += 1
            if b != '-':
                bj += 1
        srcidx = np.array(srcidx, dtype=np.int64)
        tgtidx = np.array(tgtidx, dtype=np.int64)
        return srcidx, tgtidx
    srcseq = ''.join([_ for _ in srcseq if _ != '-'])
    tgtseq = ''.join([_ for _ in tgtseq if _ != '-'])

    aln = pairwise2.align.globalms(srcseq, tgtseq, 2, -1, gap_penalty, -0.5)

    if len(aln) >= 1:
        aln = aln[0]
        srcidx, tgtidx = _get_alnidx(aln.seqA, aln.seqB)
        return aln, srcidx, tgtidx
    else:
        return None, None, None



from loguru import logger
import numpy as np
from astropy.table import Table
from pathlib import Path
from toltec_tone_utils import TlalocEtcDataStore
from contextlib import nullcontext



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--network", '-n', help='network to put output to', required=True, type=int)
    parser.add_argument("--n_tones", "-t", default=-1, help="number of tones", type=int)
    parser.add_argument("--head", '-p', help='number of tones to mask at head', default=0, type=int)
    parser.add_argument("--tail", '-q', help='number of tones to mask at tail', default=0, type=int)
    parser.add_argument("--start", '-s', help='index of tones to start', default=None, type=int)
    parser.add_argument("--end", '-e', help='index of tones to end', default=None, type=int)
    parser.add_argument("--invert", '-i', help='invert the mask', action='store_true')
    parser.add_argument("--etc_dir", help='the tlaloc etc dir', default='~/tlaloc/etc', type=Path)
    parser.add_argument("--dry_run", '-d', help='do dry run', action='store_true')

    option = parser.parse_args()

    tlaloc = TlalocEtcDataStore(option.etc_dir)

    logger.debug(f'working on {tlaloc=}')

    nw = option.network
    n_tones = option.n_tones

    if n_tones <= 0:
        n_tones = tlaloc.get_n_chans(nw)


    mask = np.ones((n_tones, )).astype(bool)
    if option.head > 0:
        start = option.start or 0
        mask[start:start + option.head] = 0
    if option.tail > 0:
        end = option.end or n_tones
        mask[end-option.tail:end] = 0
    if option.invert:
        mask = ~mask
    ctx = tlaloc.dry_run if option.dry_run else nullcontext
    with ctx():
        tlaloc.write_targ_mask_table(mask, nw=nw)
    logger.info(f'mask: {mask.sum()} / {len(mask)}')



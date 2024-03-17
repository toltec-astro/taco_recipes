

from loguru import logger
import numpy as np
from astropy.table import Table
from pathlib import Path
from toltec_tone_utils import TlalocEtcDataStore



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--network", '-n', help='network to put output to', required=True, type=int)
    parser.add_argument("--n_tones", "-t", default=-1, help="number of tones", type=int)
    parser.add_argument("--head", '-p', help='number of tones to mask at head', default=0, type=int)
    parser.add_argument("--tail", '-q', help='number of tones to mask at tail', default=0, type=int)
    parser.add_argument("--invert", '-i', help='invert the mask', action='store_true')
    parser.add_argument("--etc_dir", '-e', help='the tlaloc etc dir', default='~/tlaloc/etc', type=Path)

    option = parser.parse_args()

    tlaloc = TlalocEtcDataStore(option.etc_dir)

    logger.debug(f'working on {tlaloc=}')

    nw = option.network
    n_tones = option.n_tones

    if n_tones <= 0:
        n_tones = tlaloc.get_n_chans(nw)


    mask = np.ones((n_tones, )).astype(bool)
    if option.head > 0:
        mask[:option.head] = 0
    if option.tail > 0:
        mask[-option.tail:] = 0
    if option.invert:
        mask = ~mask

    tlaloc.write_targ_mask_table(mask, nw=nw)

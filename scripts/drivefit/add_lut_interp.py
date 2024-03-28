#!/usr/bin/env python

import argparse
from astropy.table import Table
import numpy as np
from pathlib import Path
from netCDF4 import Dataset


def _get_tone_amps(nc):
    if "Header.Toltec.ToneAmp" in nc.variables:
        ## new change 20240316
        toneAmps = nc.variables["Header.Toltec.ToneAmp"][:].data.T[:, 0]
    else:
        toneAmps = nc.variables["Header.Toltec.ToneAmps"][:].data
    return toneAmps


if __name__ == "__main__":
    import sys

    tune_path = sys.argv[1]
    tune = Dataset(tune_path)

    tone_amps_lut = _get_tone_amps(tune)

    tone_amps_lut[tone_amps_lut <= 0] = np.nan
    # print(tone_amps_lut)
    tone_amps = Table.read(sys.argv[2], names=['amp'], format='ascii.no_header')['amp']

    # print(tone_amps)

    tone_amps = tone_amps * tone_amps_lut
    # limit the range such that it does not increate the dyanmic range
    # lims = np.nanmin(tone_amps_lut), np.nanmax(tone_amps_lut)
    # tone_amps[tone_amps < lims[0]] = lims[0]
    # tone_amps[tone_amps > lims[1]] = lims[1]
    tone_amps_norm = np.max(tone_amps)
    print(f"{tone_amps_norm=}")
    tone_amps_norm_db = -20 * np.log10(tone_amps_norm)
    print(f'norm factor {tone_amps_norm} ({tone_amps_norm_db} db)')
    tone_amps = tone_amps / tone_amps_norm

    f_tones = tune.variables['Header.Toltec.ToneFreq'][0,:] + tune.variables['Header.Toltec.LoCenterFreq'][:].item()

    # i_sort = np.argsort(f_tones)

    # f_tones_sorted = f_tones[i_sort]
    # tone_amps_sorted = tone_amps[i_sort]
    print(f"interpret on tones {f_tones.shape}")

    current_targ_freqs = Table.read(sys.argv[3], format='ascii.ecsv')['f_out']
    current_tone_amps = []
    for i, f in enumerate(current_targ_freqs):
        j = np.argmin(np.abs(f_tones - f))
        # print(f'{i=} {j=} {f=} -> {f_tones[j]}')
        current_tone_amps.append(tone_amps[j])

    # print(current_tone_amps)

    # update

    Table([tone_amps]).write(sys.argv[2].replace('.txt', '.lut.txt'), format='ascii.no_header', overwrite=True)
    # get adrv
    with open(sys.argv[2].replace(".txt", ".adrv_ref.txt"), 'r') as fo:
        adrv_ref = float(fo.read())

    with open(sys.argv[2].replace('.txt', '.global_adrv.txt'), "w") as fo:
        global_adrv = tone_amps_norm_db + adrv_ref
        global_adrv = np.ceil(global_adrv * 4) / 4
        if global_adrv < 0:
            global_adrv_clip = 0
        elif global_adrv > 30:
            global_adrv_clip = 30
        else:
            global_adrv_clip = global_adrv
        print(f"{adrv_ref=} {global_adrv=} {global_adrv_clip=}")
        fo.write(f"{global_adrv_clip:.2f}")

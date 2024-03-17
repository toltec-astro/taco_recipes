
import sys
import numpy as np
from pathlib import Path
from astropy.table import Table
import astropy.units as u

from toltec_tone_power import get_tone_amplitudes_with_best_phases, transfer_func_lut


def make_random_phase(n_tones, with_cache=True):
    cache_file = Path(f"random_phase_{n_tones}.npy")
    if cache_file.exists() and with_cache:
        print(f"load phase from cache {cache_file}")
        phases = np.load(cache_file)
    else:
        print(f"generate random phase")
        phases = np.random.random(n_tones) * 2 * np.pi
        if with_cache:
            print(f"save phase to cache {cache_file}")
            np.save(cache_file, phases)
    return phases


def get_transfer_func(nw, flo_Hz):
    lut_files = {
            0: '00_LUTs/20240304/ampcor20240304_T01_N0_drive_Adrive0.csv',
            1: '00_LUTs/20240304/ampcor20240304_T01_N1_drive_Adrive0.csv',
        }
    lut_file = lut_files[nw]
    print(f"use lut_file: {lut_file}")
    return transfer_func_lut(flo_Hz, lut_file)



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tones", "-t", help="number of tones", required=True, type=int)
    # parser.add_argument("--regenerate_phase", "-r", help="regenerate phase cache", action='store_true')
    parser.add_argument("--tone_power_dbm_neg", '-q', help='tone power to the left of the LO', default='-90', type=float)
    parser.add_argument("--tone_power_dbm_pos", '-p', help='tone power to the right of the LO', default='-89', type=float)
    parser.add_argument("--network", '-n', help='network to put output to', default=0, type=int)
    parser.add_argument("--dry_run", '-d', help='do not actually write files', action='store_true')

    option = parser.parse_args()

    n_tones = option.n_tones

    ## generate tone freqs

    freq_lims = 450 << u.MHz , 900 << u.MHz
    freq_range = 200 << u.MHz
    lo_gap = 20 << u.MHz

    nw = option.network
    etcdir = Path(f"toltec{nw}")
    with etcdir.joinpath("last_centerlo.dat").open() as fo:
        lo_value = float(fo.read()) << u.Hz
        print(f"LO freq: {lo_value}")

    # do pos and net tones separately
    if n_tones >= 2:
        tone_freqs_p = np.linspace((lo_value + lo_gap).to_value(u.Hz), (lo_value + lo_gap + freq_range).to_value(u.Hz), n_tones // 2) << u.Hz
        tone_powers_p = np.full(len(tone_freqs_p), option.tone_power_dbm_pos)
        tone_freqs_n = np.linspace((lo_value - lo_gap - freq_range).to_value(u.Hz), (lo_value - lo_gap).to_value(u.Hz), n_tones // 2) << u.Hz
        tone_powers_n = np.full(len(tone_freqs_n), option.tone_power_dbm_neg)
        tone_freqs = np.hstack([tone_freqs_p, tone_freqs_n])
        tone_powers = np.hstack([tone_powers_p, tone_powers_n])
    else:
        tone_freqs = np.array([(lo_value + lo_gap + freq_range / 2).to_value(u.Hz)]) << u.Hz
        tone_powers = np.array([option.tone_power_dbm_pos])
    tone_comb_freqs = tone_freqs - lo_value

    transfer_func = get_transfer_func(nw=nw, flo_Hz = lo_value.to_value(u.Hz))
    tpt, required_atten = get_tone_amplitudes_with_best_phases(
            tone_comb_freqs.to_value(u.Hz), tone_powers, transfer_func=transfer_func)

    tpt.write(f'tone_props_{n_tones}_{nw}_{option.tone_power_dbm_pos:.0f}dbm_{option.tone_power_dbm_neg:.0f}dbm.ecsv', format='ascii.ecsv', overwrite=True)
    print(f"required global atten: {required_atten} dB")
    print(tpt)

    if option.dry_run:
        sys.exit()
    # make tlaloc compat outputs
    tone_amps = tpt['amplitude']
    phases = tpt['phase_rad']
    tbl = Table({"tone_freq": tone_freqs, "tone_amp": tone_amps, "tone_phase": phases})

    # sort fft
    tf = tbl['tone_freq'].quantity.to(u.Hz)
    cf = tf - lo_value
    max_ = 2.1 * np.max(cf)
    isort = sorted(
            range(len(cf)),
            key=lambda i: cf[i] + max_ if cf[i] < 0 else cf[i])
    tbl = tbl[isort]
    print(tbl)

    # prepare output
    targ_freqs_file = etcdir.joinpath("targ_freqs.dat")
    tbl_targ_freqs = Table()
    tf = tbl['tone_freq'].quantity.to(u.Hz)
    tbl_targ_freqs['f_centered'] = tf - lo_value
    tbl_targ_freqs['f_out'] = tf
    tbl_targ_freqs['f_in'] = tf
    for c in ['flag', 'fp', 'Qr', 'Qc', 'A', 'normI', 'normQ', 'slopeI', 'slopeQ', 'interceptI', 'interceptQ']:
        tbl_targ_freqs[c] = 0.
    tbl_targ_freqs.meta.update({
        "Header.Toltec.LoCenterFreq": lo_value.to_value(u.Hz),
        "Header.Toltec.ObsNum": 99,
        "Header.Toltec.SubObsNum": 0,
        "Header.Toltec.ScanNum": 0,
        "Header.Toltec.RoachIndex": nw,
        })
    tbl_targ_freqs.write(targ_freqs_file, format='ascii.ecsv', overwrite=True)

    targ_amps_file = etcdir.joinpath("default_targ_amps.dat")
    tbl_targ_amps = Table()
    tbl_targ_amps['targ_amp'] = tbl['tone_amp']
    tbl_targ_amps.write(targ_amps_file, format='ascii.no_header', overwrite=True)

    targ_phases_file = etcdir.joinpath("random_phases.dat")
    tbl_targ_phases = Table()
    tbl_targ_phases['targ_phase'] = tbl['tone_phase']
    tbl_targ_phases.write(targ_phases_file, format='ascii.no_header', overwrite=True)

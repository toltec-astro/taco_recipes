"""Calculation of TolTEC readout tone powers."""

from pydantic import BaseModel, Field
from loguru import logger

from astropy.table import Table
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import scipy.fftpack as fftpack
import numpy as np
import netCDF4


class DACConfig(BaseModel):
    """Constants of the DAC."""
    V_FS_Volt: float = Field(default=1.2, description="Full-scale voltage in Volt.")
    R_LOAD_Ohm: float = Field(default=50., description="Load resistance in Ohm.")
    FFT_size: int = Field(default=2**21, description="FFT size.")
    SAMPLE_FREQ_Hz: float = Field(default=512e6, description="Sample frequency in Hz.")
    BIT_DEPTH: int = Field(default=16, description="DAC bit depth.")
    # FIXED_ATTENUATION_dB: float = Field(default=35.0, description="Fixed cryogenic attenuation in dB")
    FIXED_ATTENUATION_dB: float = Field(default=0., description="Fixed cryogenic attenuation in dB")
    # TOTAL_POWER_dBm: float = Field(default=-8.6, description='Total power output, according to the Link Budget.')
    TOTAL_POWER_dBm: float = Field(default=-12.6, description='Total power output, according to the Link Budget.')


dac_config_default = DACConfig.model_validate({})


def transfer_func_null():
    """Return a function that adjust the power per tone.

    This does not do adjustment.
    """

    def func(tone_comb_freq_Hz, dac_config):
        return 0.
    return func


def transfer_func_lut(flo_Hz, lut_file):
    """Return a function that adjust the power per tone.

    This uses the LUT to adjust the power so the appear unifrom
    at the IF.
    """

    lut = Table.read(lut_file, format='ascii.no_header', delimiter=',',
            names=['f_Hz', 'amplitude']
            )
    interp_lut = interp1d(lut['f_Hz'], lut['amplitude'], kind='cubic')

    def func(tone_comb_freq_Hz, dac_config):
        amp = interp_lut(tone_comb_freq_Hz + flo_Hz)
        offset = 20 * np.log10(amp)
        # make sure the offsets adds up to 0
        offset = offset - np.sum(offset) / len(offset)
        return offset
    return func


def tone_powers_to_amplitunes(tone_comb_freqs_Hz, tone_powers_dBm, tone_phases_rad, transfer_func=None, drive_attens=None, dac_config=dac_config_default):
    """Calculate DAC amplitudes and expected KID tone powers.

    Parameters
    ----------
    tone_comb_freqs_Hz : array-like
        The tone comb frequencies in Hz (not LO scaled).
    tone_powers_dbm : array-like
        The tone powers requested at the KIDS in dBm.
    tone_phases_rad : array-like
        The tone phases in radians.
    transfer_func: callable, optional
        The transfer function to assume. This evaluates to dB offset.
    drive_attens_dB : array-like
        Additional driving attenuations for each tone.
    dac_config : DACConfig
        The DAC config.

    Returns
    -------
    tone_prop_table : astropy.table.Table
        A table contains the properties of the tones.
    atten_global_dB : float
        A overall attenuation value to set for all tones.
    """

    logger.debug(f"use DAC config: {dac_config}")

    # Generate a list of dictionaries of tone properties.
    tpt = tone_prop_table = Table()
    tpt['comb_freq_Hz'] = tone_comb_freqs_Hz
    tpt['power_dBm_requested'] = tone_powers_dBm
    tpt['phase_rad'] = tone_phases_rad
    if drive_attens is None:
        drive_attens = 0.
        # drive_attens_norm = 0
    else:
        # normalized offsets
        # drive_atten_norm = np.sum(drive_attens) / len(drive_attens)
        pass
    # drive_offsets = drive_attens - drive_attens_norm
    tpt['drive_atten_dB'] = drive_attens
    # tpt['drive_offset_dB'] = drive_offsets
    if transfer_func is None:
        transfer_func = transfer_func_no_lut()
    tpt['transfer_offset_dB'] = transfer_func(tone_comb_freqs_Hz, dac_config)

    # tpt['dac_atten_fixed_dB'] = dac_config.FIXED_ATTENUATION_dB

    # Calculate required amplitude for each tone, taking into account
    # the attenuations and various offsets.
    dac2det_atten = dac_config.FIXED_ATTENUATION_dB + tpt['drive_atten_dB'] + tpt['transfer_offset_dB']

    power_dBm_dac_req = tpt['power_dBm_requested'] + dac2det_atten
    power_mW_dac_req = 10 ** (power_dBm_dac_req / 10.)
    amps = tpt['amplitude'] = np.sqrt(2 * power_mW_dac_req / 1000 * dac_config.R_LOAD_Ohm)
    tpt['amplitude_scaled'] = amps / np.max(amps)
    amplitude_scale_offset = 20 * np.log10(np.max(amps))
    logger.debug(f"{amplitude_scale_offset=}")

    Pdac_total_mW_req = np.sum(power_mW_dac_req)
    Pdac_total_dBm_req = 10 * np.log10(Pdac_total_mW_req)

    # This shouldn't ever be the case, but let's add a total power
    # check to make sure we're not asking for more power than the DAC
    # can deliver.
    # max_power_watt = 28e-3  # 28mW (assumption by GW)
    Pdac_max_dBm = dac_config.TOTAL_POWER_dBm
    logger.debug(f"{Pdac_total_dBm_req=:3.2f} dBm {Pdac_max_dBm=:3.2f} dBm")
    if Pdac_total_dBm_req > Pdac_max_dBm:
        P_offset_factor = (Pdac_max_dBm - Pdac_total_dBm_req) / len(tpt)
        logger.warning(f"total requested power {Pdac_total_dBm_req:3.2f} dBm exceeds max DAC power: {Pdac_max_dBm:3.2f} dBm, offset power by {scale_factor} dB per tone.")
        # return the function with new power setting
        return tone_powers_to_amplitunes(tone_comb_freqs_Hz, tone_powers_dBm + P_offest_factor, tone_phases_rad, transfer_func=None, drive_attens=None, dac_config=dac_config_default)

    # A helper function for the waveform construction that follows
    def fft_bin_idx(freq):
        return int(round(freq / dac_config.SAMPLE_FREQ_Hz * dac_config.FFT_size))
        # return int(freq / dac_config.SAMPLE_FREQ_Hz * dac_config.FFT_size)

    # Generate the time domain waveforms (both I and Q)
    spec = np.zeros(dac_config.FFT_size, dtype=complex)

    for f, a, ph in tpt.iterrows("comb_freq_Hz", "amplitude", "phase_rad"):
        spec[fft_bin_idx(f)] = a * np.exp(1.j * ph)
    wave = np.fft.ifft(spec)
    waveform_I = wave.real
    waveform_Q = wave.imag

    V_FS_Volt = 2.8 * np.sqrt(2 * 10 ** (dac_config.TOTAL_POWER_dBm / 10) / 1000 * dac_config.R_LOAD_Ohm)
    logger.debug(f"use {V_FS_Volt=}")
    # Rescale the waveforms so that their peak matches the full-scale DAC voltage
    waveform_I *= V_FS_Volt / np.max(np.abs(waveform_I))
    waveform_Q *= V_FS_Volt / np.max(np.abs(waveform_Q))

    # Compute the FFT to get back to the frequency domain
    spectrum = fftpack.fft(waveform_I + 1.j * waveform_Q, n=dac_config.FFT_size)

    # Compute power of each output tone in dBm
    Pdac_total_mW_actual = 0.
    dac_powers_dBm_actual = []
    powers_dBm_actual = []
    for i, f in enumerate(tpt['comb_freq_Hz']):
        atten = dac2det_atten[i]
        idx = fft_bin_idx(f)
        a = np.abs(spectrum[idx]) / dac_config.FFT_size
        p_mW = (a ** 2) / (2 * dac_config.R_LOAD_Ohm) * 1000
        p_dbm = 10 * np.log10(p_mW)
        ap_dbm = p_dbm - atten
        Pdac_total_mW_actual += p_mW
        dac_powers_dBm_actual.append(p_dbm)
        powers_dBm_actual.append(ap_dbm)
    tpt['DAC_power_dBm_actual'] = dac_powers_dBm_actual
    tpt['power_dBm_actual'] = powers_dBm_actual
    Pdac_total_dBm_actual = 10 * np.log10(Pdac_total_mW_actual)
    logger.debug(f"{Pdac_total_dBm_actual=}")

    # Compute required drive attenuation
    power_differences = tpt["power_dBm_actual"] - tpt["power_dBm_requested"]
    required_attenuation = min(power_differences)

    # Round up to nearest 0.25 dB increment to ensure none of the
    # signals exceed the requested power
    required_attenuation = np.ceil(required_attenuation / 0.25) * 0.25

    # correction for the amps norm
    # required_attenuation_adjusted = required_attenuation + drive_atten_offset

    # add some metadata
    tpt.meta['dac_config'] = dac_config.model_dump()
    tpt.meta['Pdac_total_mW_requested'] = Pdac_total_mW_req
    tpt.meta['Pdac_total_dBm_requested'] = Pdac_total_dBm_req
    tpt.meta['Pdac_total_mW_actual'] = Pdac_total_mW_actual
    tpt.meta['Pdac_total_dBm_actual'] = Pdac_total_dBm_actual
    tpt.meta['atten_required'] = required_attenuation
    # tpt.meta['atten_required_adjusted'] = required_attenuation_adjusted

    pdiff = np.abs(tpt['power_dBm_requested'] - (tpt['power_dBm_actual']-required_attenuation))
    mpdiff = pdiff.mean()
    tpt.meta['mean_power_diff_dB'] = mpdiff
    return tpt, required_attenuation


def get_tone_amplitudes_with_best_phases(tone_comb_freqs_Hz, tone_powers_dBm, n_rands=20, random_seed=None, **kwargs):
    # Let's run this for N different sets of random phases and then choose
    # the phases that result in the lowest request-actual average.
    rng = np.random.default_rng(random_seed)
    best_phases = None
    best_match = np.inf
    for i in range(n_rands):
        ph = rng.uniform(0., 2.*np.pi, size=len(tone_comb_freqs_Hz))
        tpt, _ = tone_powers_to_amplitunes(
            tone_comb_freqs_Hz=tone_comb_freqs_Hz, tone_powers_dBm=tone_powers_dBm, tone_phases_rad=ph, **kwargs)
        m = tpt.meta['mean_power_diff_dB']
        if m < best_match:
            best_match = m
            best_phases = ph
        logger.debug(f"mean power difference: {m} dB")
    logger.debug(f"minimum power difference: {best_match} dB")
    # run again with the best phases
    return tone_powers_to_amplitunes(
        tone_comb_freqs_Hz=tone_comb_freqs_Hz, tone_powers_dBm=tone_powers_dBm, tone_phases_rad=best_phases, **kwargs)


   

def _test_gw():
    # A test case.
    # Lets try this with some actual tones
    ncFile = 'tolteca_test_data/data_lmt/toltec/tcs/toltec0/toltec0_110246_000_0001_2023_06_03_10_27_11_targsweep.nc'
    nc = netCDF4.Dataset(ncFile)
    tf = nc.variables['Header.Toltec.ToneFreq'][:].data[0]
    lo = float(nc.variables['Header.Toltec.LoCenterFreq'][:].data)
    am = nc.variables['Header.Toltec.ToneAmps'][:].data
    nc.close()

    # Read in toltec0's LUT
    lut = np.loadtxt('amps_correction_lut.csv', delimiter=',')
    flut = lut[:, 0]-lo
    alut = 1./lut[:, 1]
    alut /= alut.max()
    alut = 20.*np.log(alut)
    drive_attenuation = interp1d(flut, -alut, kind='linear')
    plt.figure(3)
    plt.clf()
    plt.plot(flut, alut, '+')

    # experiment with different powers and phases
    rp = -90 + np.random.uniform(-2., 2., size=len(tf))
    # p = np.linspace(0., 2.*np.pi, len(tf))
    # ph = np.random.choice(p, size=len(tf), replace=False)
    da = drive_attenuation(tf)

    tpt, required_attenuation = get_tone_amplitudes_with_best_phases(
        tf, rp, drive_attens=da)

    # Let's see what we got
    freq = []
    request = []
    actual = []
    amps = []
    for f, p_req, p_act, a in tpt.iterrows("comb_freq_Hz", "power_dBm_requested", "power_dBm_actual", "amplitude"):
        freq.append(f*1.e-6)
        request.append(p_req)
        actual.append(p_act)
        amps.append(a)
    plt.figure(1)
    plt.clf()
    plt.plot(freq, request, '.', label='Requested Power')
    plt.plot(freq, actual-required_attenuation, '.', label='Actual Power')
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('Power [dBm]')
    plt.legend()

    print(f"Required attenuation: {required_attenuation} dB")
    plt.show()


def make_tlaloc_tone_test_files(tone_comb_freqs_Hz, tone_powers_dBm, output_dir='OUTPUT'):
    pass


if __name__ == "__main__":
    _test_gw()

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
    TOTAL_POWER_dBm: float = Field(default=-8.6, description='Total power output, according to Eric.')


dac_config_default = DACConfig.model_validate({})


def transfer_func_no_lut():
    """Return a function that gives the power per tone."""

    def func(tone_comb_freq_Hz, dac_config):
        ptot = dac_config.TOTAL_POWER_dBm
        return ptot - 10 * np.log10(len(tone_comb_freq_Hz))
    return func


def transfer_func_lut(flo_Hz, lut_file):
    """Return a function that gives the power per tone."""

    lut = Table.read(lut_file, format='ascii.no_header', delimiter=',',
            names=['f_Hz', 'amplitude']
            )
    interp_lut = interp1d(lut['f_Hz'], lut['amplitude'], kind='cubic')

    def func(tone_comb_freq_Hz, dac_config):
        ptot = dac_config.TOTAL_POWER_dBm
        amp = interp_lut(tone_comb_freq_Hz + flo_Hz)
        offset = 20 * np.log10(amp)
        # make sure the offsets adds up to 0
        offset = offset - np.sum(offset) / len(offset)
        # import pdb
        # pdb.set_trace()
        return ptot - 10 * np.log10(len(tone_comb_freq_Hz)) + offset
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
        The transfer function to assume. This evaluates to attenuation in dB unit.
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
        drive_attens = 0
    tpt['drive_atten_dB'] = drive_attens
    if transfer_func is None:
        transfer_func = transfer_func_no_lut()
    tpt['transfer_atten_dB'] = transfer_func(tone_comb_freqs_Hz, dac_config)

    # tpt['dac_atten_fixed_dB'] = dac_config.FIXED_ATTENUATION_dB

    # Calculate required amplitude for each tone, taking into account
    # the attenuations.
    total_dac_atten = dac_config.FIXED_ATTENUATION_dB + tpt['drive_atten_dB'] + tpt['transfer_atten_dB']
    power_dBm_req = tpt['power_dBm_requested'] + total_dac_atten
    power_watt_req = 10 ** (power_dBm_req / 10.) / 1000.
    amps = tpt['amplitude'] = np.sqrt(2 * power_watt_req * dac_config.R_LOAD_Ohm)
    tpt['amplitude_norm'] = amps / np.max(amps)
    drive_atten_offset = 20 * np.log10(np.max(amps))

    Pdac_total_watt_req = np.sum(power_watt_req)
    Pdac_total_dBm_req = 10 * np.log10(Pdac_total_watt_req * 1000)

    # This shouldn't ever be the case, but let's add a total power
    # check to make sure we're not asking for more power than the DAC
    # can deliver.
    max_power_watt = 28e-3  # 28mW (assumption by GW)
    if Pdac_total_watt_req > max_power_watt:
        scale_factor = max_power_watt / Pdac_total_watt_req
        logger.info(f"total requested power {Pdac_total_watt_req:3.2f} W exceeds peak DAC power: {max_power_watt:3.2f}, rescaling amplitudes by {scale_factor}")
        tpt['amplitude'] = tpt['amplitude'] * scale_factor

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

    # Rescale the waveforms so that their peak matches the full-scale DAC voltage
    waveform_I *= dac_config.V_FS_Volt / np.max(np.abs(waveform_I))
    waveform_Q *= dac_config.V_FS_Volt / np.max(np.abs(waveform_Q))

    # Compute the FFT to get back to the frequency domain
    spectrum = fftpack.fft(waveform_I + 1.j * waveform_Q, n=dac_config.FFT_size)

    # Compute power of each output tone in dBm
    Pdac_total_watt_actual = 0.
    dac_powers_dBm_actual = []
    powers_dBm_actual = []
    for i, f in enumerate(tpt['comb_freq_Hz']):
        tda = total_dac_atten[i]
        idx = fft_bin_idx(f)
        a = np.abs(spectrum[idx]) / dac_config.FFT_size
        p_watt = (a ** 2) / (2 * dac_config.R_LOAD_Ohm)
        p_dbm = 10 * np.log10(p_watt * 1000)
        ap_dbm = p_dbm - tda
        Pdac_total_watt_actual += p_watt
        dac_powers_dBm_actual.append(p_dbm)
        powers_dBm_actual.append(ap_dbm)
    tpt['DAC_power_dBm_actual'] = dac_powers_dBm_actual
    tpt['power_dBm_actual'] = powers_dBm_actual
    Pdac_total_dBm_actual = 10 * np.log10(Pdac_total_watt_actual * 1000)


    # Compute required drive attenuation
    power_differences = tpt["power_dBm_actual"] - tpt["power_dBm_requested"]
    required_attenuation = min(power_differences)

    # Round up to nearest 0.25 dB increment to ensure none of the
    # signals exceed the requested power
    required_attenuation = np.ceil(required_attenuation / 0.25) * 0.25

    # correction for the amps norm
    required_attenuation_adjusted = required_attenuation + drive_atten_offset

    # add some metadata
    tpt.meta['dac_config'] = dac_config.model_dump()
    tpt.meta['Pdac_total_watt_reqested'] = Pdac_total_watt_req
    tpt.meta['Pdac_total_dBm_reqested'] = Pdac_total_dBm_req
    tpt.meta['Pdac_total_watt_actual'] = Pdac_total_watt_actual
    tpt.meta['Pdac_total_dBm_actual'] = Pdac_total_dBm_actual
    tpt.meta['atten_required'] = required_attenuation
    tpt.meta['atten_required_adjusted'] = required_attenuation_adjusted

    pdiff = np.abs(tpt['power_dBm_requested'] - (tpt['power_dBm_actual']-required_attenuation))
    mpdiff = pdiff.mean()
    tpt.meta['mean_power_diff_dB'] = mpdiff
    return tpt, required_attenuation_adjusted


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

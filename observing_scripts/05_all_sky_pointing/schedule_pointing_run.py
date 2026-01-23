import csv, ast
import numpy as np
from collections import OrderedDict

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

import matplotlib.pyplot as plt

def parse_sma_sources(filepath):
    """
    Read smaSources.csv, skip the first 4 comment/header lines,
    and return a list of dicts with the raw columns.
    """
    cols = ['commonName','name','ra','dec','freq','date','obs','flux']
    sources = []
    with open(filepath, 'r') as f:
        for _ in range(4):  # skip comments + header
            next(f)
        reader = csv.DictReader(f, fieldnames=cols)
        for row in reader:
            try:
                row['freq'] = ast.literal_eval(row['freq'])
                row['flux'] = ast.literal_eval(row['flux'])
            except Exception:
                continue
            sources.append(row)
    return sources


def angle_diff(a, b):
    """Smallest difference between angles a and b in deg"""
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def plan_pointing_run(
    csv_path,
    start_time: Time = None,
    elev_bin=(60.0, 80.0),
    min_flux=0.3,
    max_sources=12,
    location: EarthLocation = None,
):
    """
    Plan up to `max_sources` pointings within an elevation bin,
    sampling uniformly in azimuth and enforcing strictly decreasing elevation.
    """
    # Set defaults
    if start_time is None:
        start_time = Time.now()
    if location is None:
        location = EarthLocation.from_geodetic(
            lon=-97.31481605209875 * u.deg,
            lat= 18.98578175043638 * u.deg,
            height=4500 * u.m
        )

    # Load and initial filtering
    raw = parse_sma_sources(csv_path)
    frame0 = AltAz(obstime=start_time, location=location)
    prefilter = []
    for src in raw:
        if '1mm' not in src['freq']:
            continue
        idx = src['freq'].index('1mm')
        try:
            fval = float(src['flux'][idx].split('±')[0])
        except Exception:
            continue
        if fval < min_flux:
            continue
        coord = SkyCoord(src['ra'], src['dec'], unit=(u.hourangle, u.deg), frame='icrs')
        aa0 = coord.transform_to(frame0)
        alt0, az0 = aa0.alt.deg, aa0.az.deg
        if elev_bin[0] <= alt0 < elev_bin[1]:
            prefilter.append({'name': src['name'],
                              'coord': coord,
                              'flux': fval,
                              'alt0': alt0,
                              'az0': az0})
    if not prefilter:
        return OrderedDict()

    # Uniform az selection
    edges = np.linspace(0, 360, max_sources + 1)
    selected = []
    for i in range(max_sources):
        lo, hi = edges[i], edges[i+1]
        if i == max_sources - 1:
            in_bin = [c for c in prefilter if c['az0'] >= lo or c['az0'] < hi % 360]
        else:
            in_bin = [c for c in prefilter if lo <= c['az0'] < hi]
        if in_bin:
            best = max(in_bin, key=lambda c: c['alt0'])
            selected.append(best)
    if len(selected) < max_sources:
        remaining = [c for c in sorted(prefilter, key=lambda x: x['alt0'], reverse=True)
                     if c not in selected]
        for c in remaining:
            selected.append(c)
            if len(selected) >= max_sources:
                break

    # Schedule with monotonic altitude
    slew_rate = 1.0   # deg/s
    obs_dur = 120.0   # s
    plan = OrderedDict()
    t_now = start_time
    prev_coord = None
    prev_alt_obs = elev_bin[1]
    remaining = selected.copy()

    while remaining and len(plan) < max_sources:
        best_item = None
        best_info = None
        best_alt1 = -np.inf
        for c in remaining:
            coord = c['coord']
            if prev_coord is None:
                slew_sec = 0.0
            else:
                aa_prev = prev_coord.transform_to(AltAz(obstime=t_now, location=location))
                aa_now = coord.transform_to(AltAz(obstime=t_now, location=location))
                d_az = angle_diff(aa_now.az.deg, aa_prev.az.deg)
                d_alt = abs(aa_now.alt.deg - aa_prev.alt.deg)
                slew_sec = max(d_az, d_alt) / slew_rate
            t_start = t_now + slew_sec * u.second
            aa_obs = coord.transform_to(AltAz(obstime=t_start, location=location))
            alt1, az1 = aa_obs.alt.deg, aa_obs.az.deg
            if not (elev_bin[0] <= alt1 < elev_bin[1]):
                continue
            if alt1 > prev_alt_obs:
                continue
            if alt1 > best_alt1:
                best_alt1 = alt1
                best_item = c
                best_info = {'slew_sec': slew_sec, 't_start': t_start, 'alt1': alt1, 'az1': az1}
        if best_item is None:
            break
        t_end = best_info['t_start'] + obs_dur * u.second
        plan[best_item['name']] = {
            'ra': best_item['coord'].ra.to_string(unit=u.hour, sep=':'),
            'dec': best_item['coord'].dec.to_string(unit=u.deg, sep=':'),
            'flux_Jy': best_item['flux'],
            'slew_time_sec': best_info['slew_sec'],
            'obs_start_ISO': best_info['t_start'].iso,
            'obs_end_ISO': t_end.iso,
            'alt_at_obs': best_info['alt1'],
            'az_at_obs': best_info['az1']}
        t_now = t_end
        prev_coord = best_item['coord']
        prev_alt_obs = best_info['alt1']
        remaining.remove(best_item)
    return plan


def plot_pointing_sequence(plan):
    """
    Plot azimuth vs elevation (unwrapped) and show slew/on-source times.
    """
    raw_az = [v['az_at_obs'] for v in plan.values()]
    el = [v['alt_at_obs'] for v in plan.values()]

    # unwrap azimuth
    az_rad = np.radians(raw_az)
    az_unwrapped = np.unwrap(az_rad)
    az_plot = np.degrees(az_unwrapped)

    total_slew = sum(v['slew_time_sec'] for v in plan.values())
    obs_secs = [(Time(v['obs_end_ISO']) - Time(v['obs_start_ISO'])).sec for v in plan.values()]
    total_on = sum(obs_secs)

    fig, ax = plt.subplots()
    ax.plot(az_plot, el, marker='o', linestyle='-')
    if len(az_plot) > 0 and len(el) > 0:
        ax.scatter(az_plot[0], el[0], color='green', s=100, label='First source')
    ax.set_xlabel('Azimuth (°)')
    ax.set_ylabel('Elevation (°)')
    ax.set_title('Pointing Sequence: Az vs Elev')
    # display ticks modulo 360
    ticks = ax.get_xticks()
    ax.set_xticklabels([f"{t % 360:.0f}" for t in ticks])

    info = f"Total slew time: {total_slew/60:.1f} min\nTotal on-source: {total_on/60:.1f} min"
    ax.text(0.02, 0.98, info, transform=ax.transAxes,
            va='top', fontsize=10, bbox=dict(boxstyle='round', alpha=0.3))
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from pprint import pprint
    # Example start date/time
    startDate = "2025-07-03"
    startTimeUTC = "16:24"
    iso_str = f"{startDate}T{startTimeUTC}:00"
    start_time_obj = Time(iso_str, format='isot', scale='utc')

    schedule = plan_pointing_run(
        'smaSources.csv',
        start_time=start_time_obj,
        elev_bin=(62.5, 80.0),
        min_flux=0.3,
        max_sources=12
    )
    pprint(schedule)
    plot_pointing_sequence(schedule)

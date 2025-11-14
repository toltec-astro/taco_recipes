from schedule_pointing_run import plan_pointing_run
from astropy.table import Table
from astroplan import Observer
from pathlib import Path
from astropy import coordinates as coord
import astropy.units as u
from astropy.time import Time
from pytz import timezone
import tqdm

ot_script_template = {
    "header": """
#ObservingScript -Name  "{script_name}" -Author "toltec"
Dcs Dcs;  Dcs  -Backend null -Receiver null -SubObsMode 0;Shmem  -Backend null -ObsPgmData null -Receiver null
ProposalInfo Dcs;  Dcs  -ProjectId 2025_Tol_SRO
ObsGoal Dcs;  Dcs  -ObsGoal Pointing
Toltec
ToltecBackend -RoachCmd  config -Remote Corba -HostPort clipy:1094 -TcpHostPort clipy:8990
TelescopeBackend -IncludeSignal  0
ToltecRoachSampleFreqSet ToltecBackend;  ToltecBackend  -RoachArg[0] 122 -RoachCmd samplefreq_set
    """,
    "body": """
Source Source;  Source  -BaselineList [] -CoordSys Eq -DecProperMotionCor 0 -Dec[0] {dec} -Dec[1] {dec} -El[0] 0.000000 -El[1] 0.000000 -EphemerisTrackOn 0 -Epoch 2000.0 -GoToZenith 1 -L[0] 0.0 -L[1] 0.0 -LineList [] -Planet None -RaProperMotionCor 0 -Ra[0] {ra} -Ra[1] {ra} -SourceName {name} -VelSys Lsr -Velocity 0.000000 -Vmag 0.0
Lissajous -ExecMode  0 -RotateWithElevation 0 -TScan 60 -ScanRate 50 -TRef 0 -RefPeriod 0 -TCal 0 -CalPeriod 0 -TunePeriod 0 -XLength 2 -YLength 2 -XOmega 5 -YOmega 4 -XDelta 45 -XLengthMinor 0 -YLengthMinor 0 -XDeltaMinor 0
"""
}

def _plan_pointing_run(**kwargs):
    return plan_pointing_run("smaSources.csv", **kwargs)


def _make_time_str(t):
    return f"{t.datetime.strftime('%Y-%m-%dT%H-%M-%S')}"


def make_ot_script(label, start_time, output_dir=Path("generated"), **plan_kwargs):
    plan = _plan_pointing_run(start_time=start_time, **plan_kwargs)

    total_slew = sum(v['slew_time_sec'] for v in plan.values())
    obs_durations = [(Time(v['obs_end_ISO']) - Time(v['obs_start_ISO'])).sec
                     for v in plan.values()]
    total_on = sum(obs_durations)
    total_time = ((total_slew + total_on) << u.s).to(u.min)

    total_time_str = f"{int(total_time.to_value(u.min))}min"

    start_time_str = _make_time_str(start_time)

    script_name = f"all_sky_pointing_{label}_{start_time_str}_{total_time_str}.lmtot"
    script_dir = output_dir.joinpath(start_time_str)

    ot_script = ot_script_template["header"].format(script_name=script_name).strip()
    n_sources = len(plan)
    for name, info in plan.items():
        ra = info["ra"]
        dec = info["dec"]
        ot_script += "\n" + ot_script_template["body"].format(
            name=name, ra=ra, dec=dec,
        ).strip()
    if not script_dir.exists():
        script_dir.mkdir(parents=True)
    with script_dir.joinpath(script_name).open("w") as fo:
        fo.write(ot_script)
    return locals()


lmt_info = {
    'instru': 'lmt',
    'name': 'LMT',
    'name_long': "Large Millimeter Telescope",
    'location': coord.EarthLocation.from_geodetic(**{
        'lon': '-97d18m52.6s',
        'lat': '+18d59m10s',
        'height': 4640 << u.m,
        }),
    'timezone_local': timezone('America/Mexico_City'),
    }

lmt_observer = Observer(
        name=lmt_info['name_long'],
        location=lmt_info['location'],
        timezone=lmt_info['timezone_local'],
        )


if __name__ == "__main__":

    time_gap = 10 << u.min
    elev_bins = [
        {
            "label": "elev80",
            "bin": (62.5, 80),
        },
        {
            "label": "elev62",
            "bin": (45, 62.5),
        },
        {
            "label": "elev45",
            "bin": (27.5, 45),
        },
        {
            "label": "elev27",
            "bin": (10, 27.5),
        },
    ]
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--day_offset", type=int, default=0)
    option = parser.parse_args()
    day_offset = option.day_offset

    time_now = Time.now() + (day_offset << u.day)
    now_is_night = lmt_observer.is_night(time_now)

    if now_is_night:
        sunset_which = "previous"
    else:
        sunset_which = "next"
    sunset_time = lmt_observer.sun_set_time(time_now, which=sunset_which)
    sunrise_time = lmt_observer.sun_rise_time(sunset_time, which="next")
    print(f"Current time: {time_now.isot} ")
    print(f"Is night at LMT? {now_is_night} ")
    print(f"Generate scripts between {sunset_time.isot} and {sunrise_time.isot}")
    t0 = Time(sunset_time.datetime.replace(minute=0, second=0, microsecond=0))
    t = t0
    start_times = []
    while t < sunrise_time:
        start_times.append(t)
        t = t + time_gap
    print("Start times:\n{}".format("\n".join(t.isot for t in start_times)))

    summary = []
    output_dir = Path("generated")
    for elev_bin in tqdm.tqdm(elev_bins):
        for start_time in tqdm.tqdm(start_times):
            ctx = make_ot_script(
                elev_bin["label"], start_time,
                output_dir=output_dir,
                elev_bin=elev_bin["bin"],
                min_flux=0.3,
                max_sources=10,
                location=lmt_observer.location,
            )
            summary.append(elev_bin | {
                "start_time": start_time,
                "total_time": ctx["total_time"],
                "n_sources": ctx["n_sources"],
            })
    summary = Table(summary)
    print(summary)
    summary.write(output_dir.joinpath(f"summary_{_make_time_str(start_times[0])}.ecsv"), overwrite=True)


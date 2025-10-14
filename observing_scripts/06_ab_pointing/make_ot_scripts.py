from astropy.table import Table
from astroplan import Observer
from pathlib import Path
from astropy import coordinates as coord
import astropy.units as u
from astropy.time import Time
from pytz import timezone
from schedule_pointing_run import parse_sma_sources

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


def _make_time_str(t):
    return f"{t.datetime.strftime('%Y-%m-%dT%H-%M-%S')}"


def _get_source_coord(tbl, name):
    if tbl is None:
        raise ValueError("invalid source table")
    entry = tbl[tbl["name"] == name][0]
    return coord.SkyCoord(entry["ra"], entry["dec"], unit=(u.hourangle, u.deg), frame='icrs')


def make_ot_script(config, output_dir=Path("generated"), tbl_sources=None):
    label = config["name"]
    n_repeats = config["n_repeats"]
    script_name = f"ab_pointing_{label}.lmtot"
    script_dir = output_dir

    ot_script = ot_script_template["header"].format(script_name=script_name).strip()
    sources = config["sources"]
    n_sources = len(sources)
    for _ in range(n_repeats):
        for info in sources:
            name = info["name"]
            coo = _get_source_coord(tbl_sources, name)
            ot_script += "\n" + ot_script_template["body"].format(
                    name=name, ra=coo.ra.to_string(unit=u.hourangle, sep=":"),
                    dec=coo.dec.to_string(unit=u.deg, sep=":"),
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

    import argparse
    from astropy.io.misc import yaml

    time_now = Time.now()
    now_is_night = lmt_observer.is_night(time_now)

    print(f"Current time: {time_now.isot} ")
    print(f"Is night at LMT? {now_is_night} ")

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=Path)
    option = parser.parse_args()

    config_file = option.config_file

    with open(config_file) as fo:
        config = yaml.load(fo)
        print(f'load config: {config["name"]}')

    output_dir = Path("generated")

    tbl_sources = Table(parse_sma_sources("smaSources.csv"))
    print(f"loaded {len(tbl_sources)} source info")
    # print(tbl_sources)
    ctx = make_ot_script(config=config, output_dir=output_dir, tbl_sources=tbl_sources)


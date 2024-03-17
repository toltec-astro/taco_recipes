
import sys
import functools
import numpy as np
from pathlib import Path
from astropy.table import Table, QTable, Column
import astropy.units as u
from contextlib import contextmanager
from functools import cached_property
from loguru import logger

from astropy.utils.decorators import classproperty
import astropy.units as u

from toltec_tone_power2 import get_tone_amplitudes_with_best_phases, transfer_func_lut


class RoachToneProps:
    """A base class to manage tone properties in ROACH."""

    def __init__(self, table, **kwargs):
        self._table = self._validate_table(table, **kwargs)
        self._enable_mask = True

    @classproperty
    def _table_is_validated_key(cls):
        return  f"_{cls.__name__}_table_validated"

    @classmethod
    def _validate_table(cls, table, flo=None, flo_key='flo'):

        if table.meta.get(cls._table_is_validated_key, False):
            return table

        t = table
        cns = t.colnames
        tpt = QTable()
        meta = tpt.meta
        meta.update(t.meta)

        # update flo
        if flo is None:
            meta['flo'] = t.meta.get(flo_key, None)
        else:
            meta['flo'] = flo
        # make sure flo is in correct unit
        flo = meta['flo'] = (
                meta['flo'] << u.Hz if meta['flo'] is not None else None
                )
        # check consistency between f_comb and f_tone if exists
        if 'f_comb' in cns and 'f_tone' in cns:
            # update flo
            flos = np.unique(
                    (t['f_tone'] << u.Hz) - (t['f_comb'] << u.Hz)
                    )
            if len(flos) != 1:
                raise ValueError("inconsistent LO frequencies in data.")
            if flo is None:
                flo = meta[flo_key] = flos[0]
            elif (flo << u.Hz) != flos[0]:
                raise ValueError("inconsistent LO frequency.")
            else:
                pass
            # update tpt
            for k in ('f_comb', 'f_tone'):
                tpt[k] = t[k] << u.Hz
        elif 'f_comb' in cns:
            tpt['f_comb'] = t['f_comb'] << u.Hz
            if flo is not None:
                tpt['f_tone'] = tpt['f_comb'] + flo
        elif 'f_tone' in cns and flo is not None:
            tpt['f_comb'] = (t['f_tone'] << u.Hz) - flo
            tpt['f_tone'] = (t['f_tone'] << u.Hz)
        else:
            raise ValueError("no tone comb frequency found.")
        for k, unit, dtype in [
                ("amp_tone", None, None),
                ("phase_tone", u.rad, None),
                ("mask_tone", None, bool),
                ]:
            v = t[k]
            if dtype is not None:
                v = v.astype(dtype)
            if unit is not None:
                v = v << unit
            tpt[k] = t[k]
        meta[cls._table_is_validated_key] = True
        return tpt

    @property
    def meta(self):
        return self.table.meta

    @property
    def table(self):
        return self._table

    @cached_property
    def table_masked(self):
        return self._table[self.mask]

    @contextmanager
    def no_mask(self):
        self._enable_mask = False
        yield self
        self._enable_mask = True
        return

    def _get_data(self, key):
        if self._eanble_mask:
            return self.table_masked[k]
        return self.table[k]

    @property
    def mask(self):
        return self.table['mask_tone'].astype(bool)

    @property
    def f_combs(self):
        return self._get_data('f_comb')

    @property
    def f_tones(self):
        return self._get_data('f_tone')

    @property
    def amps(self):
        return self._get_data('amp_tone')

    @property
    def phases(self):
        return self._get_data('phase_tone')

    @property
    def n_chans(self):
        return len(self.table)

    @property
    def n_tones(self):
        return len(self.table_masked)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_tones}/{self.n_chans})"

    def __getitem__(self, *args):
        return self.__class__(self.table.__getitem__(*args))


class TlalocEtcDataStore:
    """The data files managed by tlaloc."""

    def __init__(self, path='~/tlaloc/etc'):
        etcdir = Path(path).expanduser().resolve()
        if not etcdir.exists():
            raise ValueError(f'etc directory does not exist {etcdir}')
        self._etcdir = etcdir
        self._subdir_by_nw = self._get_subdir_by_nw(etcdir)
        self._file_suffix = None
        self._do_dry_run = False

    @classmethod
    def _get_subdir_by_nw(cls, etcdir):
        result = {}
        for nw in range(13):
            path = etcdir.joinpath(f'toltec{nw}')
            if path.exists():
                result[nw] = path
        return result

    @property
    def rootpath(self):
        return self._etcdir

    @property
    def nws(self):
        return list(self._subdir_by_nw.keys())

    def get_path(self, nw):
        return self._subdir_by_nw[nw]

    _item_name_map = {
        "targ_freqs": "targ_freqs.dat",
        "targ_amps": "default_targ_amps.dat",
        "targ_phases": "random_phases.dat",
        "targ_mask": "default_targ_masks.dat",
        "lut": "amps_correction_lut.csv",
        "last_flo": "last_centerlo.dat",
        }

    _item_tone_prop_map = {
        "targ_amps": "amp_tone",
        "targ_phases": "phase_tone",
        "targ_mask": "mask_tone",
        }

    _kids_model_colnames = [
            'f_in', 'flag',
            'fp', 'Qr', 'Qc', 'A',
            'normI', 'normQ', 'slopeI', 'slopeQ',
            'interceptI', 'interceptQ'
        ]

    @contextmanager
    def set_file_suffix(self, suffix):
        self._file_suffix = suffix
        yield self
        self._file_suffix = None
        return

    @contextmanager
    def dry_run(self):
        self._do_dry_run = True
        yield self
        self._do_dry_run = False
        return

    def get_item_path(self, nw, item):
        if item not in self._item_name_map:
            raise ValueError(f"invalid item name {item}")
        name = self._item_name_map[item]
        s = self._file_suffix
        if s:
            s = s.strip('.')
            name = f"{name}.{s}"
        return self.get_path(nw).joinpath(name)

    @classmethod
    def _mark_validated(cls, tbl, key):
        tbl.meta[f"_{cls.__name__}_{key}_validated"] = True

    @classmethod
    def _check_validated(cls, tbl, key):
        if not hasattr(tbl, 'meta'):
            return False
        return tbl.meta.get(f"_{cls.__name__}_{key}_validated", False)

    @classmethod
    def _make_simple_data_table(cls, data, item, nw=None):
        if item not in cls._item_tone_prop_map:
            raise ValueError('invalid item')
        col = cls._item_tone_prop_map[item]
        if isinstance(data, RoachToneProps):
            tbl = data.table
        elif isinstance(data, Table):
            tbl = data
        else:
            data = np.array(data)
            if data.ndim != 1:
                raise ValueError('invalid data shape.')
            tbl = Table()
            tbl[col] = data
        if col not in tbl.colnames:
            raise ValueError("no data column found.")

        if 'nw' not in tbl.colnames:
            # try to infer nw
            if nw is None:
                # try get it from the table meta
                _nw = None
                for k in ["nw", "roach"]:
                    v = t.meta.get(k, None)
                    if v is not None:
                        _nw = v
                        break
                if _nw is None:
                    raise ValueError("unable to find network id.")
                nw = _nw
        elif nw is not None:
            # nw will be used to filter the table
            tbl = tbl[tbl["nw"] == nw]
        else:
            # the table may contain multiple nws.
            pass

        def _use_unit(c, unit):
            return (c << unit).to(unit)

        tbl_out = Table()
        tbl_out.meta.update({
            "nw": nw,
            })

        unit, dtype = {
                "targ_amps": (None, None),
                "targ_phases": (u.rad, None),
                "targ_mask": (None, int),
                }[item]
        v = tbl[col]
        if dtype is not None:
            v = v.astype(dtype)
        if unit is not None:
            v = v << unit
        tbl_out[item] = v
        cls._mark_validated(tbl_out, item)
        return tbl_out

    @classmethod
    def _gen_data_if_not_given(cls, data, n_chans, gen_none, gen_scalar):
        if isinstance(data, RoachToneProps):
            return data
        if data is None:
            if n_chans is None:
                raise ValueError('n_chans required when data not provided.')
            data = get_none(n_chan)
        else:
            _data = np.array(data)
            if _data.ndim == 0:
                if n_chans is None:
                    raise ValueError('n_chans required when data is scalar.')
                data = gen_scalar(_data.item(), n_chan)
        return data

    @classmethod
    def make_targ_amps_table(cls, data=None, nw=None, n_chans=None):
        data = cls._gen_data_if_not_given(
                data=data, n_chans=n_chans,
                gen_none=lambda n: np.ones((n, ), dtype=float),
                gen_scalar=lambda d, n: np.full((n, ), d)
                )
        return cls._make_simple_data_table(data, "targ_amps", nw=nw)

    @staticmethod
    def make_random_phases(n_chans, seed=None):
        rng1 = np.random.default_rng(seed=seed)
        return rng1.random(n_chans) * 2 * np.pi

    @classmethod
    def make_targ_phases_table(cls, data=None, nw=None, n_chans=None, seed=None):
        data = cls._gen_data_if_not_given(
                data=data, n_chans=n_chans,
                gen_none=functools.partial(cls.make_random_phases, seed=seed),
                gen_scalar=lambda d, n: np.full((n, ), d)
                )
        return cls._make_simple_data_table(data, "targ_phases", nw=nw)

    @classmethod
    def make_targ_mask_table(cls, data, nw=None, n_chans=None):
        data = cls._gen_data_if_not_given(
                data=data, n_chans=n_chans,
                gen_none=lambda n: np.ones((n, ), dtype=bool),
                gen_scalar=lambda d, n: np.full((n, ), bool(d))
                )
        return cls._make_simple_data_table(data, "targ_mask", nw=nw)

    @classmethod
    def make_targ_freqs_table(cls, tone_prop_table, nw=None, flo=None):
        if isinstance(tone_prop_table, RoachToneProps):
            tbl = tone_prop_table.table
        else:
            tbl = QTable(tone_prop_table)
        if 'nw' not in tbl.colnames:
            # try to infer nw
            if nw is None:
                # try get it from the table meta
                _nw = None
                for k in ["nw", "roach"]:
                    v = t.meta.get(k, None)
                    if v is not None:
                        _nw = v
                        break
                if _nw is None:
                    raise ValueError("unable to find network id.")
                nw = _nw
        elif nw is not None:
            # nw will be used to filter the table
            tbl = tbl[tbl["nw"] == nw]
        else:
            # the table may contain multiple nws.
            pass

        # make sure we have at lease f_comb
        cns = tbl.colnames
        if flo is None:
            flo = tbl.meta.get("Header.Toltec.LoCenterFreq", None)
            flo = tbl.meta.get("flo", None)
        # make sure it has a unit
        if flo is not None:
            flo = flo << u.Hz

        def _use_unit(c, unit):
            return (c << unit).to(unit)

        if 'f_comb' not in cns and 'f_centered' not in cns:
            # try compute from LO and f_tone
            if flo is None:
                raise ValueError("unable to infer f_comb without LO freq.")
            if 'f_tone' not in cns and 'f_out' not in cns:
                raise ValueError("no f_tone found for infering f_comb.")
            f_combs = None  # to e updated later
        elif 'f_comb' in cns:
            f_combs = _use_unit(tbl['f_comb'], u.Hz)
        elif 'f_centered' in cns:
            f_combs = _use_unit(tbl['f_centered'], u.Hz)
        else:
            raise ValueError("should not happen.")
        if 'f_tone' in cns:
            f_tones = _use_unit(tbl['f_tone'], u.Hz)
        elif 'f_out' in cns:
            f_tones = _use_unit(tbl['f_out'], u.Hz)
        elif flo is not None:
            f_tones = f_combs + flo
        else:
            # no way to infer f_tone
            raise ValueError('unable to infer f_tone.')
        if f_combs is None:
            f_combs = f_tones - flo

        tbl_targ_freqs = Table()
        tbl_targ_freqs.meta.update({
            "flo": flo.to_value(u.Hz),
            "nw": nw,
            })

        tbl_targ_freqs['f_centered'] = f_combs.to_value(u.Hz)
        tbl_targ_freqs['f_out'] = f_tones.to_value(u.Hz)
        # copy over extra model params if not present in table.
        for c in cls._kids_model_colnames:
            if c not in cns:
                if c == 'f_in':
                    v = tbl_targ_freqs['f_out']
                else:
                    v = 0.
            else:
                v = tbl[c]
            tbl_targ_freqs[c] = v
        # update metadata
        for k, defval in [
                ("Header.Toltec.LoCenterFreq", flo.to_value(u.Hz) if flo is not None else None),
                ("Header.Toltec.ObsNum", 99),
                ("Header.Toltec.SubObsNum", 0),
                ("Header.Toltec.ScanNum", 0),
                ("Header.Toltec.RoachIndex", nw),
                ]:
            tbl_targ_freqs.meta[k] = tbl.meta.get(k, defval)
        # add a flag to allow the table to be recognized and to avoid
        # re-compute
        cls._mark_validated(tbl_targ_freqs, "targ_freqs_table")
        return tbl_targ_freqs

    def _write_table(self, tbl, item, tbl_fmt, data_maker, nw=None,  **kwargs):
        logger.debug(f"write table {item=}\n{tbl=}")
        if not self._check_validated(tbl, item):
            tbl = data_maker(tbl, nw=nw, **kwargs)
        if nw is None:
            nw = tbl.meta.get("nw", None)
        if nw is None:
            raise ValueError("cannot find network id to write to.")
        outpath = self.get_item_path(nw, item)
        logger.debug(f"write validated {item} data to {outpath}\n{tbl}")
        if self._do_dry_run:
            print(f"DRY RUN: {outpath=}\n{tbl}")
        else:
            tbl.write(outpath, format=tbl_fmt, overwrite=True)
        return outpath

    def write_targ_amps_table(self, tbl, nw=None):
        return self._write_table(
                tbl, "targ_amps", "ascii.no_header",
                self.make_targ_amps_table, nw=nw
                )

    def write_targ_phases_table(self, tbl, nw=None):
        return self._write_table(
                tbl, "targ_phases", "ascii.no_header",
                self.make_targ_phases_table, nw=nw
                )

    def write_targ_mask_table(self, tbl, nw=None):
        return self._write_table(
                tbl, "targ_mask", "ascii.no_header",
                self.make_targ_mask_table, nw=nw
                )

    def write_targ_freqs_table(self, tbl, nw=None, flo=None):
        return self._write_table(
                tbl, "targ_freqs", "ascii.ecsv",
                self.make_targ_freqs_table, nw=nw,
                flo=flo,
                )

    def write_tone_prop_table(self, tbl, nw=None, flo=None, items=None):
        paths = {}
        if items is None:
            items = all_items = {
                    "targ_freqs", "targ_amps", "targ_phases", "targ_mask"}
        if any(i not in all_items for i in items):
            raise ValueError('invalid item')
        if 'targ_freqs' in items:
            paths['targ_freqs'] = self.write_targ_freqs_table(
                    tbl, nw=nw, flo=flo)
        if 'targ_amps' in items:
            paths['targ_amps'] = self.write_targ_amps_table(tbl, nw=nw)
        if 'targ_phases' in items:
            paths['targ_phases'] = self.write_targ_phases_table(tbl, nw=nw)
        if 'targ_mask' in items:
            paths['targ_mask']= self.write_targ_mask_table(tbl, nw=nw)
        return paths

    def _set_tone_props_file_suffix(self, nw, old_suffix, suffix, items=None):
        paths = {}
        if items is None:
            items = all_items = {
                    "targ_freqs", "targ_amps", "targ_phases", "targ_mask"}
        if any(i not in all_items for i in items):
            raise ValueError('invalid item')
        with self.set_file_suffix(old_suffix):
            for item in items:
                paths[item] = [self.get_item_path(nw, item)]
        with self.set_file_suffix(suffix):
            for item in items:
                paths[item].append(self.get_item_path(nw, item))
        print(paths)
        for old, new in paths.values():
            logger.debug(
                f"rename file {old.name} -> {new.name} in {old.parent.name}")
            if self._do_dry_run:
                print(f"DRY RUN: rename {old.name} -> {new.name} in {old.parent.name}")
            else:
                old.rename(new)
        return paths

    def backup_tone_prop_files(self, nw, suffix='backup', items=None):
        return self._set_tone_props_file_suffix(nw, None, suffix, items=items)

    def restore_tone_prop_files(self, suffix):
        return self._set_tone_props_file_suffix(nw, suffix, None, items=items)

    def _read_table(self, nw, item, tbl_fmt, tbl_kw):
        path = self.get_item_path(nw, item)
        tbl = QTable.read(path, format=tbl_fmt, **tbl_kw)
        tbl.meta['nw'] = nw
        self._mark_validated(tbl, item)
        return tbl

    def read_targ_freqs_table(self, nw):
        return self._read_table(nw, "targ_freqs", "ascii.ecsv", {})

    def read_targ_amps_table(self, nw):
        item = 'targ_amps'
        return self._read_table(
            nw, item, "ascii.no_header",
            {"names": [item]}
            )

    def read_targ_phases_table(self, nw):
        item = 'targ_phases'
        return self._read_table(
            nw, item, "ascii.no_header",
            {"names": [item]}
            )

    def read_targ_mask_table(self, nw):
        item = 'targ_mask'
        return self._read_table(
            nw, item, "ascii.no_header",
            {"names": [item]}
            )

    def get_tone_props(self, nw):
        tpt = QTable()

        targ_freqs = self.read_targ_freqs_table(nw)

        tpt['f_comb'] = targ_freqs['f_centered']
        tpt['f_tone'] = targ_freqs['f_out']

        targ_amps = self.read_targ_amps_table(nw)
        if len(targ_amps) != len(tpt):
            logger.warning("inconsistent amp file, ignore")
            tpt['amp_tone'] = 1.
        else:
            tpt['amp_tone'] = targ_amps['targ_amps']

        targ_phases = self.read_targ_phases_table(nw)
        if len(targ_phases) != len(tpt):
            logger.warning("inconsistent phase file, ignore")
            tpt['phase_tone'] = self.make_random_phases(len(tpt))
        else:
            tpt['phase_tone'] = targ_phases['targ_phases']

        targ_mask = self.read_targ_mask_table(nw)
        if len(targ_mask) != len(tpt):
            logger.warning("inconsistent phase file, ignore")
            tpt['mask_tone'] = 1
        else:
            tpt['mask_tone'] = targ_mask['targ_mask']
        return RoachToneProps(tpt)


    def get_n_chans(self, nw):
        return len(self.read_targ_freqs_table(nw))


    def get_last_flo(self, nw):
        with self.get_item_path(nw, 'last_flo').open('r') as fo:
            flo = float(fo.read()) << u.Hz
            logger.debug(f"get last {flo=} in {nw=}")
            return flo


    @classmethod
    def create(cls, tlaloc_etc_rootpath, nws=None, exist_ok=False):
        etcdir = Path(tlaloc_etc_rootpath)
        if etcdir.exists() and not exist_ok:
            raise ValueError('etc directory exists, abort.')
        if nws is None:
            nws = range(13)
        for nw in nws:
            etcdir.joinpath(f'toltec{nw}').mkdir(
                    exist_ok=exist_ok, parents=True)
        return cls(path=etcdir)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.rootpath}){self.nws}'


if __name__ == '__main__':

    from astropy.table import Table
    # some tests

    test_path = 'test_tone_utils_tlaloc_etc'
    tlaloc = TlalocEtcDataStore.create(test_path, exist_ok=True)

    # make sample data

    tbl = Table()

    tbl['f_comb'] = range(-10, 10)
    tbl['amp_tone'] = 0.5
    tbl['phase_tone'] = tlaloc.make_random_phases(len(tbl))
    tbl['mask_tone'] = 1
    tbl['mask_tone'][:5] = 0

    paths = tlaloc.write_tone_prop_table(tbl, nw=1, flo=100)
    print(paths)

    rtp = tlaloc.get_tone_props(nw=1)
    print(rtp)
    print(rtp.table)

    paths2 = tlaloc.write_tone_prop_table(rtp, nw=2)

    paths3 = tlaloc.backup_tone_prop_files(nw=2)

    rtp2 = rtp[:2]
    print(rtp2)
    print(rtp2.table)
    tlaloc.write_tone_prop_table(rtp2, nw=2)

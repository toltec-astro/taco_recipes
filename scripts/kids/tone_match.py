from __future__ import annotations
from typing import TYPE_CHECKING, Literal
from tollan.utils.fmt import pformat_yaml, pformat_mask
from tollan.utils.log import logger, timeit, reset_logger
from tolteca_config.core import RuntimeContext
from tolteca_datamodels.toltec.file import SourceInfoModel
from tolteca_datamodels.toltec.types import ToltecDataKind, ToltecArray
from tolteca_datamodels.toltec.ncfile import NcFileIO
from tolteca_kids.core import Kids
from tolteca_kids.match1d import Match1D, Match1DResult
from tollan.config.types import AbsDirectoryPath
from tolteca_config.core import ConfigHandler, ConfigModel, SubConfigKeyTransformer
from tolteca_kids.plot import PlotMixin
from tolteca_kids.filestore import FileStoreConfigMixin
from pydantic import Field
import astropy.units as u
import pandas as pd
from pathlib import Path
from tollan.config.types import FrequencyQuantityField
from astropy.table import QTable, vstack, Column, join

if TYPE_CHECKING:
    from tolteca_datamodels.toltec.file import SourceInfoDataFrame


def _validate_inputs(tbl: SourceInfoDataFrame, check_unique_key=None):
    """Return validated inputs for pipeline."""
    stbl = tbl.query(
        "file_suffix in ['tune', 'targsweep', 'vnasweep'] & file_ext == '.nc'").copy()
    if len(stbl) == 0:
        raise ValueError("no sweep data found in inputs.")
    stbl = stbl.toltec_file.read()
    logger.debug(
        f"{len(stbl)} of {len(tbl)} sweep data found in inputs\n"
        f"{stbl.toltec_file.pformat(type='short')}"
    )
    for k in check_unique_key or []:
        v_unique = np.unique(stbl[k])
        if len(v_unique) > 1:
            raise ValueError(f"inconsistent {k}: {v_unique}")
        v = v_unique[0]
        logger.debug(f"validated input for pipeline unique {k}={v}")
    return stbl

class ToneMatchConfig(ConfigModel):
    """The config for tone match"""

    match: Match1D = Field(
        default={
            "method": "dtw_python",
        },
        description="match parameters.",
    )
    match_shift_max: FrequencyQuantityField = Field(
        default=10 << u.MHz,
        description="The maximum shift allowed in match.",
    )
    match_shift_step: FrequencyQuantityField = Field(
        default=1 << u.kHz,
        description="The grid step in computing match shift.",
    )
    output_path: AbsDirectoryPath = Field(
        description="Output path.",
    )
    save_plot: bool = Field(default=False, description="Save plots.")


class ToneMatch(
    SubConfigKeyTransformer[Literal["tone_match"]],
    ConfigHandler[ToneMatchConfig],
):
    """The class to perform tone match."""

    def __call__(
            self,
            tbl: SourceInfoDataFrame,
            ref: None | str | QTable=None,
    ):
        """Run tone match for given dataset."""
        tbl = _validate_inputs(tbl, check_unique_key=["roach"])
        return self.tone_match(tbl, ref=ref)

    @timeit
    def tone_match(
            self,
            tbl: SourceInfoDataFrame,
            ref: None | str | QTable=None,
            ):
        cfg = self.config
        output_path = cfg.output_path 
        tbl = tbl.toltec_file.read()
        swps = tbl.toltec_file.data_objs
        uid_ref, swp_ref, tbl_kids_ref, f_kids_ref = self._resolve_ref(tbl, ref)
        tbl_result = []
        for swp in swps:
            uid = swp.meta["uid_raw_obs_file"]
            if uid == uid_ref:
                continue
            logger.debug(f"run tone match {uid} to {uid_ref}")
            tbl_kids, f_kids = self._get_tbl_kids(swp)
            if tbl_kids is None:
                logger.warning(f"unable to get kids table for {uid}, skipped")
                continue
            try:
                ctx = self.tone_match2(
                    cfg.match,
                    tbl_kids,
                    f_kids,
                    tbl_kids_ref,
                    f_kids_ref,
                    match_shift_max=cfg.match_shift_max,
                    match_shift_step=cfg.match_shift_step,
                    )
            except Exception:
                logger.warning(f"failed to match {uid}, skipped")
                continue
            tbl_result.append({
                "swp": swp,
                "swp_ref": swp_ref,
                "ctx": ctx,
                "uid": uid,
                "uid_ref": uid_ref,
                "n_kids": len(ctx["tbl_kids"]),
                "n_kids_ref": len(ctx["tbl_kids_ref"]),
                "n_matched": ctx["tbl_matched"]["mask_match_unique"].sum(),
                "n_matched_ref": ctx["tbl_matched_ref"]["mask_match_unique"].sum(),
                "shift": ctx["matched"].shift.to(u.kHz),
            })
            ctx["tbl_matched"].write(
                self._make_kpt_filepath(swp, f"_matched_{uid_ref}.ecsv", create=True),
                format='ascii.ecsv',
                overwrite=True,
                )
        if not tbl_result:
            message = "no kids data found" 
            return locals()
        tbl_result = pd.DataFrame.from_records(tbl_result)
        tbl_result["frac_matched"] = tbl_result["n_matched"] / tbl_result["n_kids"]
        tbl_result["frac_matched_ref"] = tbl_result["n_matched_ref"] / tbl_result["n_kids_ref"]
        tbl_result_display = tbl_result[
            [c for c in tbl_result.columns if c not in ["swp", "swp_ref", "ctx"]]]
        logger.info(f"tbl_result:\n{tbl_result_display}")
        n_kids_ref = tbl_result["n_kids_ref"].min().item()
        n_kids_min = tbl_result["n_kids"].min().item()
        n_kids_max = tbl_result["n_kids"].max().item()
        frac_matched_min = tbl_result["frac_matched_ref"].min().item()
        frac_matched_max = tbl_result["frac_matched_ref"].max().item()
        n_success = len(tbl_result)
        message = f"{n_success=} {n_kids_ref=} {n_kids_min=} {n_kids_max=} frac_min={frac_matched_min:.2%} frac_max={frac_matched_max:.2%}"

        # save all in ref output
        tbl_collated = None
        for entry in tbl_result.itertuples():
            if tbl_collated is None:
                tbl_collated = entry.ctx["tbl_matched_ref"].copy()
            else:
                # add the uid_ref columns
                tr = entry.ctx["tbl_matched_ref"]
                for c in tr.colnames:
                    if c.endswith(entry.ctx["matched_suffix"]):
                        tbl_collated[c] = tr[c]
        tbl_collated_filepath_suffix = f"_matched_{tbl_result['uid'].iloc[0]}_g{len(tbl_result)}.ecsv"
        if swp_ref is not None:
            tbl_collated_filepath = self._make_kpt_filepath(
                swp_ref,
                tbl_collated_filepath_suffix,
                )
        else:
            tbl_collated_dir = self.config.output_path.joinpath(uid_ref)
            if not tbl_collated_dir.exists():
                tbl_collated_dir.mkdir(exist_ok=True)
            tbl_collated_filepath_stem = Path(tbl_kids_ref.meta["filepath"]).stem
            tbl_collated_filepath = tbl_collated_dir.joinpath(
                f"{tbl_collated_filepath_stem}{tbl_collated_filepath_suffix}"
            )
        tbl_collated.write(
            tbl_collated_filepath,
            format='ascii.ecsv',
            overwrite=True,
            )
        result = locals()
        if cfg.save_plot:
            self.plot_tone_match(result)
        return result
    
    def _resolve_ref(self, tbl, ref):
        if ref is None:
            swp_ref = tbl.toltec_file.data_objs.iloc[0]
            uid_ref = swp_ref.meta['uid_raw_obs_file']
            logger.debug(f"use first swp as ref {uid_ref=}")
            tbl_kids_ref, f_kids_ref = self._get_tbl_kids(swp_ref)
        elif isinstance(ref, str):
            stbl_ref = tbl.query(ref)
            if len(stbl_ref) > 1:
                logger.warning(f"found more than one ref with select clause {ref=}, use first one.")
            elif len(stbl_ref) == 0:
                raise ValueError(f"ref not found with select clause {ref=}")
            swp_ref = stbl_ref.toltec_file.data_objs.iloc[0]
            uid_ref = swp_ref.meta['uid_raw_obs_file']
            logger.debug(f"use seleted ref {ref=} {uid_ref=}")
            tbl_kids_ref, f_kids_ref = self._get_tbl_kids(swp_ref)
        elif isinstance(ref, QTable):
            # select matching roach
            roach = tbl["roach"].iloc[0]
            swp_ref = None
            # TODO rewrite after we got the new APT output format spec.
            tbl_kids_ref = ref[ref["nw"] == roach]
            uid_ref = f"apt-{tbl_kids_ref.meta['obsnum']}"
            f_kids_ref = tbl_kids_ref["f_chan"] = tbl_kids_ref["tone_freq"] << u.Hz
            tbl_kids_ref["idx_chan"] = tbl_kids_ref["kids_tone"].astype(int)
            tbl_kids_ref["amp_tone"] = 0.
            logger.info(f"use ref table:\n{tbl_kids_ref}")
        else:
            raise TypeError(f"invalid ref type specified: {type(ref)}")
        return uid_ref, swp_ref, tbl_kids_ref, f_kids_ref


    def _get_kpt_dir(self, swp, uid=None, create=False):
        if uid is None:
            uid = swp.meta["uid_raw_obs"]
        kpt_dir = self.config.output_path.joinpath(uid)
        if not kpt_dir.exists() and create:
            kpt_dir.mkdir(exist_ok=True, parents=True)
        return kpt_dir

    def _make_kpt_filepath(self, swp, suffix, uid=None, create=False):
        return self._get_kpt_dir(swp, uid=uid, create=create).joinpath(
            Path(swp.meta['filename_orig']).stem + suffix
        )
       
    def _get_tbl_kids(self, swp):

        def _get_kpt(swp, suffix):
            kpt_dir = self._get_kpt_dir(swp)
            roach = swp.meta["roach"]
            obsnum = swp.meta["obsnum"]
            subobsnum = swp.meta["subobsnum"]
            scannum = swp.meta["scannum"]
            pattern = f"toltec{roach}_{obsnum:06d}_{subobsnum:03d}_{scannum:04d}_*_{suffix}.ecsv"
            filepaths = list(kpt_dir.glob(pattern))
            if not filepaths:
                return None
            return QTable.read(filepaths[0], format='ascii.ecsv')

        # this tries to locate kids data product for more info
        is_vna = swp.meta["file_suffix"] == "vnasweep"
        if is_vna:
            tbl_kf = _get_kpt(swp, "kids_find")
            if tbl_kf is None:
                return None, None
            return tbl_kf, tbl_kf["f_det"]
        # use chan info for other sweeps
        tbl_chans = swp.meta["chan_axis_data"]
        mask_tone = tbl_chans["mask_tone"]
        tbl_chans = tbl_chans[mask_tone]
        tbl_chans2 = _get_kpt(swp, "chan_prop")
        if tbl_chans2 is not None:
            assert len(tbl_chans2) == len(tbl_chans)
            tbl_chans = tbl_chans2
        else:
            tbl_chans2 = tbl_chans.copy()
            tbl_chans2.meta.clear()
            for k, v in tbl_chans.meta.items():
                if isinstance(v, (str, int, float, bool)):
                    tbl_chans2.meta[k] = v
                elif isinstance(v, Path):
                    tbl_chans2.meta[k] = str(v)
                else:
                    pass
            tbl_chans2["idx_chan"] = tbl_chans["id"]
            tbl_chans = tbl_chans2
        if tbl_chans is None:
            return None, None
        return tbl_chans, tbl_chans["f_chan"]
    
    @staticmethod
    def tone_match2(
        matcher, tbl_kids, f_kids, tbl_kids_ref, f_kids_ref,
        match_shift_max=10 << u.MHz,
        match_shift_step=2 << u.kHz,
    ):
        logger.debug(
            f"n_tones={len(tbl_kids)} n_tones_ref={len(tbl_kids_ref)}"
            f" {match_shift_max=} {match_shift_step=}"
            )

        def _match_postproc(r: Match1DResult):
            matched = r.matched
            # current chan info
            iq = matched["idx_query"]
            matched["idx_chan"] = tbl_kids["idx_chan"][iq]
            matched["f_chan"] = tbl_kids["f_chan"][iq]
            matched["amp_tone"] = tbl_kids["amp_tone"][iq]
            if "Qr" in tbl_kids.colnames:
                Qr = matched["Qr"] = tbl_kids["Qr"][iq]
            else:
                Qr = None

            # ref chan info
            ir = matched["idx_ref"]
            matched["idx_chan_ref"] = tbl_kids_ref["idx_chan"][ir]
            matched["f_chan_ref"] = tbl_kids_ref["f_chan"][ir]
            matched["amp_tone_ref"] = tbl_kids_ref["amp_tone"][ir]
            xx = matched["x_match"] = matched["dist"] / matched["ref"]
            if "Qr" in tbl_kids_ref.colnames:
                Qr_ref = matched["Qr_ref"] = tbl_kids_ref["Qr"][ir]
            else:
                Qr_ref = None
            if not (Qr is None or Qr_ref is None):
                rr = 0.5 / Qr
                matched["d_phi_match"] = np.rad2deg(np.arctan2(xx, rr)) << u.deg
            return r

        matched = matcher(
            query=f_kids.to(u.MHz),
            ref=f_kids_ref.to(u.MHz),
            postproc_hook=_match_postproc,
            shift_kw={
                "shift_max": match_shift_max,
                "dx": match_shift_step,
            },
        )

        def _make_unique(tbl, idx_key, idx_other_key, dist_key):
            # assign flags to matched and remove duplicates
            t = tbl.copy()
            t.sort(dist_key)
            _, unique_indices = np.unique(t[idx_other_key], return_index=True)
            tbl["mask_match_unique"] = False
            tbl["mask_match_unique"][t[idx_key][unique_indices]] = True
            return tbl
 
        tbl_matched = tbl_kids.copy()
        tq = matched.data["query_matched"]
        tbl_matched["f"] = tq["query"]
        tbl_matched["f_ref"] = tq["ref"]
        for colname in [
            "idx_query",
            "idx_ref",
            "dist",
            "dist_shifted",
            "adist_shifted",
            "idx_chan_ref",
            "f_chan_ref",
            "amp_tone_ref",
            "x_match",
            "Qr_ref",
            "d_phi_match",
        ]:
            if colname in tq.colnames:
                tbl_matched[colname] = tq[colname]
        tbl_matched = _make_unique(tbl_matched, "idx_query", "idx_ref", "adist_shifted")
        tbl_matched_ref = tbl_kids_ref.copy()
        matched_suffix = tbl_kids.meta["uid_raw_obs"]
        tr = matched.data["ref_matched"]
        tbl_matched_ref["idx_ref"] = tr["idx_ref"]
        tbl_matched_ref["f_ref"] = tr["ref"]
        tbl_matched_ref[f"idx_{matched_suffix}"] = tr["idx_query"]
        tbl_matched_ref[f"f_{matched_suffix}"] = tr["query"]
        for colname in [
            "dist",
            "dist_shifted",
            "adist_shifted",
            "idx_chan",
            "f_chan",
            "amp_tone",
            "x_match",
            "Qr"
            "d_phi_match",
        ]:
            if colname in tr.colnames:
                tbl_matched_ref[f"{colname}_{matched_suffix}"] = tr[colname]
        tbl_matched_ref = _make_unique(tbl_matched_ref, "idx_ref", f"idx_{matched_suffix}", f"adist_shifted_{matched_suffix}")
        logger.debug(f"tbl_matched_ref:\n{tbl_matched_ref}")
        tbl_matched_ref.meta.update({f"tone_match_{matched_suffix}": tbl_kids.meta})
        return locals()

    def plot_tone_match(self, ctx):
        cfg = self.config
        tbl_result = ctx["tbl_result"]
        for entry in tbl_result.itertuples():
            filepath = self._make_kpt_filepath(
                entry.swp, f"_matched_{entry.uid_ref}.html")
            fig = self._make_tone_match_fig(entry)
            FileStoreConfigMixin.save_plotly_fig(filepath, fig)

    def _make_tone_match_fig(self, entry):
        ctx = entry.ctx
        matched = ctx["matched"]
        fig = PlotMixin.make_subplots(
            n_rows=2,
            n_cols=1,
            vertical_spacing=40 / 1200,
            fig_layout=PlotMixin.fig_layout_default
            | {
                "showlegend": False,
                "height": 1200,
            },
        )
        match_panel_kw = {"row": 1, "col": 1}
        match_density_panel_kw = {"row": 2, "col": 1}

        matched.make_plotly_fig(
            type="match",
            fig=fig,
            panel_kw=match_panel_kw,
            label_value="Frequency (MHz)",
            label_ref=f"Ref {entry.uid_ref}",
            label_query=f"Current {entry.uid}",
        )
        matched.make_plotly_fig(
            type="density",
            fig=fig,
            panel_kw=match_density_panel_kw,
            label_ref=f"Ref Channel Id of {entry.uid_ref}",
            label_query=f"Current Channel Id of {entry.uid}",
        )
        fig.update_layout(
            title={
                "text": f"ToneMatch summary: {entry.uid} with ref {entry.uid_ref}",
            },
            margin={
                "t": 100,
                "b": 100,
            },
        )
        return fig


class AptMakeConfig(ConfigModel):
    """The config for APT make."""

    dist_max: FrequencyQuantityField = Field(
        default=100 << u.kHz,
        description="The maximum allowed separation in match.",
    )
    output_path: AbsDirectoryPath = Field(
        description="Output path.",
    )
    save_plot: bool = Field(default=False, description="Save plots.")


class AptMake(
    SubConfigKeyTransformer[Literal["apt_make"]],
    ConfigHandler[AptMakeConfig],
):
    """The class to make APT."""

    def __call__(
            self,
            tbl: SourceInfoDataFrame,
            ref: None | str | QTable=None,
            tbl_model_search_paths=None,
    ):
        """Run apt make for given dataset."""
        tbl = _validate_inputs(tbl, check_unique_key=["uid_raw_obs"])
        return self.apt_make(tbl, ref=ref, tbl_model_search_paths=tbl_model_search_paths)

    @timeit
    def apt_make( 
            self,
            tbl: SourceInfoDataFrame,
            ref: None | str | QTable=None,
            tbl_model_search_paths=None,
            ):
        cfg = self.config
        output_path = cfg.output_path 

        if ref is None:
            uid_ref_pattern = "*"
        elif isinstance(ref, str):
            uid_ref_patten=tbl.query(ref)["uid_raw_obs"].iloc[0]
        elif isinstance(ref, QTable):
            uid_ref_pattern = f"apt-{ref.meta['obsnum']}"

        else:
            raise TypeError(f"invalid ref type {type(ref)}")
        uid = tbl["uid_raw_obs"].iloc[0]
        logger.debug(f"make apt for {uid=} use {uid_ref_pattern=}")
        
        def _tbl_model_to_apt(entry, tbl_model):
            interface = entry.interface
            roach = entry.roach
            array_name = ToltecArray.interface_array_name[interface]
            array = ToltecArray.array_names.index(array_name) 
            tbl = tbl_model.copy()
            # prefix all columns with kids_model_
            for c in tbl.colnames:
                tbl.rename_column(c, f'kids_{c}')
            n_chans = len(tbl)
            tbl.add_column(Column(np.full((n_chans, ), roach), name='nw'), 0)
            tbl.add_column(Column(np.full((n_chans, ), array), name='array'), 0)
            tbl.add_column(Column(range(n_chans), name='kids_tone'), 0)
            return tbl
        
        def _clean_up_matched(entry, tbl):
            # check dist and update the flag
            dist_max = cfg.dist_max
            mask_dist = (tbl["adist_shifted"] <= dist_max)
            logger.debug(f"mask dist with {dist_max=}: {pformat_mask(mask_dist)}")
            tbl["m_good_ref"] = tbl["mask_match_unique"] & mask_dist
            return tbl
        
        def _add_apt_cols(entry, tbl_apt, tbl_apt_ref):
            tbl_apt_ref.meta["obsnum_matched"] = tbl_apt_ref.meta.pop("obsnum")
            t = join(
                tbl_apt,
                tbl_apt_ref,
                keys="idx_ref",
                join_type="left",
                keep_order=True,
                uniq_col_name="{col_name}{table_name}",
                table_names=["", "_ref"],
                )
            return t

        tbl_apt = []
        for entry in tbl.sort_values("roach").itertuples():
            tbl_matched = self._get_tbl_kids(
                entry,                
                suffix=f"_matched_{uid_ref_pattern}",
                ext='ecsv'
            )
            tbl_matched = _clean_up_matched(entry, tbl_matched)
            logger.debug(f"use good matches: {pformat_mask(tbl_matched['m_good_ref'])}")

            tbl_model = self._get_tbl_model(entry, search_paths=tbl_model_search_paths)
            tbl = _tbl_model_to_apt(entry, tbl_model)
            for c in ["idx_ref", "idx_chan_ref", "dist", "adist_shifted"]:
                tbl[c] = tbl_matched[c]
            tbl["idx_ref"][~tbl_matched["m_good_ref"]] = -1
            if isinstance(ref, QTable):
                # attach apt info from match
                tbl_apt_ref = ref[ref["nw"] == entry.roach]
                tbl_apt_ref["idx_ref"] = tbl_apt_ref["kids_tone"].astype(int)
                tbl_apt_ref["flag_ref"] = tbl_apt_ref["flag"]
                tbl = _add_apt_cols(entry, tbl, tbl_apt_ref)
            # remove model columns
            tbl_apt.append(tbl)

        tbl_apt = vstack(tbl_apt, metadata_conflicts="silent")
        tbl_apt.add_column(Column(range(len(tbl_apt)), name='det_id'), 0)
        for k in ['obsnum', 'subobsnum', 'scannum']:
            tbl_apt.meta[k] = getattr(entry, k)
        tbl_apt = tbl_apt.filled(0.)
        tbl_apt["flag"] = (tbl_apt["idx_ref"] < 0).astype(float)
        # propogate flag from ref
        if "flag_ref" in tbl_apt.colnames:
            flag = tbl_apt["flag"].astype(bool)
            flag_ref = tbl_apt["flag_ref"].astype(bool)
            flag_new = flag | flag_ref
            logger.debug(f"update ref flag {pformat_mask(flag_ref)} with flag {pformat_mask(flag)} to new flag {pformat_mask(flag_new)}")
            tbl_apt["flag"] = flag_new.astype(float)
   
        n_good =(tbl_apt["flag"] == 0).sum().item()
        n_chans = len(tbl_apt)
        message = f"{n_good=} {n_chans=} {n_good/n_chans:.2%}"
        return locals()

    def _get_tbl_kids(self, entry, suffix, ext):
        output_path = self.config.output_path
        kpt_dir = output_path.joinpath(entry.uid_raw_obs)
        roach = entry.roach
        obsnum = entry.obsnum
        subobsnum = entry.subobsnum
        scannum = entry.scannum
        pattern = f"toltec{roach}_{obsnum:06d}_{subobsnum:03d}_{scannum:04d}_*{suffix}.{ext}"
        filepaths = list(kpt_dir.glob(pattern))
        if not filepaths:
            logger.warning(f"unable to locate matched tbl for {entry.uid_raw_obs}")
            return None
        filepath = filepaths[0]
        logger.debug(f"load matched tbl for {entry.uid_raw_obs_file}: {filepath}")
        return QTable.read(filepaths[0], format='ascii.ecsv')
    
    def _get_tbl_model(self, entry, search_paths):
        filepath = entry.filepath
        filestem = filepath.stem
        search_paths_default = [
                filepath.parent.parent.parent / "reduced"
        ]
        _search_paths = [Path(p) for p in (search_paths or []) + search_paths_default]
        logger.debug(f"tbl_model search paths: {_search_paths}")
        for sp in _search_paths:
            tbl_model_path = sp.joinpath(f"{filestem}.txt")
            if tbl_model_path.exists():
                break
        else:
            tbl_model_path = None
        if tbl_model_path is None:
            raise ValueError("unable to find tbl_model file")
        logger.debug(f"load model tbl for {entry.uid_raw_obs_file}: {tbl_model_path}")
        return QTable.read(tbl_model_path, format='ascii.ecsv')


def _run(
    rc: RuntimeContext,
    tbl: SourceInfoDataFrame,
    step_name: str,
    step_cls: type[ConfigHandler],
    rc_ctx: None | dict = None,
    step_kw: None | dict = None,
):
    with rc.config_backend.set_context(rc_ctx or {}):
        step = step_cls(rc)
        logger.debug(f"{step_name} config:\n{step.config.model_dump_yaml()}")
        try:
            ctx = step(tbl, **(step_kw or {}))
        except Exception as e:
            logger.opt(exception=True).error(f"{step_name} failed for {roach=}: {e}")
            returncode = 1
            message = f"{step_name} failed: {e}"
        else:
            message = f"{step_name} {ctx['message']}"
            returncode = 0
        return returncode, locals()

def run_tone_match(
        rc: RuntimeContext,
        tbl: SourceInfoDataFrame,
        ref: None  | str | QTable = None,
    ):
    try:
        tbl = _validate_inputs(tbl, check_unique_key=["roach"])
    except ValueError as e:
        message = str(e)
        if "no sweep data" in message:
            returncode = 0
        else:
            returncode = 1
        return returncode, locals() 
    roach = tbl["roach"].iloc[0].item()
    rc_ctx = {"roach": roach}

    return _run(
        rc,
        tbl,
        "tone match",
        ToneMatch,
        rc_ctx=rc_ctx,
        step_kw={
            "ref": ref,
            },
        )

def run_apt_make(
        rc: RuntimeContext,
        tbl: SourceInfoDataFrame,
        ref: None  | str | QTable = None,
        search_paths=None,
    ):
    try:
        tbl = _validate_inputs(tbl, check_unique_key=["uid_raw_obs"])
    except ValueError as e:
        message = str(e)
        if "no sweep data" in message:
            returncode = 0
        else:
            returncode = 1
        return returncode, locals()
    return _run(
            rc,
            tbl,
            "apt make",
            AptMake,
            rc_ctx={},
            step_kw={"ref": ref, "tbl_model_search_paths": search_paths},
            )


if __name__ == "__main__":
    import sys
    import argparse
    import numpy as np
    from toltec_file_utils import LmtToltecPathOption
    from tollan.utils.cli import dict_from_cli_args, split_cli_args

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="tolteca.yaml")
    parser.add_argument("--log_level", default="INFO", help="The log level.")
    parser.add_argument("--save_plot", action="store_true", help="Save plots.")
    parser.add_argument("--select_ref", default=None, help="Select the ref data.")
    parser.add_argument("--apt_ref", default=None, help="APT to use as ref.")
    parser.add_argument(
        "--search_paths",
        nargs="*",
    )
    parser.add_argument(
        "--save_apt", action="store_true", help="Save APTs.")
    LmtToltecPathOption.add_args_to_parser(
        parser, obs_spec_required=True, obs_spec_multi=True
    )

    cli_args, args = split_cli_args("^tone_match\..+", sys.argv[1:])
    logger.debug(
        f"tone_match_cli_args:\n{pformat_yaml(cli_args)}\n"
        f"other_args:\n{pformat_yaml(args)}",
    )
    option = parser.parse_args(args)
    reset_logger(level=option.log_level)
    logger.debug(f"parsed options: {option}")
    path_option = LmtToltecPathOption(option)

    tbl = path_option.get_raw_obs_info_table(raise_on_empty=True).sort_values(
        ["roach", "file_timestamp"]
        )
    cli_args[:0] = [
        "--tone_match.output_path",
        path_option.dataprod_path,
        "--apt_make.output_path",
        path_option.dataprod_path,
    ]
    if option.save_plot:
        cli_args.extend(
            [
                "--tone_match.save_plot",
                "--apt_make.save_plot",
            ],
        )

    select_ref = option.select_ref
    apt_ref = option.apt_ref
    if select_ref is not None and apt_ref is not None:
        raise ValueError("can only have one of select_ref or apt_ref.")
    elif select_ref is not None:
        ref = select_ref
    elif apt_ref is not None:
        ref = QTable.read(option.apt_ref, format='ascii.ecsv')
        ref.meta["filepath"] = option.apt_ref
    else:
        ref = None

    rc = RuntimeContext(option.config)
    rc.config_backend.update_override_config(dict_from_cli_args(cli_args))
    logger.info(f"{pformat_yaml(rc.config.model_dump())}")

    def _report_run_stats(report, exit_fail_only=False):
        with pd.option_context('display.max_rows', None, 'display.max_colwidth', 200):
            logger.info(f"run status:\n{report}")
        n_failed = np.sum(report["returncode"] != 0)
        if n_failed == 0:
            logger.info("Job's done!")
        else:
            logger.error("Job's failed.")
        if (exit_fail_only and n_failed > 0) or not exit_fail_only:
            sys.exit(n_failed)

    with timeit("run match tones"):
        report_tone_match = []
        for roach, stbl in tbl.groupby("roach", sort=False):
            logger.debug(f"{roach=} n_files={len(stbl)}")
            r, ctx = run_tone_match(
                rc,
                stbl,
                ref=ref,
            )
            report_tone_match.append(
                {
                    "roach": roach,
                    "n_items": len(stbl),
                    "returncode": r,
                    "message": ctx.get("message", None),
                }
            )
    report_tone_match = pd.DataFrame.from_records(report_tone_match)
    _report_run_stats(report_tone_match, exit_fail_only=True)
    if not option.save_apt:
        sys.exit(0)

    # handle APT generation
    with timeit("make apts"):
        report_apt_make = []
        for uid_raw_obs, stbl in tbl.groupby("uid_raw_obs", sort=False):
            logger.debug(f"{uid_raw_obs=} n_files={len(stbl)}")
            r, ctx = run_apt_make(rc, stbl, ref=ref, search_paths=option.search_paths)
            report_apt_make.append(
                {
                    "uid_raw_obs": uid_raw_obs,
                    "n_items": len(stbl),
                    "returncode": r,
                    "message": ctx.get("message", None),
                }
            )
    report_apt_make = pd.DataFrame.from_records(report_apt_make)
    _report_run_stats(report_apt_make)
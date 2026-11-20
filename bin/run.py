# run.py

import argparse
import collections
import hashlib
import importlib
import json
import os
import subprocess as sp
import sys
import time
import types
from glob import glob
from os import path
from pathlib import Path
from zipfile import ZipFile

import bids
import nibabel as nib
import numpy as np


def _import_any(names):
    """Import the first module name that succeeds from a list of candidates.

    Parameters
    ----------
    names : list[str]
        List of module names to try importing in order.

    Returns
    -------
    module
        The imported module object.

    Raises
    ------
    ImportError
        If none of the module names can be imported.
    """
    last_e = None
    for n in names:
        try:
            return importlib.import_module(n)
        except Exception as e:
            last_e = e
            continue
    raise last_e if last_e else ImportError(f"Could not import any of {names}")


# ---------------------------- small utilities -------------------------------
def _sidecar_for(p: Path) -> Path:
    """Return BIDS sidecar JSON path for a NIfTI or GIFTI file.

    Parameters
    ----------
    p : Path
        Path to the NIfTI or GIFTI file.

    Returns
    -------
    Path
        Path to the corresponding JSON sidecar file.
    """
    return p.with_suffix("").with_suffix(".json")


def _read_tr_from_sidecar(json_path: Path, LOG=None) -> float | None:
    """Read TR (seconds) from BIDS sidecar JSON.

    Prefers RepetitionTime; also checks RepetitionTimeEffective, TR, tr.
    Returns None if not available.

    Parameters
    ----------
    json_path : Path
        Path to the JSON sidecar file.
    LOG : logger, optional
        Logger instance for debug/warning messages.

    Returns
    -------
    float or None
        TR value in seconds, or None if not found.
    """
    try:
        if not json_path.exists():
            if LOG:
                LOG.debug(f"Sidecar not found for TR: {json_path}")
            return None
        d = json.loads(json_path.read_text())
        for key in ("RepetitionTime", "RepetitionTimeEffective", "TR", "tr"):
            if key in d and d[key] is not None:
                return float(d[key])
    except Exception as e:
        if LOG:
            LOG.warn(f"Failed reading TR from sidecar {json_path}: {e}")
    return None


def _get_bold_T_and_TR(
    p: Path, LOG=None, img=None
) -> tuple[int | None, float | None]:
    """Return (#timepoints, TR seconds) from a BOLD file and its sidecar.

    T is derived from the image (nibabel) where possible.
    TR is read from sidecar JSON, with optional fallback to header for NIfTI.

    Parameters
    ----------
    p : Path
        Path to the BOLD file (NIfTI or GIFTI).
    LOG : logger, optional
        Logger instance for debug/warning messages.
    img : nibabel image, optional
        The already loaded image of ``p``. Loading a GIFTI parses the whole
        file, so callers that need the image anyway should pass it in.

    Returns
    -------
    tuple[int or None, float or None]
        Tuple of (timepoints, TR_seconds). Either can be None if not available.
    """
    try:
        if img is None:
            img = nib.load(str(p))
        # T from data
        T = None
        try:
            # For GIFTI, one darray per timepoint (or a single 2D darray);
            # for NIfTI shape[-1]
            if hasattr(img, "agg_data"):
                darrays = img.darrays
                if len(darrays) == 1 and np.ndim(darrays[0].data) > 1:
                    T = int(darrays[0].data.shape[-1])
                else:
                    T = len(darrays) or None
            else:
                shp = img.shape
                T = int(shp[-1]) if len(shp) >= 4 else None
        except Exception:
            T = None

        # TR from sidecar
        TR = _read_tr_from_sidecar(_sidecar_for(p), LOG=LOG)

        # Fallback TR for NIfTI if sidecar missing
        if TR is None and not hasattr(img, "agg_data"):
            try:
                TR = float(img.header.get_zooms()[3])
            except Exception:
                TR = None
        return T, TR
    except Exception as e:
        if LOG:
            LOG.warn(f"Failed to read T/TR from {p}: {e}")
        return None, None


def _log_stim_file_tr_mismatch(stim, file_TR: float | None, LOG) -> None:
    """If file_TR is present and differs from stim.tr, warn and update stim.tr.

    Parameters
    ----------
    stim : object
        Stimulus object with a 'tr' attribute.
    file_TR : float or None
        TR value from the file header/sidecar.
    LOG : logger
        Logger instance for warning messages.
    """
    try:
        if file_TR is None:
            return
        if getattr(stim, "tr", None) is None or np.isnan(stim.tr):
            return
        if not np.isclose(float(stim.tr), float(file_TR), rtol=1e-5, atol=1e-8):
            LOG.warn(
                f"Stimulus TR ({float(stim.tr):.4f}s) does not match file TR ({float(file_TR):.4f}s)."
            )
            LOG.warn(f"Using stimulus TR ({float(stim.tr):.4f}s)!")
    except Exception:
        # keep calm if stim.tr is read-only or non-float
        pass


def _log_timepoints_mismatch(
    file_T: int | None,
    aperture_T: int,
    where: str,
    LOG,
    file_TR: float | None = None,
    frame_tr: float | None = None,
) -> None:
    """Log an error if the aperture and the file do not cover the same time.

    With an aperture on the scan's TR the frame counts must match. With
    output_all_frames the aperture is on the stimulus' own frame clock
    (frame_tr, e.g. 0.125 s for 2400 frames over 300 volumes at TR 1 s), so
    the counts differ by design and the durations are compared instead,
    allowing one frame of rounding.

    Parameters
    ----------
    file_T : int or None
        Number of timepoints in the file.
    aperture_T : int
        Number of timepoints in the aperture/stimulus.
    where : str
        Description of where the mismatch was found.
    LOG : logger
        Logger instance for error messages.
    file_TR : float, optional
        TR of the file in seconds.
    frame_tr : float, optional
        Seconds per aperture frame.
    """
    if file_T is None:
        return
    high_res = (
        frame_tr is not None
        and file_TR is not None
        and not np.isclose(float(frame_tr), float(file_TR), rtol=1e-5, atol=1e-8)
    )
    if high_res:
        ap_dur = int(aperture_T) * float(frame_tr)
        file_dur = int(file_T) * float(file_TR)
        if abs(ap_dur - file_dur) > float(frame_tr) + 1e-6:
            LOG.error(
                f"Duration mismatch: stimulus/aperture {ap_dur:g}s (T={int(aperture_T)} @ {float(frame_tr):g}s) "
                f"vs file {file_dur:g}s (T={int(file_T)} @ TR {float(file_TR):g}s) at {where}"
            )
    elif int(file_T) != int(aperture_T):
        LOG.error(
            f"Timepoints mismatch: stimulus/aperture T={int(aperture_T)} vs file T={int(file_T)} at {where}"
        )


def _bold_path_for(
    bold_base_dir: Path,
    analysis_space: str,
    sub: str,
    ses: str | None,
    task: str,
    run: str,
    hemi: str | None,
) -> Path:
    """Construct the expected BOLD file path for the given parameters.

    For fsnative/fsaverage: requires hemi (l/r); returns .func.gii
    For volume: ignores hemi; returns T1w preproc .nii.gz

    Parameters
    ----------
    bold_base_dir : Path
        Base directory containing BOLD files.
    analysis_space : str
        Analysis space ('fsnative', 'fsaverage', or 'volume').
    sub : str
        Subject identifier.
    ses : str or None
        Session identifier.
    task : str
        Task identifier.
    run : str
        Run identifier.
    hemi : str or None
        Hemisphere ('l' or 'r') for surface spaces.

    Returns
    -------
    Path
        Path to the expected BOLD file.

    Raises
    ------
    ValueError
        If hemi is required but not provided, or if analysis_space is invalid.
    """
    ses_part = f"ses-{ses}" if ses else "ses-NA"
    if analysis_space == "fsnative":
        if hemi is None:
            raise ValueError("hemi is required for fsnative analysis_space")
        return (
            bold_base_dir
            / f"sub-{sub}_{ses_part}_task-{task}_run-{run}_hemi-{hemi.upper()}_space-fsnative_bold.func.gii"
        )
    elif analysis_space == "fsaverage":
        if hemi is None:
            raise ValueError("hemi is required for fsaverage analysis_space")
        return (
            bold_base_dir
            / f"sub-{sub}_{ses_part}_task-{task}_run-{run}_hemi-{hemi.upper()}_space-fsaverage_bold.func.gii"
        )
    elif analysis_space == "volume":
        return (
            bold_base_dir
            / f"sub-{sub}_{ses_part}_task-{task}_run-{run}_space-T1w_desc-preproc_bold.nii.gz"
        )
    else:
        raise ValueError(f"Unsupported analysis_space: {analysis_space}")


def _average_runs_bold(bold_paths, LOG, imgs=None):
    """Crop to shortest T and compute voxel-wise mean across multiple BOLD runs.

    Parameters
    ----------
    bold_paths : list[Path]
        List of paths to BOLD files to average.
    LOG : logger
        Logger instance for info messages.
    imgs : list, optional
        Already loaded images aligned with ``bold_paths``; None entries (or no
        list at all) are loaded from their path.

    Returns
    -------
    nibabel image
        The averaged BOLD as an in-memory image (GiftiImage or Nifti1Image); it is
        not written to disk, so it has no filename.

    Raises
    ------
    ValueError
        If BOLD files have different spatial dimensions.
    """
    # Load all inputs not already in memory
    imgs = imgs or [None] * len(bold_paths)
    imgs = [
        img if img is not None else nib.load(str(p))
        for p, img in zip(bold_paths, imgs)
    ]

    # Determine file type from suffix (all must match)
    def _suffix_kind(p: Path) -> str:
        s = "".join(p.suffixes).lower()
        if s.endswith(".nii") or s.endswith(".nii.gz"):
            return "nifti"
        if s.endswith(".gii") or s.endswith(".func.gii"):
            return "gifti"
        return "unknown"

    kinds = {_suffix_kind(Path(p)) for p in bold_paths}
    if len(kinds) != 1 or "unknown" in kinds:
        raise ValueError(
            f"Unsupported or mixed input formats for averaging: {sorted(kinds)}"
        )
    kind = kinds.pop()

    if kind == "nifti":
        # Validate spatial dims and find the shortest T from the headers, so
        # the data can be summed one run at a time instead of all held at once
        spatial_shape = None
        Ts = []
        for i, img in enumerate(imgs):
            if len(img.shape) != 4:
                raise ValueError(f"Expected 4D BOLD NIfTI; got shape {img.shape}")
            if spatial_shape is None:
                spatial_shape = img.shape[:3]
            elif img.shape[:3] != spatial_shape:
                raise ValueError(
                    "All BOLD files must have the same spatial dimensions for averaging"
                )
            elif not np.allclose(img.affine, imgs[0].affine, atol=1e-5):
                raise ValueError(
                    "All BOLD files must share the same voxel grid (affine) for "
                    f"averaging; {bold_paths[i]} differs from {bold_paths[0]}"
                )
            Ts.append(img.shape[-1])

        min_T = min(Ts)
        if min_T < max(Ts):
            LOG.debug(f"Cropping runs to shortest T={min_T} for averaging")

        # Running float32 sum in run order, then divide: bit-identical to
        # np.mean over the stacked runs
        mean = None
        for img in imgs:
            d = img.get_fdata(dtype=np.float32)[..., :min_T]
            if mean is None:
                mean = d.copy()
            else:
                mean += d
            img.uncache()
            del d
        mean /= len(imgs)
        avg_img = nib.Nifti1Image(mean, imgs[0].affine, imgs[0].header)
        LOG.info(f"Averaged {len(bold_paths)} runs (T={min_T})")
        return avg_img

    elif kind == "gifti":
        # Validate vertex count and crop to shortest T (number of darrays)
        nverts = None
        t_counts = []
        for img in imgs:
            if not hasattr(img, "darrays") or len(img.darrays) == 0:
                raise ValueError("GIFTI file has no data arrays")
            v0 = int(np.asarray(img.darrays[0].data).size)
            if nverts is None:
                nverts = v0
            elif v0 != nverts:
                raise ValueError(
                    "All GIFTI files must have the same number of vertices for averaging"
                )
            t_counts.append(len(img.darrays))

        min_T = min(t_counts)
        if min_T < max(t_counts):
            LOG.debug(f"Cropping runs to shortest T={min_T} for averaging")

        gi = nib.gifti.GiftiImage()
        # Preserve top-level meta/labeltable when available
        try:
            gi.meta = imgs[0].meta
        except Exception:
            pass
        try:
            gi.labeltable = imgs[0].labeltable
        except Exception:
            pass

        for t in range(min_T):
            arrs = []
            for img in imgs:
                a = np.asarray(img.darrays[t].data, dtype=np.float32).reshape(-1)
                if a.size != nverts:
                    raise ValueError("Mismatched vertex count across runs")
                arrs.append(a)
            avg = np.mean(np.stack(arrs, axis=0), axis=0).astype(np.float32)

            da = nib.gifti.GiftiDataArray(avg)
            # Attempt to preserve intent/meta from first file
            try:
                da.intent = imgs[0].darrays[t].intent
                da.meta = imgs[0].darrays[t].meta
            except Exception:
                pass
            gi.add_gifti_data_array(da)

        LOG.info(f"Averaged {len(bold_paths)} runs (T={min_T})")
        return gi

    else:
        raise ValueError(f"Unsupported analysis file type for averaging: {kind}")


def _resolve_n_jobs(n_jobs) -> int:
    """Number of masking workers; 0, "auto" or "all" means all usable CPUs."""
    if str(n_jobs).strip().lower() in ("0", "auto", "all", "-1"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except AttributeError:  # not available on macOS
            return max(1, os.cpu_count() or 1)
    return max(1, int(n_jobs))


def _mask_job(job: dict) -> dict:
    """Load, check and mask the BOLD runs of one job; average them if asked.

    Runs in a worker process. Only reads the ROI packs (prepared beforehand
    by the main process), so jobs can run concurrently. Items are processed
    in order; an item marked done only contributes its path to the average.

    Returns
    -------
    dict
        {"averaged": bool}: whether an average was computed, so the main
        process can write its events.tsv.
    """
    apply_masks_to_run = _import_any(
        ["nii_to_surfNii", "scripts.nii_to_surfNii"]
    ).apply_masks_to_run
    prepare_roi_pack = _import_any(["roi_pack", "scripts.roi_pack"]).prepare_roi_pack

    common = job["common"]
    stim = common["stim"]
    timepoints = common["timepoints"]
    continue_on_error = common["continue_on_error"]
    avg_ctx = job.get("avg_ctx")

    # one image per file, so volume jobs masking both hemis read it once
    loaded = {}
    run_paths, run_imgs = [], []
    for it in job["items"]:
        c = it["ctx"]
        LOG = c["log"]
        bold_path = it["bold_path"]
        if it["done"]:
            run_paths.append(bold_path)
            run_imgs.append(None)
            continue
        try:
            t0 = time.perf_counter()
            if bold_path not in loaded:
                loaded[bold_path] = nib.load(str(bold_path))
            bold_img = loaded[bold_path]
            LOG.debug(f"BOLD loaded in {time.perf_counter()-t0:.2f}s: {bold_path.name}")
            c = {**c, "bold_path": bold_path, "bold_img": bold_img}

            # Per-run check: compare aperture T vs file T
            file_T, file_TR = _get_bold_T_and_TR(bold_path, LOG=LOG, img=bold_img)
            _log_stim_file_tr_mismatch(stim, file_TR, LOG)
            _log_timepoints_mismatch(
                file_T,
                timepoints,
                str(bold_path),
                LOG,
                file_TR=file_TR if file_TR is not None else stim.tr,
                frame_tr=common.get("frame_tr"),
            )

            # The pack exists already; this validates the BOLD against it.
            # Never rebuild here: other jobs may be reading the same file.
            roi_pack = prepare_roi_pack(
                ctx={**c, "force": False, "roi_pack_read_only": True},
                bold_img=bold_img,
                **common["pack_args"],
            )

            # ROI masking using the appropriate grid
            t0 = time.perf_counter()
            apply_masks_to_run(
                ctx=c,
                stim=stim,
                roi_pack=roi_pack,
                output_only_average=common["output_only_average"],
            )
            LOG.debug(f"Masking complete in {time.perf_counter()-t0:.2f}s")

            run_paths.append(bold_path)
            # Keep parsed GIFTIs for averaging. A NIfTI is cheap to reopen
            # and would hold its 4D data in memory.
            run_imgs.append(
                bold_img if avg_ctx and hasattr(bold_img, "agg_data") else None
            )
        except Exception as e:
            LOG.error(
                f"Failed run: sub-{c.get('sub')} ses-{c.get('ses') or 'NA'} "
                f"task-{c.get('task')} run-{c.get('run')} hemi-{c.get('hemi')} ({e})"
            )
            if not continue_on_error:
                raise
    loaded.clear()

    averaged = False
    if avg_ctx and len(run_paths) >= 2:
        LOG = avg_ctx["log"]
        LOG.info(
            f"\tAveraging {len(run_paths)} runs for sub-{avg_ctx['sub']} "
            f"task-{avg_ctx['task']} hemi-{avg_ctx['hemi']}"
        )
        try:
            t0 = time.perf_counter()
            avg_img = _average_runs_bold(run_paths, LOG, imgs=run_imgs)
            del run_imgs
            averaged = True
            LOG.debug(f"Average computed in {time.perf_counter()-t0:.2f}s")
            c = {**avg_ctx, "bold_path": avg_img}
            roi_pack = prepare_roi_pack(
                ctx={**c, "force": False, "roi_pack_read_only": True},
                bold_img=avg_img,
                **common["pack_args"],
            )
            t0 = time.perf_counter()
            apply_masks_to_run(
                ctx=c, stim=stim, roi_pack=roi_pack, output_only_average=False
            )
            LOG.debug(f"Masking (avg) complete in {time.perf_counter()-t0:.2f}s")
        except Exception as e:
            LOG.error(
                f"Averaging failed for sub-{avg_ctx['sub']} ses-{avg_ctx.get('ses') or 'NA'} "
                f"task-{avg_ctx['task']} ({e})"
            )
            if not continue_on_error:
                raise
    return {"averaged": averaged}


class _MaskRunner:
    """Runs masking jobs inline (n_jobs=1) or in a pool of spawned processes.

    Jobs are submitted per subject as soon as it is prepared, so masking
    overlaps with preparing the next subject; that is safe because every
    subject has its own ROI pack. Spawned processes inherit nothing (open
    HDF5 handles, caches) from the main process. If a job raises
    (continue_on_error off), pending jobs are cancelled and the error is
    re-raised.
    """

    def __init__(self, n_jobs: int, on_result):
        self.n_jobs = n_jobs
        self.on_result = on_result  # called in the main process per finished job
        self.ex = None
        self.n_submitted = 0
        self.pending = []  # (job, future) not yet handed to on_result

    def submit(self, jobs: list) -> None:
        import concurrent.futures as cf
        import multiprocessing as mp

        for job in jobs:
            self.n_submitted += 1
            if self.n_jobs <= 1:
                self.on_result(job, _mask_job(job))
                continue
            if self.ex is None:
                self.ex = cf.ProcessPoolExecutor(
                    max_workers=self.n_jobs, mp_context=mp.get_context("spawn")
                )
            self.pending.append((job, self.ex.submit(_mask_job, job)))
        self.collect(wait=False)

    def collect(self, wait: bool) -> None:
        """Hand finished jobs to on_result; with wait, wait for all of them.

        A job that raised (continue_on_error off) re-raises here, so a
        failure stops the run at the next subject instead of at the end.
        """
        still = []
        for job, f in self.pending:
            if wait or f.done():
                self.on_result(job, f.result())
            else:
                still.append((job, f))
        self.pending = still

    def close(self, cancel: bool = False) -> None:
        if self.ex is not None:
            self.ex.shutdown(wait=True, cancel_futures=cancel)
            self.ex = None


def _stim_signature(stim) -> str:
    """Digest of everything an aperture is built from, to tell whether two
    sessions' stimuli for a task would produce the same aperture file."""
    h = hashlib.sha1()
    frames = getattr(stim, "frames", None)
    for v in (
        getattr(frames, "images", None),
        getattr(stim, "seq", None),
        getattr(stim, "tr", None),
        getattr(stim, "seqtiming", None),
        getattr(stim, "prescan", None),
        getattr(stim, "numImages", None),
        getattr(stim, "aperture_nii", None),
        getattr(stim, "task", None),
    ):
        if isinstance(v, np.ndarray):
            h.update(repr((v.dtype.str, v.shape)).encode())
            h.update(np.ascontiguousarray(v).tobytes())
        else:
            h.update(repr(v).encode())
        h.update(b"|")
    return h.hexdigest()


def _update_standard_config(in_config: dict, scripts_dir: str) -> dict:
    """Update the input config with values from the default config file.

    Parameters
    ----------
    in_config : dict
        User-provided configuration dictionary.
    scripts_dir : str
        Path to the scripts directory containing default_config.json.

    Returns
    -------
    dict
        Updated configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the default config file is not found.
    """
    default_config_path = scripts_dir / "default_config.json"
    if not default_config_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_config_path}")
    default_config = json.loads(default_config_path.read_text())

    def update(d, u):
        """Recursively update dictionary d with values from dictionary u.

        Parameters
        ----------
        d : dict
            Dictionary to be updated (modified in-place).
        u : dict
            Dictionary containing updates to apply.

        Returns
        -------
        dict
            The updated dictionary d.
        """
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    return update(default_config, in_config)


def _get_out_analysis_name(config: dict, derivatives_dir: Path, LOG) -> str:
    """Define the output directory automatically, starting a new one when config differs.

    Creates analysis-XX directories and reuses existing ones if the config matches.
    Stops after 100 attempts to prevent infinite loops.

    Parameters
    ----------
    config : dict
        Configuration dictionary to compare against existing analysis directories.
    derivatives_dir : Path
        Base derivatives directory.
    LOG : logger
        Logger instance for info messages.

    Returns
    -------
    str
        Analysis directory name (e.g., 'analysis-01').
    """
    analysis_number = 0
    found_outbids_dir = False

    while not found_outbids_dir and analysis_number < 100:
        analysis_number += 1

        out_dir = derivatives_dir / "prfprepare" / f"analysis-{analysis_number:02d}"

        optsFile = out_dir / "options.json"

        # if the analyis-XX directory exists check for the config file
        if out_dir.is_dir() and optsFile.is_file():
            opts = json.loads(optsFile.read_text())

            # check for the options file equal to the config
            if sorted(opts.items()) == sorted(config.items()):
                found_outbids_dir = True

        # when we could not find a fitting analysis-XX forlder we make a new one
        else:
            os.makedirs(out_dir, exist_ok=True)

            # dump the options file in the output directory
            optsFile.write_text(json.dumps(config, indent=4))
            found_outbids_dir = True

    LOG.info(f"Output directory: {out_dir}")
    return f"analysis-{analysis_number:02d}"


def _str2list(s):
    """Turn a string into a list of strings.

    Accepts forms like "[a, b]", "(a,b)", "a,b", with quotes/whitespace.
    None/"" -> [].

    Parameters
    ----------
    s : str or None
        Input string to parse.

    Returns
    -------
    list[str]
        List of parsed strings, empty list for None/empty input.
    """
    if not s:
        return []
    s = str(s).strip()

    # strip outer brackets/parentheses if present
    if "[" in s and "]" in s:
        s = s[s.find("[") + 1 : s.rfind("]")]
    elif "(" in s and ")" in s:
        s = s[s.find("(") + 1 : s.rfind(")")]

    parts = s.split(",")
    out = []
    for p in parts:
        p = p.strip().strip('"').strip("'")
        if p != "":
            out.append(p)
    return out


def _config2list(c, base=None):
    """Minimal, robust normalizer for configuration values.

    If base is given and 'all' (case-insensitive) appears -> sorted(base)
    Strings -> parsed via _str2list(); if all floatable -> cast to float
    Lists/tuples/sets -> list; if all floatable -> cast to float
    Single number/bool -> [c]; None -> []

    Parameters
    ----------
    c : any
        Configuration value to normalize.
    base : list or None, optional
        Base list for 'all' expansion.

    Returns
    -------
    list
        Normalized list of configuration values.
    """
    # 1) Build list
    if c is None:
        items = []
    elif isinstance(c, (list, tuple, set)):
        items = list(c)
    elif isinstance(c, (int, float, bool)):
        items = [c]
    elif isinstance(c, str):
        items = _str2list(c)
        if not items and c.strip():
            items = [c.strip()]
    else:
        items = [c]

    # 2) 'all' expansion when base provided
    if base is not None:
        for x in items:
            if isinstance(x, str) and x.strip().lower() == "all":
                return sorted(base)

    return items


# Known atlas keywords mapped to their canonical (snake_case) spelling.
# Anything not in here (e.g. custom annotation filenames) is left untouched.
_ATLAS_ALIASES = {
    "benson": "benson",
    "wang": "wang",
    "fs_custom": "fs_custom",
    "fscustom": "fs_custom",
    "full_brain": "full_brain",
    "fullbrain": "full_brain",
    "whole_brain": "full_brain",
    "wholebrain": "full_brain",
}


def _canon_atlas(name):
    """Canonicalize a known atlas keyword case-insensitively.

    Recognized keywords (benson, wang, fs_custom, full_brain and their common
    aliases) map to their canonical spelling regardless of casing; unrecognized
    values such as custom annotation filenames are returned unchanged so their
    original casing is preserved.
    """
    if not isinstance(name, str):
        return name
    return _ATLAS_ALIASES.get(name.strip().lower(), name)


# -------------------------------- orchestrator -------------------------------
def process_subject_session(config: dict, in_root: Path):
    """Orchestrate processing for all tasks/runs across subjects and sessions.

    This is the main processing function that handles the entire pRF analysis
    pipeline, including stimulus resolution, aperture generation, ROI masking,
    and output organization.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing all processing parameters.
    in_root : Path
        Root directory containing BIDS and derivatives folders.
    """
    # Import project modules now that sys.path has been set in main();
    # try package-style first for static analysis, fall back to flat modules for runtime in Docker.
    write_events_mod = _import_any(["link_stimuli", "scripts.link_stimuli"])
    write_events = write_events_mod.write_events
    nii_to_surfNii_mod = _import_any(["nii_to_surfNii", "scripts.nii_to_surfNii"])
    apply_masks_to_run = nii_to_surfNii_mod.apply_masks_to_run
    masked_bold_path = nii_to_surfNii_mod.masked_bold_path
    roi_pack_mod = _import_any(["roi_pack", "scripts.roi_pack"])
    prepare_roi_pack = roi_pack_mod.prepare_roi_pack
    generate_aperture_nii = _import_any(["stim_as_nii", "scripts.stim_as_nii"])
    generate_aperture_nii = generate_aperture_nii.generate_aperture_nii
    resolve_stim_source = _import_any(["stim_loader", "scripts.stim_loader"])
    resolve_stim_source = resolve_stim_source.resolve_stim_source
    get_logger = _import_any(
        ["prfprepare_logging", "scripts.prfprepare_logging"]
    ).get_logger

    # start a context dictionary that holds all relevant variables
    verbose = bool(config.get("verbose", False))
    LOG = get_logger(__file__, verbose=verbose)
    ctx = {"verbose": verbose, "log": LOG}

    cconfig = config.get("config", {})

    # now that logger is available, log imported module names at debug level
    LOG.debug(
        "Imported modules: "
        f"write_events={write_events_mod.__name__}, "
        f"apply_masks_to_run={apply_masks_to_run.__module__}, "
        f"prepare_roi_pack={roi_pack_mod.__name__}, "
        f"generate_aperture_nii={generate_aperture_nii.__module__}, "
        f"resolve_stim_source={resolve_stim_source.__module__}"
    )

    bids_dir = in_root / "BIDS"
    derivatives_dir = in_root / "derivatives"
    fmriprep_dir = in_root / "derivatives" / "fmriprep"

    in_dir = fmriprep_dir / f"analysis-{cconfig.get('fmriprep_analysis')}"
    LOG.debug(f"Input (fMRIPrep) dir: {in_dir}")

    legacy_layout = bool(config.get("fmriprep_legacy_layout", False))
    if legacy_layout:
        fs_dir = in_dir / "freesurfer"
    else:
        fs_dir = in_dir / "sourcedata" / "freesurfer"
    LOG.debug(f"FreeSurfer dir: {fs_dir}")

    if not in_dir.exists():
        raise FileNotFoundError(f"fMRIPrep dir not found: {fmriprep_dir}")

    if not fs_dir.exists():
        raise FileNotFoundError(f"FreeSurfer dir not found: {fs_dir}")

    # Core options
    analysis_space = str(cconfig.get("analysisSpace", "fsnative")).strip().lower()
    average_runs = bool(cconfig.get("average_runs", False))
    output_only_average = bool(config.get("output_only_average", False))
    continue_on_error = bool(config.get("continue_on_error", True))
    use_numImages = bool(config.get("use_numImages", False))
    output_all_frames = bool(cconfig.get("output_all_frames", False))
    ctx["flip_ud"] = bool(cconfig.get("flip_ud", False))
    ctx["flip_lr"] = bool(cconfig.get("flip_lr", False))
    ctx["etcorr"] = bool(config.get("etcorrection", False))
    ctx["force"] = bool(config.get("force", False))

    # check for the analysis_spaces
    if analysis_space not in ("fsnative", "fsaverage", "volume"):
        if analysis_space == "surface":
            analysis_space = "fsnative"
        else:
            raise ValueError(
                f"analysisSpace {analysis_space} not recognized, should be in [fsnative, fsaverage or volume]!"
            )

    out_analysis_name = _get_out_analysis_name(cconfig, derivatives_dir, LOG)

    out_base = derivatives_dir / "prfprepare" / out_analysis_name
    out_base.mkdir(parents=True, exist_ok=True)
    LOG.debug(f"Output base: {out_base}")

    # non-subject-specific things
    atlases = _config2list(
        cconfig.get("atlases", ["wang"]), ["benson", "wang", "fs_custom"]
    )
    # Canonicalize known atlas keywords case-insensitively (custom filenames pass through)
    atlases = [_canon_atlas(a) for a in atlases]

    # full_brain export: write all cortical vertices/voxels to the prfprepare
    # output while still storing wang+benson in all_roi_masks.h5 as an aid for
    # manual visual-area segmentation. Co-store the standard atlases so the h5
    # always carries the atlas information (set rois:"all" for the full aid).
    if "full_brain" in atlases:
        for std in ("wang", "benson"):
            if std not in atlases:
                atlases.append(std)
        LOG.info(
            "full_brain requested: exporting all cortical vertices/voxels and "
            "storing wang+benson atlases in all_roi_masks.h5 as a segmentation aid."
        )

    rois = _config2list(cconfig.get("rois", ["V1"]), ["all"])

    # check if there is the custom.zip, if yes unzip
    if "fs_custom" in atlases:
        LOG.debug("fs_custom requested; checking annotations zip…")
        fsAvgLabelCustom = fs_dir / "fsaverage" / "customLabel"
        fsAvgLabelCustomZip = fsAvgLabelCustom / "custom.zip"

        if not fsAvgLabelCustomZip.exists():
            LOG.warn(f"We could not find a custom.zip in {fsAvgLabelCustom}!")
            LOG.warn("Removing fs_custom atlas from list.")
            atlases.remove("fs_custom")
            custom_annots = []
        else:
            LOG.debug(f"Found custom.zip in {fsAvgLabelCustom}!")
            if not (fsAvgLabelCustom / "DONE").exists():
                LOG.debug("Unzipping custom.zip → customLabel/")
                # Unzip the annotations
                with ZipFile(fsAvgLabelCustomZip, "r") as zipObj:
                    zipObj.extractall(fsAvgLabelCustom)

                # create a check file
                with open(fname := (fsAvgLabelCustom / "DONE"), "a"):
                    os.utime(fname, None)

            # Read all the annotations
            custom_annots = glob(fsAvgLabelCustom / "*.annot")
            LOG.debug(f"Custom annotations: {len(custom_annots)} found")
            atlases.remove("fs_custom")
            atlases += [a.name for a in custom_annots]
    else:
        custom_annots = []

    # get the BIDS layout
    LOG.debug("Discovering BIDS layout (this may take a moment)…")
    layout_indexer = bids.BIDSLayoutIndexer(
        validate=False,
        ignore=["20*/", "sourcedata/", "fmriprep*/", "logs/"],
        index_metadata=False,
    )
    layout = bids.BIDSLayout(
        in_dir,
        indexer=layout_indexer,
    )

    # discover subjects
    bids_subs = layout.get_subjects()
    LOG.debug(f"Found {len(bids_subs)} subjects in layout")
    cfg_subs = _config2list(config.get("subjects", "all"))
    if cfg_subs[0] != "all" and not np.all([a in bids_subs for a in cfg_subs]):
        LOG.warn(f"We did not find given subject {cfg_subs} in BIDS dir!")

    subs = _config2list(cfg_subs, bids_subs)

    pack_args = {
        "analysis_space": analysis_space,
        "atlases": atlases,
        "rois": rois,
        "fs_dir": fs_dir,
        "custom_annots": custom_annots if custom_annots else None,
        "out_base": out_base,
    }
    # (sub, grid) packs already rebuilt in this process
    rebuilt = set()

    def _pack(c, bold_path):
        """Prepare the ROI pack for a run without loading its data.

        Surface packs are validated against geometry and atlases here; the
        BOLD vertex check happens in the masking job. A volume grid comes from
        the NIfTI header. With force, each pack (surface) or grid (volume) is
        rebuilt once instead of once per run: same result, written once.
        """
        bimg = nib.load(str(bold_path)) if analysis_space == "volume" else None
        gid = (
            roi_pack_mod._grid_signature(bimg.shape[:3], bimg.affine)
            if bimg is not None
            else None
        )
        first = (c["sub"], gid) not in rebuilt
        rebuilt.add((c["sub"], gid))
        return prepare_roi_pack(
            ctx={**c, "force": bool(c.get("force")) and first},
            bold_img=bimg,
            **pack_args,
        )

    # masking jobs of the subject being prepared; submitted when it is done
    jobs = []

    def _job_done(job, result):
        # Events of a computed average are written here, in the main process,
        # so the two hemispheres' jobs never write the same file.
        if result["averaged"]:
            c = job["common"]
            _ = write_events(
                ctx=job["avg_ctx"],
                stim=c["stim"],
                stim_name=c["stim_name"],
                timepoints=c["timepoints"],
                output_only_average=False,
                frame_tr=c.get("frame_tr"),
            )

    runner = _MaskRunner(_resolve_n_jobs(config.get("n_jobs", 1)), _job_done)
    # apertures already built in this process (with force)
    apertures_made = set()

    t_start = time.perf_counter()
    try:
        for sub in subs:
            ctx["sub"] = sub

            # discover sessions
            bids_sess = layout.get_sessions(subject=sub)
            LOG.debug(f"Subject {sub}: found {len(bids_sess)} sessions in layout")
            cfg_sess = _config2list(config.get("sessions", "all"))
            if cfg_sess[0] != "all" and not np.all([a in bids_sess for a in cfg_sess]):
                LOG.warn(
                    f"We did not find given session {cfg_sess} for subject {sub} in BIDS dir!"
                )

            sess = _config2list(cfg_sess, bids_sess)

            for ses in sess:
                ctx["ses"] = ses
                LOG.info(f"Subject {sub} | Session {ses or 'NA'}")

                # discover tasks
                bids_tasks = layout.get_tasks(subject=sub, session=ses)
                LOG.debug(
                    f"Subject {sub} | Session {ses or 'NA'}: found {len(bids_tasks)} tasks"
                )
                cfg_tasks = _config2list(config.get("tasks", "all"))
                if cfg_tasks[0] != "all" and not np.all(
                    [a in bids_tasks for a in cfg_tasks]
                ):
                    LOG.warn(
                        f"We did not find given task {cfg_tasks} for subject {sub} in BIDS dir!"
                    )

                tasks = _config2list(cfg_tasks, bids_tasks)

                for task in tasks:
                    ctx["task"] = task

                    # discover hemis
                    bids_hemis = ["l", "r"]
                    cfg_hemis = _config2list(config.get("hemis", "all"))
                    if cfg_hemis[0] != "all" and not np.all(
                        [a in bids_hemis for a in cfg_hemis]
                    ):
                        LOG.warn(
                            f"We did not find given hemisphere {cfg_hemis} for subject {sub} in BIDS dir!"
                        )

                    hemis = _config2list(cfg_hemis, bids_hemis)

                    # discover runs
                    bids_runs = layout.get_runs(subject=sub, session=ses, task=task)
                    LOG.debug(
                        f"Subject {sub} | Session {ses or 'NA'} | Task {task}: found {len(bids_runs)} runs"
                    )
                    cfg_runs = _config2list(config.get("runs", "all"))
                    if cfg_runs[0] != "all" and not np.all(
                        [a in bids_runs for a in cfg_runs]
                    ):
                        LOG.warn(
                            f"We did not find given run {cfg_runs} for subject {sub} in BIDS dir!"
                        )

                    runs = [str(a) for a in _config2list(cfg_runs, bids_runs)]

                    try:
                        LOG.debug("Resolving stimulus and parameters…")
                        # Resolve stimulus+params once
                        t0 = time.perf_counter()
                        stim = resolve_stim_source(
                            ctx=ctx,
                            bids_root=bids_dir,
                            run=runs[0],
                            force_params=_config2list(config.get("force_params", None)),
                        )
                        if stim is None:
                            raise Warning("No stimulus/params for this run; skipping.")
                        LOG.debug(
                            f"Stimulus resolved (kind={getattr(stim, 'kind', 'n/a')}, n_frames={getattr(stim, 'n_frames', 'n/a')}) in {time.perf_counter()-t0:.2f}s"
                        )
                    except Exception as e:
                        LOG.error(
                            f"Skipping task-{task} for sub-{sub} ses-{ses or 'NA'}: {e}"
                        )
                        if continue_on_error:
                            # clean up inner-scope ctx keys before skipping the task
                            ctx.pop("func_out", None)
                            ctx.pop("stim_out", None)
                            continue
                        raise

                    # Output dirs
                    ctx["func_out"] = out_base / f"sub-{sub}" / f"ses-{ses}" / "func"
                    ctx["stim_out"] = out_base / f"sub-{sub}" / "stimuli"
                    bold_base_dir = in_dir / f"sub-{sub}" / f"ses-{ses}" / "func"
                    for d in (ctx["func_out"], ctx["stim_out"]):
                        d.mkdir(parents=True, exist_ok=True)

                    # compute the aperture for the given task
                    LOG.debug("Generating aperture NIfTI…")
                    t0 = time.perf_counter()
                    # The aperture file is per subject and task, shared by all
                    # sessions. With force, build it once per distinct stimulus
                    # instead of rewriting the same file for every session.
                    ap_key = (ctx["stim_out"], _stim_signature(stim))
                    aperture_name, timepoints = generate_aperture_nii(
                        {**ctx, "force": ctx["force"] and ap_key not in apertures_made},
                        stim,
                        use_numImages,
                        output_all_frames,
                    )
                    apertures_made.add(ap_key)
                    # Seconds per aperture frame: the scan TR, or with
                    # output_all_frames the stimulus' own frame time, which
                    # stim_as_nii writes to pixdim[4] (header only, cheap).
                    frame_tr = float(stim.tr)
                    if output_all_frames:
                        try:
                            zoom_t = float(
                                nib.load(str(aperture_name)).header.get_zooms()[3]
                            )
                            if np.isfinite(zoom_t) and zoom_t > 0:
                                frame_tr = zoom_t
                        except Exception as e:
                            LOG.warn(
                                f"Could not read frame time from {aperture_name} ({e}); using TR {frame_tr}s"
                            )
                    LOG.debug(
                        f"Aperture ready: {aperture_name} (T={timepoints}, frame={frame_tr:g}s) in {time.perf_counter()-t0:.2f}s"
                    )

                    # Prepare phase. Everything that writes shared files (events,
                    # the subject's ROI pack and its meta JSON) runs here, in order,
                    # without opening the BOLD. Loading and masking the BOLD is
                    # queued as jobs and run after all subjects are prepared.
                    task_ctx = dict(ctx)
                    common = {
                        "stim": types.SimpleNamespace(
                            tr=stim.tr, start_scan=getattr(stim, "start_scan", 0.0)
                        ),
                        "stim_name": aperture_name.name,
                        "timepoints": timepoints,
                        "frame_tr": frame_tr,
                        "output_only_average": output_only_average,
                        "continue_on_error": continue_on_error,
                        "pack_args": pack_args,
                    }
                    # volume runs, both hemis in one job so the NIfTI is read once
                    volume_items = {run: [] for run in runs}

                    for hemi in hemis:
                        hctx = {**task_ctx, "hemi": hemi}

                        do_average = average_runs and len(runs) >= 2
                        avg_ctx = {**hctx, "run": "".join(runs) + "avg"}
                        avg_done = (
                            do_average
                            and not ctx["force"]
                            and masked_bold_path(avg_ctx).exists()
                        )
                        items = []
                        for run in runs:
                            rctx = {**hctx, "run": run}
                            LOG.info(
                                f"Subject {sub} | Session {ses or 'NA'} | Task {task} | Run {run} | Hemi {hemi.upper() or 'NA'}"
                            )
                            try:
                                bold_path = _bold_path_for(
                                    bold_base_dir,
                                    analysis_space,
                                    sub,
                                    ses,
                                    task,
                                    run,
                                    hemi,
                                )

                                # check if bold_path exists
                                if not bold_path.exists():
                                    LOG.error(
                                        f"BOLD file not found: {bold_path}, skipping."
                                    )
                                    continue

                                # write events.tsv for each run
                                LOG.debug("Writing events.tsv…")
                                _ = write_events(
                                    ctx=rctx,
                                    stim=stim,
                                    stim_name=aperture_name.name,
                                    timepoints=timepoints,
                                    output_only_average=output_only_average,
                                    frame_tr=frame_tr,
                                )

                                # Prepare/load ROI pack for this specific run/grid
                                LOG.debug("Preparing ROI cache for this run…")
                                t0 = time.perf_counter()
                                roi_pack = _pack(rctx, bold_path)
                                LOG.debug(
                                    f"ROI cache ready (key={getattr(roi_pack, 'key', 'n/a')}, file={getattr(roi_pack, 'h5_path', 'n/a')}) in {time.perf_counter()-t0:.2f}s"
                                )

                                # Already done: no job, so the BOLD is never loaded,
                                # which for GIFTI means parsing the whole file. With
                                # output_only_average there is no per-run output, so
                                # the run is done once the average is. Volume is not
                                # skipped: nib.load only reads its header, so there
                                # is little to save.
                                done = (
                                    analysis_space != "volume"
                                    and not ctx["force"]
                                    and (
                                        masked_bold_path(rctx).exists()
                                        or (output_only_average and avg_done)
                                    )
                                )
                                if done:
                                    LOG.info(
                                        f"Output exists, skipping run-{run} hemi-{hemi.upper()}"
                                    )
                                items.append(
                                    {"ctx": rctx, "bold_path": bold_path, "done": done}
                                )

                            except Exception as e:
                                LOG.error(
                                    f"Failed run: sub-{sub} ses-{ses or 'NA'} task-{task} run-{run} ({e})"
                                )
                                if continue_on_error:
                                    continue
                                raise

                        # Averaging across runs (optional)
                        if do_average and avg_done:
                            if len(items) >= 2:
                                LOG.info(
                                    f"Averaged masked BOLD exists, skipping run-{avg_ctx['run']} hemi-{hemi.upper()}"
                                )
                                try:
                                    _ = write_events(
                                        ctx=avg_ctx,
                                        stim=stim,
                                        stim_name=aperture_name.name,
                                        timepoints=timepoints,
                                        output_only_average=False,
                                        frame_tr=frame_tr,
                                    )
                                    # Averaging needs all runs on one grid, so the
                                    # first run stands in for the average (header
                                    # only for volume, recording the grid provenance)
                                    _pack(avg_ctx, items[0]["bold_path"])
                                except Exception as e:
                                    LOG.error(
                                        f"Averaging failed for sub-{sub} ses-{ses or 'NA'} task-{task} ({e})"
                                    )
                                    if not continue_on_error:
                                        raise
                        elif do_average:
                            # One job for the hemisphere: the runs are masked in
                            # order and their GIFTIs reused for the average.
                            if len(items) >= 2:
                                try:
                                    # provenance for the average, written here
                                    # rather than concurrently from a job
                                    _pack(avg_ctx, items[0]["bold_path"])
                                except Exception as e:
                                    LOG.error(
                                        f"Averaging failed for sub-{sub} ses-{ses or 'NA'} task-{task} ({e})"
                                    )
                                    if not continue_on_error:
                                        raise
                            jobs.append(
                                {"common": common, "items": items, "avg_ctx": avg_ctx}
                            )
                            continue

                        for it in items:
                            if it["done"]:
                                continue
                            if analysis_space == "volume":
                                volume_items[it["ctx"]["run"]].append(it)
                            else:
                                jobs.append(
                                    {"common": common, "items": [it], "avg_ctx": None}
                                )

                    for run in runs:
                        if volume_items[run]:
                            jobs.append(
                                {
                                    "common": common,
                                    "items": volume_items[run],
                                    "avg_ctx": None,
                                }
                            )
                    ctx.pop("task", None)
                    ctx.pop("func_out", None)
                    ctx.pop("stim_out", None)
                ctx.pop("ses", None)
            if jobs:
                LOG.info(
                    f"Subject {sub}: queued {len(jobs)} masking jobs "
                    f"({runner.n_jobs} worker(s))"
                )
            runner.submit(jobs)
            jobs = []
            ctx.pop("sub", None)

        # Wait for the remaining masking jobs
        if runner.pending:
            LOG.info(f"Waiting for {len(runner.pending)} masking jobs…")
        runner.collect(wait=True)
        if runner.n_submitted:
            LOG.info(
                f"All {runner.n_submitted} masking jobs done "
                f"({time.perf_counter()-t_start:.1f}s since start)"
            )
    except BaseException:
        runner.close(cancel=True)
        raise
    runner.close()

    # copy the dataset_description from fmriprep
    LOG.debug("Copying dataset_description.json to output…")
    sp.call(f'cp {bids_dir / "dataset_description.json"} {out_base}', shell=True)

    # if defined write link for custom output folder name
    if config.get("custom_output_name", "") != "":
        try:
            os.chdir(out_base / "..")
            if not (
                out_base / ".." / f"analysis-{config.get('custom_output_name', '')}"
            ).is_symlink():
                os.symlink(
                    out_analysis_name,
                    f"analysis-{config.get('custom_output_name', '')}",
                )
                LOG.debug("Created custom_output_name symlink")
        except:
            LOG.warn(
                f"Could not create the custom_output_name analysis-{config.get('custom_output_name', '')} link!"
            )


# ---------------------------------- CLI -------------------------------------


def main():
    """Main entry point for the pRF pipeline orchestrator.

    Parses command line arguments, loads configuration, and orchestrates
    the pRF analysis pipeline for all subjects, sessions, tasks, and runs.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    SystemExit
        If configuration loading fails.
    """
    ap = argparse.ArgumentParser(description="pRF pipeline orchestrator (clean run.py)")
    ap.add_argument(
        "--config", type=str, default="config.json", help="Path to config JSON"
    )
    ap.add_argument(
        "--force", action="store_true", help="Force overwrite existing files"
    )
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    ap.add_argument(
        "--n_jobs",
        type=str,
        default=None,
        help="Parallel masking workers (overrides config n_jobs; 0/auto = all CPUs)",
    )
    args = ap.parse_args()

    # get all needed functions
    data_dir = Path("/base/data")
    scripts_dir = Path("/base/scripts")
    sys.path.insert(0, str(data_dir))
    sys.path.insert(0, str(scripts_dir))

    # load the config file
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    config = json.loads(cfg_path.read_text())

    # update the config with the default config
    config = _update_standard_config(config, scripts_dir)

    # if force or verbose in parser is True, overwrite the config
    if args.force:
        config["force"] = True
    if args.verbose:
        config["verbose"] = True
    if args.n_jobs is not None:
        config["n_jobs"] = args.n_jobs

    if config.get("verbose"):
        # Make verbose available to modules that use env fallback
        os.environ["PRFPREPARE_VERBOSE"] = "1"

        _get_logger = _import_any(
            ["prfprepare_logging", "scripts.prfprepare_logging"]
        ).get_logger
        LOG = _get_logger(__file__, verbose=True)
    LOG.debug(f"Merged configuration:\n{json.dumps(config, indent=4)}")

    # run the orchestrator
    process_subject_session(config, data_dir)

    os.chdir(path.expanduser("~"))


if __name__ == "__main__":
    main()

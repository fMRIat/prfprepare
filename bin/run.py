# run.py

import argparse
import collections
import importlib
import json
import os
import subprocess as sp
import sys
import time
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


def _get_bold_T_and_TR(p: Path, LOG=None) -> tuple[int | None, float | None]:
    """Return (#timepoints, TR seconds) from a BOLD file and its sidecar.

    T is derived from the image (nibabel) where possible.
    TR is read from sidecar JSON, with optional fallback to header for NIfTI.

    Parameters
    ----------
    p : Path
        Path to the BOLD file (NIfTI or GIFTI).
    LOG : logger, optional
        Logger instance for debug/warning messages.

    Returns
    -------
    tuple[int or None, float or None]
        Tuple of (timepoints, TR_seconds). Either can be None if not available.
    """
    try:
        img = nib.load(str(p))
        # T from data
        T = None
        try:
            data = None
            # For GIFTI, use agg_data; for NIfTI shape[-1]
            if hasattr(img, "agg_data"):
                data = img.agg_data()
                T = int(data.shape[-1]) if data is not None else None
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
        if float(stim.tr) != float(file_TR):
            LOG.warn(
                f"Stimulus TR ({float(stim.tr):.4f}s) does not match file TR ({float(file_TR):.4f}s)."
            )
            LOG.warn(f"Using stimulus TR ({float(stim.tr):.4f}s)!")
    except Exception:
        # keep calm if stim.tr is read-only or non-float
        pass


def _log_timepoints_mismatch(
    file_T: int | None, aperture_T: int, where: str, LOG
) -> None:
    """Log an error if file_T is present and differs from aperture_T.

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
    """
    if file_T is not None and int(file_T) != int(aperture_T):
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


def _average_runs_bold(bold_paths, LOG) -> Path:
    """Crop to shortest T and compute voxel-wise mean across multiple BOLD runs.

    Parameters
    ----------
    bold_paths : list[Path]
        List of paths to BOLD files to average.
    LOG : logger
        Logger instance for info messages.

    Returns
    -------
    Path
        Path to the averaged BOLD file.

    Raises
    ------
    ValueError
        If BOLD files have different spatial dimensions.
    """
    # Load all inputs
    imgs = [nib.load(str(p)) for p in bold_paths]

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
        # Validate spatial dims and crop to shortest T
        datas = []
        spatial_shape = None
        Ts = []
        for img in imgs:
            d = img.get_fdata(dtype=np.float32)
            if d.ndim != 4:
                raise ValueError(f"Expected 4D BOLD NIfTI; got shape {d.shape}")
            if spatial_shape is None:
                spatial_shape = d.shape[:3]
            elif d.shape[:3] != spatial_shape:
                raise ValueError(
                    "All BOLD files must have the same spatial dimensions for averaging"
                )
            datas.append(d)
            Ts.append(d.shape[-1])

        min_T = min(Ts)
        if min_T < max(Ts):
            LOG.debug(f"Cropping runs to shortest T={min_T} for averaging")
        datas = [d[..., :min_T] for d in datas]

        mean = np.mean(np.stack(datas, axis=0), axis=0)
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
    apply_masks_to_run = _import_any(["nii_to_surfNii", "scripts.nii_to_surfNii"])
    apply_masks_to_run = apply_masks_to_run.apply_masks_to_run
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
    analysis_space = cconfig.get("analysisSpace", "fsnative")
    average_runs = bool(cconfig.get("average_runs", False))
    output_only_average = bool(config.get("output_only_average", False))
    use_numImages = bool(config.get("use_numImages", False))
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
                except Exception:
                    raise

                # Output dirs
                ctx["func_out"] = out_base / f"sub-{sub}" / f"ses-{ses}" / "func"
                ctx["stim_out"] = out_base / f"sub-{sub}" / "stimuli"
                bold_base_dir = in_dir / f"sub-{sub}" / f"ses-{ses}" / "func"
                for d in (ctx["func_out"], ctx["stim_out"]):
                    d.mkdir(parents=True, exist_ok=True)

                # Prepare a sample BOLD path (first run/hemi) to validate TR vs stim and T vs apertures
                sample_hemi = None
                sample_run = None
                if len(hemis) > 0 and len(runs) > 0:
                    sample_hemi = hemis[0]
                    sample_run = runs[0]
                sample_bold_path = None
                if sample_hemi is not None and sample_run is not None:
                    sample_bold_path = _bold_path_for(
                        bold_base_dir,
                        analysis_space,
                        sub,
                        ses,
                        task,
                        sample_run,
                        sample_hemi,
                    )

                # If sample exists, compare stim TR vs file TR
                if sample_bold_path and sample_bold_path.exists():
                    _, file_TR = _get_bold_T_and_TR(sample_bold_path, LOG=LOG)
                    _log_stim_file_tr_mismatch(stim, file_TR, LOG)
                else:
                    LOG.debug(
                        "Sample BOLD file not found; will validate per-run instead."
                    )

                # compute the aperture for the given task
                LOG.debug("Generating aperture NIfTI…")
                t0 = time.perf_counter()
                aperture_name, timepoints = generate_aperture_nii(
                    ctx, stim, use_numImages
                )
                LOG.debug(
                    f"Aperture ready: {aperture_name} (T={timepoints}) in {time.perf_counter()-t0:.2f}s"
                )

                # If sample exists, compare aperture T vs file T
                if sample_bold_path and sample_bold_path.exists():
                    file_T, _ = _get_bold_T_and_TR(sample_bold_path, LOG=LOG)
                    _log_timepoints_mismatch(
                        file_T, timepoints, str(sample_bold_path), LOG
                    )

                for hemi in hemis:
                    ctx["hemi"] = hemi

                    all_run_paths = []
                    for run in runs:
                        ctx["run"] = run
                        LOG.info(
                            f"Subject {sub} | Session {ses or 'NA'} | Task {task} | Run {run} | Hemi {hemi.upper() or 'NA'}"
                        )
                        try:
                            ctx["bold_path"] = _bold_path_for(
                                bold_base_dir,
                                analysis_space,
                                sub,
                                ses,
                                task,
                                run,
                                hemi,
                            )

                            # check if bold_path exists
                            if not ctx["bold_path"].exists():
                                LOG.error(
                                    f"BOLD file not found: {ctx['bold_path']}, skipping."
                                )
                                continue

                            # Per-run check: compare aperture T vs file T
                            file_T, file_TR = _get_bold_T_and_TR(
                                ctx["bold_path"], LOG=LOG
                            )
                            _log_stim_file_tr_mismatch(stim, file_TR, LOG)
                            _log_timepoints_mismatch(
                                file_T, timepoints, str(ctx["bold_path"]), LOG
                            )

                            # write events.tsv for each run
                            LOG.debug("Writing events.tsv…")
                            _ = write_events(
                                ctx=ctx,
                                stim=stim,
                                stim_name=aperture_name.name,
                                timepoints=timepoints,
                                output_only_average=output_only_average,
                            )

                            # Prepare/load ROI pack for this specific run/grid
                            LOG.debug("Preparing ROI cache for this run…")
                            t0 = time.perf_counter()
                            roi_pack = prepare_roi_pack(
                                ctx=ctx,
                                analysis_space=analysis_space,
                                atlases=atlases,
                                rois=rois,
                                fs_dir=fs_dir,
                                custom_annots=custom_annots if custom_annots else None,
                                out_base=out_base,
                                bold_img=(
                                    ctx.get("bold_path")
                                    if analysis_space == "volume"
                                    else None
                                ),
                            )
                            LOG.debug(
                                f"ROI cache ready (key={getattr(roi_pack, 'key', 'n/a')}, file={getattr(roi_pack, 'h5_path', 'n/a')}) in {time.perf_counter()-t0:.2f}s"
                            )

                            # ROI masking using the appropriate grid
                            LOG.debug("Applying ROI masks to run…")
                            t0 = time.perf_counter()
                            apply_masks_to_run(
                                ctx=ctx,
                                stim=stim,
                                roi_pack=roi_pack,
                                output_only_average=output_only_average,
                            )
                            LOG.debug(
                                f"Masking complete in {time.perf_counter()-t0:.2f}s"
                            )

                            all_run_paths.append(ctx["bold_path"])

                            del ctx["bold_path"]

                        except Exception as e:
                            LOG.error(
                                f"Failed run: sub-{sub} ses-{ses or 'NA'} task-{task} run-{run} ({e})"
                            )
                            raise

                        del ctx["run"]

                    # Averaging across runs (optional)
                    if average_runs and len(all_run_paths) >= 2:
                        ctx["run"] = "".join(runs) + "avg"
                        LOG.info(
                            f"\tAveraging {len(all_run_paths)} runs: {', '.join(runs)}"
                        )

                        try:

                            LOG.debug("Computing average BOLD across runs…")
                            t0 = time.perf_counter()
                            ctx["bold_path"] = _average_runs_bold(all_run_paths, LOG)
                            LOG.debug(
                                f"Average computed in {time.perf_counter()-t0:.2f}s"
                            )

                            # write events.tsv for each run
                            LOG.debug("Writing events.tsv for averaged run…")
                            _ = write_events(
                                ctx=ctx,
                                stim=stim,
                                stim_name=aperture_name.name,
                                timepoints=timepoints,
                                output_only_average=False,
                            )

                            # Prepare/load ROI pack for the averaged run/grid
                            LOG.debug("Preparing ROI cache for averaged run…")
                            t0 = time.perf_counter()
                            roi_pack = prepare_roi_pack(
                                ctx=ctx,
                                analysis_space=analysis_space,
                                atlases=atlases,
                                rois=rois,
                                fs_dir=fs_dir,
                                custom_annots=custom_annots if custom_annots else None,
                                out_base=out_base,
                                bold_img=(
                                    ctx["bold_path"]
                                    if analysis_space == "volume"
                                    else None
                                ),
                            )
                            LOG.debug(
                                f"ROI cache ready (key={getattr(roi_pack, 'key', 'n/a')}, file={getattr(roi_pack, 'h5_path', 'n/a')}) in {time.perf_counter()-t0:.2f}s"
                            )

                            # ROI masking
                            LOG.debug("Applying ROI masks to averaged run…")
                            t0 = time.perf_counter()
                            apply_masks_to_run(
                                ctx=ctx,
                                stim=stim,
                                roi_pack=roi_pack,
                                output_only_average=False,
                            )
                            LOG.debug(
                                f"Masking (avg) complete in {time.perf_counter()-t0:.2f}s"
                            )

                        except Exception as e:
                            LOG.error(
                                f"Averaging failed for sub-{sub} ses-{ses or 'NA'} task-{task} ({e})"
                            )
                            raise

                        del ctx["run"]
                    del ctx["hemi"]
                del ctx["task"]
                del ctx["func_out"]
                del ctx["stim_out"]
            del ctx["ses"]
        del ctx["sub"]

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

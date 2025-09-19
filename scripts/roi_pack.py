# roi_pack.py

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Iterable, Optional

import h5py
import neuropythy as ny
import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
from prfprepare_logging import get_logger
from roipack import RoiPack

# ------------------------------ Constants ------------------------------


# Custom exceptions
class AtlasNotFoundError(FileNotFoundError):
    """Raised when an atlas file cannot be found."""

    pass


class UnsupportedAtlasError(ValueError):
    """Raised when an unsupported atlas is requested."""

    pass


class GridMismatchError(ValueError):
    """Raised when volume grids have mismatched shapes."""

    pass


# Atlas names
ATLAS_WANG = "wang"
ATLAS_BENSON = "benson"
ATLAS_FULLBRAIN = "fullBrain"
ATLAS_FS_CUSTOM = "fs_custom"

# Special ROI labels
ROI_UNKNOWN = "Unknown"
ROI_ALL = "all"

# Analysis spaces
SPACE_FSNATIVE = "fsnative"
SPACE_FSAVERAGE = "fsaverage"
SPACE_VOLUME = "volume"

# Hemisphere labels
HEMI_LEFT = "l"
HEMI_RIGHT = "r"
HEMI_BOTH = "both"

# Atlas file mappings
ATLAS_FILES = {
    ATLAS_WANG: "wang15_mplbl.mgz",
    ATLAS_BENSON: "benson14_varea.mgz",
}

ATLAS_VOLUME_FILES = {
    ATLAS_WANG: "wang15_mplbl.mgz",
    ATLAS_BENSON: "benson14_varea.mgz",
}

# ------------------------------ validation helpers ------------------------------


def _validate_inputs(
    analysis_space: str, atlases: Iterable[str], rois: Iterable[str]
) -> None:
    """Validate input parameters for prepare_roi_pack."""
    valid_spaces = {SPACE_FSNATIVE, SPACE_FSAVERAGE, SPACE_VOLUME}
    if analysis_space not in valid_spaces:
        raise ValueError(
            f"Invalid analysis_space '{analysis_space}'. Must be one of: {valid_spaces}"
        )

    if not atlases:
        raise ValueError("At least one atlas must be specified")

    if not rois:
        raise ValueError("At least one ROI must be specified")


# ------------------------------ meta / key utils ------------------------------


def _canonical_meta(
    sub: str, analysis_space: str, atlases, rois, fs_dir: Path | str
) -> dict:
    """Canonicalize meta for a stable key (order-insensitive for atlases/rois)."""
    return {
        "sub": sub,
        "analysis_space": str(analysis_space),
        "atlases": sorted(list(atlases or [])),
        "rois": sorted(list(rois or [])),
        "fs_dir": str(fs_dir),
    }


def _meta_digest(meta_dict: dict) -> str:
    """SHA-1 over canonical JSON (stable across dict order)."""
    b = json.dumps(meta_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(b).hexdigest()[:12]


def _grid_signature(shape, affine, tol=1e-5):
    """
    Generate a unique signature for a 3D grid based on shape and affine matrix.

    Parameters
    ----------
    shape : tuple
        3D shape of the grid.
    affine : array-like
        4x4 affine transformation matrix.
    tol : float, optional
        Tolerance for rounding (default: 1e-5).

    Returns
    -------
    str
        12-character hexadecimal signature.
    """
    A = np.asarray(affine, float).round(6)
    s = np.asarray(shape, int)
    h = hashlib.sha1()
    h.update(s.tobytes())
    h.update(A.tobytes())
    return h.hexdigest()[:12]


# --------------------------- HDF5 save/load helpers ---------------------------


def _lock_path_for_grid(h5_path: Path, gid: str) -> Path:
    """Return a lock file path for a specific grid ID next to the HDF5."""
    return h5_path.parent / f"{h5_path.name}.{gid}.lock"


def _acquire_lock(lock_path: Path, timeout: float = 60.0, poll: float = 0.5, LOG=None):
    """Acquire an exclusive lock by atomically creating a lock file.

    Retries until timeout; raises TimeoutError if not acquired.
    """
    start = time.time()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            if LOG:
                LOG.debug(f"Acquired lock: {lock_path}")
            return
        except FileExistsError:
            if time.time() - start > timeout:
                raise TimeoutError(f"Timeout acquiring lock {lock_path}")
            if LOG:
                LOG.debug(f"Lock busy, waiting: {lock_path}")
            time.sleep(poll)


def _release_lock(lock_path: Path, LOG=None):
    """
    Release an exclusive lock by removing the lock file.

    Parameters
    ----------
    lock_path : Path
        Path to the lock file to remove.
    LOG : logger, optional
        Logger instance for debug messages.
    """
    try:
        os.unlink(str(lock_path))
        if LOG:
            LOG.debug(f"Released lock: {lock_path}")
    except FileNotFoundError:
        pass


def _write_union_membership(f: h5py.File, rois: dict, base_group: str | None = None):
    """
    Build masked space membership structures for both volume (3D) and surface (1D) ROI masks.

    Inputs
    ------
    f : h5py.File
        Open file handle to write into.
    rois : dict
        {(hemi, atlas, roi): ndarray}
        - Surface ROI masks are 1D (per-vertex per hemi)
        - Volume ROI masks are 3D (grid-aligned)
    base_group : str | None
        Path relative to file root under which to write. For volume, this is
        typically 'grids/<gid>'. For surface, it's None (root).

    Writes
    ------
    Volume: <base_group>/masked_space/<atlas>/<roi>/<hemi>/indices
    Surface: <base_group or root>/masked_space/<atlas>/<roi>/<hemi>/indices

    Notes
    -----
    - Each ROI gets its own dataset with indices into the masked space
    - Masked space is per-hemisphere to accommodate differing vertex counts
    - Also stores flat_index dataset per hemisphere with all masked space positions
    """
    # Resolve base path (grid group for volume, or root for surface)
    gbase = f if base_group is None else f.require_group(base_group)

    # Split surface (1D) and volume (3D) ROI masks by hemisphere
    surf_items = {}
    vol_items = {}
    for (h, a, r), m in rois.items():
        arr = np.asarray(m, bool)
        if arr.ndim == 1:
            surf_items.setdefault(h, []).append(((h, a, r), arr))
        elif arr.ndim == 3:
            vol_items.setdefault(h, []).append(((h, a, r), arr))

    # Helper to build per-hemi masked space structure
    def _build_masked_space_structure(parent, hemi_label: str, items, grid_order: str):
        if not items:
            return

        # deterministic ordering
        items.sort(key=lambda x: (x[0][1], x[0][2], x[0][0]))

        # shape consistency (for volume)
        if grid_order == "C":
            shape0 = items[0][1].shape
            for _, m in items:
                if m.shape != shape0:
                    raise GridMismatchError(
                        f"Volume shape mismatch for hemi {hemi_label}: {m.shape} vs {shape0}"
                    )

        # Compute masked space over this hemisphere
        stack = np.stack([m for _, m in items], axis=0)
        masked_bool = np.any(stack, axis=0).ravel()
        masked_idx = np.flatnonzero(masked_bool).astype(np.int32)

        # Create mapping from global indices to masked space indices
        g2m = np.full(masked_bool.size, -1, np.int32)
        g2m[masked_idx] = np.arange(masked_idx.size, dtype=np.int32)

        # Create masked space group for this hemisphere
        masked_hemi_group = parent.require_group(f"masked_space")

        # Store the flat_index for this hemisphere (all masked space positions)
        if f"flat_index_{hemi_label}" in masked_hemi_group:
            del masked_hemi_group[f"flat_index_{hemi_label}"]
        masked_hemi_group.create_dataset(
            f"flat_index_{hemi_label}",
            data=masked_idx,
            compression="gzip",
            shuffle=True,
            fletcher32=True,
        )
        masked_hemi_group[f"flat_index_{hemi_label}"].attrs["grid_order"] = grid_order

        # For each ROI, store indices into masked space
        for (h, a, r), mask in items:
            # Get positions in masked space for this ROI
            roi_global_indices = np.flatnonzero(mask.ravel())
            roi_masked_indices = g2m[roi_global_indices]
            roi_masked_indices = roi_masked_indices[roi_masked_indices != -1].astype(
                np.int32
            )

            # Create hierarchical structure: masked_space/atlas/roi/hemi
            atlas_group = masked_hemi_group.require_group(a)
            roi_group = atlas_group.require_group(r)

            # Store indices for this (atlas, roi, hemi) combination
            if h in roi_group:
                del roi_group[h]

            dset = roi_group.create_dataset(
                h,
                data=roi_masked_indices,
                compression="gzip",
                shuffle=True,
                fletcher32=True,
            )
            dset.attrs["atlas"] = a
            dset.attrs["roi"] = r
            dset.attrs["hemi"] = h
            dset.attrs["kind"] = "masked_space_indices"
            dset.attrs["dtype"] = "int32"
            dset.attrs["grid_order"] = grid_order

    # Process volume items (handle 'both' masks by duplicating into l/r)
    if vol_items:
        # If a 'both' mask exists (e.g., fullBrain), duplicate into l and r sets
        both_masks = vol_items.get(HEMI_BOTH, [])
        if both_masks:
            for hemi_label in (HEMI_LEFT, HEMI_RIGHT):
                if hemi_label not in vol_items:
                    vol_items[hemi_label] = []
                # duplicate tuples with hemi replaced
                for (_, a, r), arr in both_masks:
                    vol_items[hemi_label].append(((hemi_label, a, r), arr))

        for hemi_label in (HEMI_LEFT, HEMI_RIGHT):
            _build_masked_space_structure(
                gbase, hemi_label, vol_items.get(hemi_label, []), grid_order="C"
            )

    # Process surface items
    if surf_items:
        for hemi_label in (HEMI_LEFT, HEMI_RIGHT):
            _build_masked_space_structure(
                gbase, hemi_label, surf_items.get(hemi_label, []), grid_order="vertex"
            )


def _save_rois_h5(
    h5_path: Path,
    rois: dict,
    meta: dict,
    key: str,
    base_group: str | None = None,
) -> None:
    """
    Save ROI masks under /original_space/<atlas>/<roi>/<hemi>.
    Booleans are stored as uint8; converted back on read.
    """
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(h5_path), "a") as f:
        # file-level attrs
        if "schema" not in f.attrs:
            f.attrs["schema"] = "roi_pack_tree_v2"
        f.attrs["key"] = key
        f.attrs["meta_json"] = json.dumps(meta)

        # resolve base path to write into
        gbase = f if base_group is None else f.require_group(base_group)
        g_rois_root = gbase.require_group("original_space")

        # write datasets
        for (hemi, atlas, roi), arr in sorted(rois.items(), key=lambda kv: kv[0]):
            a = np.asarray(arr)
            grp = g_rois_root.require_group(atlas).require_group(roi)
            if hemi in grp:
                del grp[hemi]

            # Store flat indices of True values instead of dense mask
            full_shape = a.shape
            idx = np.flatnonzero(a.ravel().astype(bool)).astype(np.int32)
            dset = grp.create_dataset(
                hemi,
                data=idx,
                compression="gzip",
                shuffle=True,
                fletcher32=True,
            )
            dset.attrs["atlas"] = atlas
            dset.attrs["roi"] = roi
            dset.attrs["hemi"] = hemi
            dset.attrs["kind"] = "index"
            dset.attrs["dtype"] = "int32"
            dset.attrs["full_shape"] = np.asarray(full_shape, dtype=np.int32)
            dset.attrs["order"] = "C"
            dset.attrs["space_hint"] = "surface" if a.ndim == 1 else "volume"

        # build a flat index for fast listing
        rows = []
        g_rois = g_rois_root
        if g_rois is not None:
            for atlas in g_rois:
                g_atlas = g_rois[atlas]
                for roi in g_atlas:
                    g_roi = g_atlas[roi]
                    for hemi in g_roi:
                        rows.append(
                            (atlas, roi, hemi, f"{g_rois.name}/{atlas}/{roi}/{hemi}")
                        )
        if rows:
            dt = h5py.string_dtype(encoding="utf-8")
            g_index = gbase.require_group("index")
            for name in ("atlas", "roi", "hemi", "path"):
                if name in g_index:
                    del g_index[name]
            g_index.create_dataset(
                "atlas",
                data=[r[0] for r in rows],
                dtype=dt,
            )
            g_index.create_dataset(
                "roi",
                data=[r[1] for r in rows],
                dtype=dt,
            )
            g_index.create_dataset(
                "hemi",
                data=[r[2] for r in rows],
                dtype=dt,
            )
            g_index.create_dataset(
                "path",
                data=[r[3] for r in rows],
                dtype=dt,
            )

        # build info for masked space membership
        _write_union_membership(f, rois, base_group=base_group)


# ------------------------------- label dictionaries ---------------------------


def _wang_labels() -> dict:
    """
    Get Wang et al. (2015) visual area labels mapping.

    Returns
    -------
    dict
        Mapping from area names to integer labels.
    """
    names = [
        "Unknown",
        "V1v",
        "V1d",
        "V2v",
        "V2d",
        "V3v",
        "V3d",
        "hV4",
        "VO1",
        "VO2",
        "PHC1",
        "PHC2",
        "V3a",
        "V3b",
        "LO1",
        "LO2",
        "TO1",
        "TO2",
        "IPS0",
        "IPS1",
        "IPS2",
        "IPS3",
        "IPS4",
        "IPS5",
        "SPL1",
        "hFEF",
    ]
    return {names[k]: k for k in range(len(names))}


def _benson_labels(hemi: str) -> dict:
    """
    Get Benson et al. (2014) visual area labels mapping for a hemisphere.

    Parameters
    ----------
    hemi : str
        Hemisphere identifier ('l' or 'r').

    Returns
    -------
    dict
        Mapping from area names to integer labels.

    Raises
    ------
    RuntimeError
        If neuropythy is not available.
    """
    if ny is None:
        raise RuntimeError("neuropythy is required to derive Benson labels")
    mdl = ny.vision.retinotopy_model("benson17", f"{hemi}h")
    areaLabels = dict(mdl.area_id_to_name)  # id -> name
    return {areaLabels[k]: k for k in areaLabels}  # name -> id


# -------------------------------- atlas helpers --------------------------------


def _build_atlas_path(
    fs_dir: Path | str, sub: str, hemi: str, atlas: str, analysis_space: str
) -> Path:
    """Build file path for surface atlas files."""
    sub_path = f"sub-{sub}" if analysis_space == SPACE_FSNATIVE else "fsaverage"
    atlas_file = ATLAS_FILES[atlas]
    return Path(fs_dir) / sub_path / "surf" / f"{hemi}h.{atlas_file}"


def _load_fullbrain_mask(
    fs_dir: Path | str, sub: str, hemi: str, analysis_space: str
) -> dict:
    """Load fullbrain mask by inferring vertex count from existing atlas."""
    fs_dir = Path(fs_dir)
    sub_path = f"sub-{sub}" if analysis_space == SPACE_FSNATIVE else "fsaverage"

    # Try wang atlas first, then benson
    for atlas in [ATLAS_WANG, ATLAS_BENSON]:
        areas_path = fs_dir / sub_path / "surf" / f"{hemi}h.{ATLAS_FILES[atlas]}"
        if areas_path.exists():
            n = int(nib.load(str(areas_path)).get_fdata().squeeze().shape[0])
            return {(hemi, ATLAS_FULLBRAIN, ATLAS_FULLBRAIN): np.ones((n,), dtype=bool)}

    raise AtlasNotFoundError(
        f"Could not infer vertex count from any atlas for subject {sub}, hemisphere {hemi}"
    )


def _load_atlas_data_and_labels(
    fs_dir: Path | str, sub: str, hemi: str, atlas: str, analysis_space: str
) -> tuple[np.ndarray, dict]:
    """Load atlas data and corresponding labels."""
    atlas_path = _build_atlas_path(fs_dir, sub, hemi, atlas, analysis_space)
    if not atlas_path.exists():
        raise AtlasNotFoundError(f"{atlas.title()} atlas missing: {atlas_path}")

    areas = nib.load(str(atlas_path)).get_fdata().squeeze()

    if atlas == ATLAS_BENSON:
        labels = _benson_labels(hemi)
    elif atlas == ATLAS_WANG:
        labels = _wang_labels()
    else:
        raise UnsupportedAtlasError(f"Unknown atlas: {atlas}")

    return areas, labels


def _load_custom_atlas(
    fs_dir: Path | str, sub: str, hemi: str, atlas: str, analysis_space: str
) -> tuple[np.ndarray, dict, str]:
    """Load custom FreeSurfer annotation atlas."""
    if analysis_space == SPACE_VOLUME:
        raise UnsupportedAtlasError("Custom atlas not supported in volume space")
    if f"{hemi}h." not in atlas:
        raise UnsupportedAtlasError(f"Custom atlas {atlas} not for hemisphere {hemi}")

    annot_path = Path(fs_dir) / f"sub-{sub}" / "customLabel" / atlas
    if not annot_path.exists():
        raise AtlasNotFoundError(f"Custom atlas missing: {annot_path}")

    a, c, l = nib.freesurfer.io.read_annot(str(annot_path))
    areas = a + 1
    area_labels = {ROI_UNKNOWN: 0} | {
        l[k].decode("utf-8"): k + 1 for k in range(len(l))
    }
    atlas_name = atlas.split(".")[1]  # Extract atlas name from filename

    return areas, area_labels, atlas_name


def _create_roi_masks(
    areas: np.ndarray, labels: dict, hemi: str, atlas: str, rois: Iterable[str]
) -> dict:
    """Create ROI masks from atlas areas and labels."""
    masks = {}

    # Determine effective ROIs
    if rois and list(rois)[0] == ROI_ALL:
        if atlas == ATLAS_BENSON:
            rois_eff = list(labels.keys())
        else:  # WANG atlas
            rois_eff = [k for k in labels.keys() if k != ROI_UNKNOWN]
    else:
        rois_eff = list(rois)

    # Create masks for each ROI
    for roi in rois_eff:
        roi_labels = [v for k, v in labels.items() if roi in k]
        if not roi_labels:
            continue
        mask = np.any([areas == lab for lab in roi_labels], axis=0)
        masks[(hemi, atlas, roi)] = mask.astype(bool)

    return masks


# -------------------------------- mask builders --------------------------------


def _surface_masks_for_atlas(
    fs_dir: Path | str,
    sub: str,
    hemi: str,
    atlas: str,
    rois: Iterable[str],
    analysis_space: str,
    verbose: bool = False,
) -> dict:
    """
    Return dict of boolean masks keyed by (hemi, atlas, roi) for surface spaces.
    """
    try:
        if atlas == ATLAS_FULLBRAIN:
            return _load_fullbrain_mask(fs_dir, sub, hemi, analysis_space)

        elif atlas in [ATLAS_BENSON, ATLAS_WANG]:
            areas, labels = _load_atlas_data_and_labels(
                fs_dir, sub, hemi, atlas, analysis_space
            )
            return _create_roi_masks(areas, labels, hemi, atlas, rois)

        elif ATLAS_FS_CUSTOM in atlas:
            areas, labels, atlas_name = _load_custom_atlas(
                fs_dir, sub, hemi, atlas, analysis_space
            )
            return _create_roi_masks(areas, labels, hemi, atlas_name, rois)

        else:
            # Unknown atlas
            return {}

    except (AtlasNotFoundError, UnsupportedAtlasError) as e:
        if verbose:
            print(f"Warning: Could not load atlas {atlas} for hemisphere {hemi}: {e}")
        return {}


def _resample_images_to_bold(atlas_vol_img, lh_ribbon_img, rh_ribbon_img, bold_img):
    """Resample atlas and ribbon images to BOLD space if needed."""
    target_shape = bold_img.shape[:-1]

    # Resample atlas volume
    if atlas_vol_img.shape != target_shape:
        atlas_vol_img = resample_to_img(
            atlas_vol_img,
            bold_img,
            interpolation="nearest",
            force_resample=True,
        )

    # Resample left hemisphere ribbon
    if lh_ribbon_img.shape != target_shape:
        lh_ribbon_img = resample_to_img(
            lh_ribbon_img,
            bold_img,
            interpolation="nearest",
            force_resample=True,
        )

    # Resample right hemisphere ribbon
    if rh_ribbon_img.shape != target_shape:
        rh_ribbon_img = resample_to_img(
            rh_ribbon_img,
            bold_img,
            interpolation="nearest",
            force_resample=True,
        )

    return atlas_vol_img, lh_ribbon_img, rh_ribbon_img


def _create_volume_roi_masks(
    atlas_vol, lh_ribbon, rh_ribbon, labels_l, labels_r, atlas, rois, dilate=True
):
    """Create volume ROI masks for both hemispheres."""
    from scipy.ndimage import grey_dilation

    # Apply dilation if requested
    if dilate:
        atlas_l = grey_dilation(
            atlas_vol * lh_ribbon.astype(atlas_vol.dtype), size=(1, 1, 1)
        )
        atlas_r = grey_dilation(
            atlas_vol * rh_ribbon.astype(atlas_vol.dtype), size=(1, 1, 1)
        )
    else:
        atlas_l = atlas_vol * lh_ribbon
        atlas_r = atlas_vol * rh_ribbon

    # Determine effective ROIs
    if rois and list(rois)[0] == ROI_ALL:
        rois_eff = (
            list(labels_l.keys())
            if atlas == ATLAS_BENSON
            else [k for k in labels_l.keys() if k != ROI_UNKNOWN]
        )
    else:
        rois_eff = list(rois)

    masks = {}
    for roi in rois_eff:
        roi_labels_l = [v for k, v in labels_l.items() if roi in k]
        roi_labels_r = [v for k, v in labels_r.items() if roi in k]

        if not roi_labels_l and not roi_labels_r:
            continue

        if roi_labels_l:
            mask_l = np.any([atlas_l == lab for lab in roi_labels_l], axis=0)
            masks[(HEMI_LEFT, atlas, roi)] = mask_l.astype(bool)

        if roi_labels_r:
            mask_r = np.any([atlas_r == lab for lab in roi_labels_r], axis=0)
            masks[(HEMI_RIGHT, atlas, roi)] = mask_r.astype(bool)

    return masks


def _volume_masks_for_atlas(
    fs_dir: Path | str,
    sub: str,
    atlas: str,
    rois: Iterable[str],
    resample: bool = True,
    bold_img: Path | str | None = None,
    dilate: bool = True,
) -> dict:
    """
    Return dict of boolean 3D masks in T1w space keyed by (hemi, atlas, roi),
    or ('both','fullBrain','fullBrain') for volume-wide masks.
    """
    fs_dir = Path(fs_dir)

    # Load cortical ribbon files
    lh_ribbon_path = fs_dir / f"sub-{sub}" / "mri" / "lh.ribbon.mgz"
    rh_ribbon_path = fs_dir / f"sub-{sub}" / "mri" / "rh.ribbon.mgz"

    if not (lh_ribbon_path.exists() and rh_ribbon_path.exists()):
        raise AtlasNotFoundError(
            f"Missing cortical ribbon files for subject {sub}: {lh_ribbon_path}, {rh_ribbon_path}"
        )

    lh_ribbon_img = nib.load(str(lh_ribbon_path))
    rh_ribbon_img = nib.load(str(rh_ribbon_path))

    if atlas == ATLAS_FULLBRAIN:
        lh_ribbon = lh_ribbon_img.get_fdata().astype(bool)
        rh_ribbon = rh_ribbon_img.get_fdata().astype(bool)
        # Store separate per-hemi fullBrain masks for per-hemi unions
        return {
            (HEMI_LEFT, ATLAS_FULLBRAIN, ATLAS_FULLBRAIN): lh_ribbon,
            (HEMI_RIGHT, ATLAS_FULLBRAIN, ATLAS_FULLBRAIN): rh_ribbon,
        }

    # Handle Benson and Wang atlases
    if atlas == ATLAS_BENSON:
        labels_l = _benson_labels(HEMI_LEFT)
        labels_r = _benson_labels(HEMI_RIGHT)
    elif atlas == ATLAS_WANG:
        labels_l = labels_r = _wang_labels()
    else:
        # Volume supports only benson/wang/fullBrain
        raise UnsupportedAtlasError(
            f"Atlas '{atlas}' not supported in volume space. Use: {ATLAS_BENSON}, {ATLAS_WANG}, or {ATLAS_FULLBRAIN}"
        )

    atlas_vol_path = fs_dir / f"sub-{sub}" / "mri" / ATLAS_VOLUME_FILES[atlas]
    if not atlas_vol_path.exists():
        raise AtlasNotFoundError(f"Atlas volume missing: {atlas_vol_path}")

    atlas_vol_img = nib.load(str(atlas_vol_path))

    # Resample to BOLD space if requested
    if resample and bold_img is not None:
        if isinstance(bold_img, (str, Path)):
            bold_img = nib.load(str(bold_img))
        atlas_vol_img, lh_ribbon_img, rh_ribbon_img = _resample_images_to_bold(
            atlas_vol_img, lh_ribbon_img, rh_ribbon_img, bold_img
        )

    atlas_vol = atlas_vol_img.get_fdata()
    lh_ribbon = lh_ribbon_img.get_fdata().astype(bool)
    rh_ribbon = rh_ribbon_img.get_fdata().astype(bool)

    return _create_volume_roi_masks(
        atlas_vol, lh_ribbon, rh_ribbon, labels_l, labels_r, atlas, rois, dilate
    )


# --------------------------- Neuropythy integration ---------------------------


def _run_neuropythy(
    fs_dir: Path | str, sub: str, analysis_space: str, custom_annots=None, LOG=None
) -> None:
    """
    Ensure Neuropythy outputs exist for subject (and fsaverage if needed),
    and project custom fsaverage annot files to subject space. Idempotent.
    """
    try:
        from neuropythy.commands import atlas
    except ImportError as e:
        raise RuntimeError(
            "Neuropythy is required for ROI generation but is not installed."
        ) from e

    fs_dir = Path(fs_dir)
    subject = f"sub-{sub}" if not str(sub).startswith("sub-") else str(sub)

    # Subject-level Benson maps
    subj_benson = fs_dir / subject / "mri" / "benson14_varea.mgz"
    if not subj_benson.exists():
        LOG = LOG or get_logger(__file__)
        LOG.debug(f"Neuropythy: generating Benson maps for {subject}...")
        try:
            cwd_prev = Path.cwd()
            os.chdir(fs_dir)
            atlas.main(subject, "-v", "-S")
            os.chdir(cwd_prev)
        except Exception as e:
            raise RuntimeError(f"Neuropythy failed for {subject}: {e}") from e

    # Custom annot projection fsaverage -> subject
    if custom_annots:
        os.environ["SUBJECTS_DIR"] = str(fs_dir)
        dest_dir = fs_dir / subject / "customLabel"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for annot in map(Path, custom_annots):
            dst = dest_dir / annot.name
            if dst.exists():
                continue
            hemi = annot.name.split(".")[0]  # 'lh' or 'rh'
            cmd = [
                "mri_surf2surf",
                "--srcsubject",
                "fsaverage",
                "--trgsubject",
                subject,
                "--hemi",
                hemi,
                "--sval-annot",
                str(annot),
                "--tval",
                str(dst),
            ]
            LOG = LOG or get_logger(__file__)
            LOG.debug(f"Projecting annot {annot.name} -> {subject} ({hemi})")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"mri_surf2surf failed for {annot}: {e}") from e

    # fsaverage-level atlases if needed
    if str(analysis_space).lower() == SPACE_FSAVERAGE:
        fsavg_wang_rh = fs_dir / "fsaverage" / "surf" / "rh.wang15_fplbl.mgz"
        if not fsavg_wang_rh.exists():
            LOG = LOG or get_logger(__file__)
            LOG.debug(f"Neuropythy: generating atlases for fsaverage...")
            try:
                cwd_prev = Path.cwd()
                os.chdir(fs_dir)
                atlas.main("fsaverage", "-v")
                os.chdir(cwd_prev)
            except Exception as e:
                raise RuntimeError(f"Neuropythy failed for fsaverage: {e}") from e


# ------------------------------- public builder -------------------------------


def prepare_roi_pack(
    ctx: dict,
    analysis_space: str,
    atlases: Iterable[str],
    rois: Iterable[str],
    fs_dir: Path | str,
    out_base: Path | str,
    custom_annots: Optional[Iterable[str]] = None,
    bold_img: Optional[Path | str] = None,
    verbose: bool = False,
) -> RoiPack:
    """
    Build or load the ROI cache for a given (sub, analysis_space, atlases, rois, fs_dir).

    File outputs (space separation is expected to be done by the caller via out_base path):
      <out_base>/sub-<sub>/
        ├── all_roi_masks.h5          (contains 'key' and 'meta_json' attrs)
        └── all_roi_masks_meta.json   (contains the same 'key' and canonical meta)
    """
    sub = ctx.get("sub")
    LOG = ctx.get("log") or get_logger(
        __file__, verbose=bool(verbose or ctx.get("verbose", False))
    )
    force = ctx.get("force")
    store_as_indices = bool(ctx.get("store_as_indices", False))

    # Validate inputs
    _validate_inputs(analysis_space, atlases, rois)

    # Ensure prerequisites (idempotent)
    _run_neuropythy(fs_dir, sub, analysis_space, custom_annots=custom_annots, LOG=LOG)

    # Output locations (space lives outside in folder structure as you prefer)
    out_dir = out_base / f"sub-{sub}"
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "all_roi_masks.h5"
    meta_path = out_dir / "all_roi_masks_meta.json"

    # Canonical meta + key
    meta_canon = _canonical_meta(sub, analysis_space, atlases, rois, fs_dir)
    key = _meta_digest(meta_canon)
    meta = dict(meta_canon)
    meta["key"] = key

    # If cache exists and not forcing, refresh meta JSON and, for surface spaces, reuse immediately.
    if h5_path.exists() and not force:
        if meta_path.exists():
            try:
                m = json.loads(meta_path.read_text())
            except Exception:
                m = meta
            if "key" not in m:
                m["key"] = key
            meta = m
        meta_path.write_text(json.dumps(meta, indent=2))
        if analysis_space in ("fsnative", "fsaverage"):
            return RoiPack(key, h5_path, meta)

    # Compute ROI masks
    roi_masks = {}
    atlases = list(atlases or [])
    rois = list(rois or [])

    if analysis_space in (SPACE_FSNATIVE, SPACE_FSAVERAGE):
        for atlas in atlases:
            for hemi in (HEMI_LEFT, HEMI_RIGHT):
                roi_masks.update(
                    _surface_masks_for_atlas(
                        fs_dir, sub, hemi, atlas, rois, analysis_space, verbose=verbose
                    )
                )
        # Persist surface under root
        _save_rois_h5(
            h5_path,
            roi_masks,
            meta,
            key,
            base_group=None,
        )
        meta_path.write_text(json.dumps(meta, indent=2))
        return RoiPack(key, h5_path, meta)
    elif analysis_space == SPACE_VOLUME:
        if bold_img is None:
            raise ValueError(
                f"For {SPACE_VOLUME} analysis_space, bold_img must be provided to define the grid."
            )
        # compute grid id from BOLD (load only if path-like)
        if isinstance(bold_img, (str, Path)):
            bimg = nib.load(str(bold_img))
        else:
            bimg = bold_img
        gid = _grid_signature(bimg.shape[:3], bimg.affine)
        base_group = f"grids/{gid}"

        # If grid exists and not forcing, reuse
        if h5_path.exists() and not force:
            with h5py.File(str(h5_path), "r") as f:
                if f.get(f"/{base_group}/original_space") is not None:
                    return RoiPack(key, h5_path, meta)

        # build ROI masks for this grid
        for atlas in atlases:
            if atlas not in (ATLAS_BENSON, ATLAS_WANG, ATLAS_FULLBRAIN):
                LOG.debug(f"Skipping atlas '{atlas}' in volume space (unsupported).")
                continue
            roi_masks.update(
                _volume_masks_for_atlas(
                    fs_dir,
                    sub,
                    atlas,
                    rois,
                    bold_img=bimg,
                )
            )
        # write grid metadata and masks under the grid group with a lightweight lock
        try:
            with h5py.File(str(h5_path), "a") as f:
                gg = f.require_group(base_group)
                for name in ("grid_shape", "grid_affine"):
                    if name in gg:
                        del gg[name]
                gg.create_dataset(
                    "grid_shape",
                    data=np.asarray(bimg.shape[:3], np.int32),
                    compression="gzip",
                    shuffle=True,
                    fletcher32=True,
                )
                gg.create_dataset(
                    "grid_affine",
                    data=np.asarray(bimg.affine, np.float64),
                    compression="gzip",
                    shuffle=True,
                    fletcher32=True,
                )
            _save_rois_h5(
                h5_path,
                roi_masks,
                meta,
                key,
                base_group=base_group,
            )
        except Exception as e:
            LOG.error(f"Failed to write ROI pack to {h5_path}: {e}")
            raise e
        meta_path.write_text(json.dumps(meta, indent=2))
        return RoiPack(key, h5_path, meta)
    else:
        raise ValueError(
            f"Unsupported analysis_space '{analysis_space}'. Choose from {SPACE_FSNATIVE}, {SPACE_FSAVERAGE}, {SPACE_VOLUME}."
        )

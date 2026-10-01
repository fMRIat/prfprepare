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
ATLAS_FULLBRAIN = "full_brain"
ATLAS_FS_CUSTOM = "fs_custom"
ATLAS_NIFTI_CUSTOM = "nifti_custom"

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

# Alternative surface filenames per atlas. Neuropythy writes the full-probability
# labels (wang15_fplbl) for fsaverage but the maximum-probability labels
# (wang15_mplbl) for individual subjects, so vertex-count checks must probe both.
ATLAS_FILE_VARIANTS = {
    ATLAS_WANG: ("wang15_mplbl.mgz", "wang15_fplbl.mgz"),
    ATLAS_BENSON: ("benson14_varea.mgz",),
}

# Canonical vertex count per hemisphere for the fsaverage surface. Guards against
# a lower-density BOLD (fsaverage6 = 40962, fsaverage5 = 10242) being processed as
# if it were fsaverage, which the hardcoded 'space-fsaverage' filename allows.
FSAVERAGE_N_VERTICES = 163842

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
    sub: str,
    analysis_space: str,
    atlases,
    rois,
    fs_dir: Path | str,
    vertex_counts: Optional[dict] = None,
) -> dict:
    """Canonicalize meta for a stable key (order-insensitive for atlases/rois).

    ``vertex_counts`` participates in the key so that regenerating the FreeSurfer
    surfaces invalidates a cached ROI pack whose indices describe the old surface.
    """
    meta = {
        "sub": sub,
        "analysis_space": str(analysis_space),
        "atlases": sorted(list(atlases or [])),
        "rois": sorted(list(rois or [])),
        "fs_dir": str(fs_dir),
    }
    if vertex_counts:
        meta["vertex_counts"] = {k: int(v) for k, v in sorted(vertex_counts.items())}
    return meta


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

        # Shape consistency. Required for both grids: mismatched surface masks
        # otherwise reach np.stack as a bare ValueError, or -- when the union is
        # built from the shorter mask -- index out of bounds in g2m below.
        key0, mask0 = items[0]
        for (h, a, r), m in items:
            if m.shape != mask0.shape:
                kind = "Volume" if grid_order == "C" else "Surface"
                raise GridMismatchError(
                    f"{kind} shape mismatch for hemi {hemi_label}: "
                    f"atlas '{a}' roi '{r}' has shape {m.shape}, but "
                    f"atlas '{key0[1]}' roi '{key0[2]}' has shape {mask0.shape}. "
                    "All masks for a hemisphere must describe the same grid."
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
        # If a 'both' mask exists (e.g., full_brain), duplicate into l and r sets
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


def _resolve_fs_subject_dir(
    fs_dir: Path | str, sub: str, sess: str | None = None
) -> Path:
    """
    Resolve FreeSurfer subject directory, checking for new layout if old doesn't exist.

    Parameters
    ----------
    fs_dir : Path | str
        Base FreeSurfer directory
    sub : str
        Subject ID
    sess : str | None, optional
        Session ID for new layout fallback

    Returns
    -------
    Path
        Resolved subject directory path
    """
    fs_dir = Path(fs_dir)
    old_path = fs_dir / f"sub-{sub}"

    if old_path.exists():
        return old_path

    # Try new layout with session if available
    if sess:
        new_path = fs_dir / f"sub-{sub}_ses-{sess}"
        if new_path.exists():
            return new_path

    # Try to find any session directory for this subject
    pattern = fs_dir / f"sub-{sub}_ses-*"
    matches = sorted(pattern.parent.glob(pattern.name))
    if matches:
        if len(matches) > 1:
            # Per-session recon-all produces different vertex counts per session,
            # so an arbitrary pick here silently changes the surface being used.
            get_logger(__file__).warning(
                f"Multiple FreeSurfer session directories for sub-{sub} "
                f"(session {sess!r} did not resolve): "
                f"{', '.join(m.name for m in matches)}. Using {matches[0].name}."
            )
        return matches[0]

    # Return old path as default (will fail later if doesn't exist)
    return old_path


def _build_atlas_path(
    fs_dir: Path | str,
    sub: str,
    hemi: str,
    atlas: str,
    analysis_space: str,
    sess: str | None = None,
) -> Path:
    """Build file path for surface atlas files."""
    sub_path = (
        _resolve_fs_subject_dir(fs_dir, sub, sess).name
        if analysis_space == SPACE_FSNATIVE
        else "fsaverage"
    )
    atlas_file = ATLAS_FILES[atlas]
    return Path(fs_dir) / sub_path / "surf" / f"{hemi}h.{atlas_file}"


def _surface_subject_dir(
    fs_dir: Path | str, sub: str, analysis_space: str, sess: str | None = None
) -> Path:
    """Resolve the FreeSurfer directory holding surfaces for this analysis space."""
    fs_dir = Path(fs_dir)
    if analysis_space == SPACE_FSNATIVE:
        return _resolve_fs_subject_dir(fs_dir, sub, sess)
    return fs_dir / "fsaverage"


def _geometry_vertex_count(sub_dir: Path, hemi: str) -> tuple[int, Path]:
    """Vertex count from FreeSurfer surface geometry (the authoritative source)."""
    tried = []
    for surf in ("white", "orig", "pial"):
        path = sub_dir / "surf" / f"{hemi}h.{surf}"
        tried.append(path)
        if path.exists():
            coords, _ = nib.freesurfer.read_geometry(str(path))
            return int(coords.shape[0]), path

    raise AtlasNotFoundError(
        f"No FreeSurfer surface geometry found for hemisphere {hemi}. Tried: "
        + ", ".join(str(p) for p in tried)
    )


def _atlas_vertex_count(path: Path) -> int:
    """
    Vertex count of a surface atlas overlay.

    Maximum-probability labels (wang15_mplbl, benson14_varea) squeeze to a plain
    (vertices,) vector, but the full-probability labels (wang15_fplbl) squeeze to
    (n_areas, vertices) -- one probability map per visual area. Taking the largest
    axis reads the vertex count from either, since the number of visual areas is
    orders of magnitude smaller than the number of vertices.
    """
    shape = nib.load(str(path)).get_fdata().squeeze().shape
    return int(max(shape))


def _bold_vertex_count(
    bold_img, hemi: str, bold_hemi: Optional[str] = None
) -> Optional[tuple[int, str]]:
    """
    Vertex count from a surface BOLD GIFTI, or None if it says nothing about `hemi`.

    A BOLD only constrains the hemisphere it actually belongs to. That hemisphere
    comes from ``bold_hemi`` when the caller knows it (authoritative, and the only
    option for in-memory images such as an averaged run), otherwise from the
    filename. An image we cannot attribute to a hemisphere is skipped rather than
    compared against both -- comparing both would fail whichever hemisphere it is
    not, and the bounds check in nii_to_surfNii still guards the masking itself.
    """
    if bold_img is None:
        return None

    try:
        if bold_hemi is not None:
            if str(bold_hemi).lower() != hemi:
                return None
        elif isinstance(bold_img, (str, Path)):
            # Only the matching hemisphere's file describes this hemi's surface.
            if f"hemi-{hemi.upper()}" not in Path(str(bold_img)).name:
                return None
        else:
            # In-memory image with no declared hemisphere: not attributable.
            return None

        if isinstance(bold_img, (str, Path)):
            path = str(bold_img)
            img = nib.load(path)
        else:
            img = bold_img
            path = (
                getattr(img, "get_filename", lambda: None)()
                or "<in-memory averaged run>"
            )

        # Volume inputs have no vertex count to compare against.
        if not hasattr(img, "agg_data"):
            return None

        # fMRIPrep writes one darray per timepoint, each holding all vertices, and
        # agg_data stacks them as (vertices, timepoints) -- matching the axis that
        # nii_to_surfNii indexes. Read the vertex count from a single darray so we
        # do not depend on how agg_data orients the stack.
        darrays = getattr(img, "darrays", None)
        if darrays:
            return int(np.asarray(darrays[0].data).shape[0]), path

        return int(np.asarray(img.agg_data()).shape[0]), path
    except AtlasNotFoundError:
        raise
    except Exception:
        # A BOLD we cannot read is not evidence of a mismatch; the per-run bounds
        # check in nii_to_surfNii still guards the actual masking step.
        return None


def _validate_surface_vertex_counts(
    fs_dir: Path | str,
    sub: str,
    analysis_space: str,
    atlases: Iterable[str],
    sess: str | None = None,
    bold_img=None,
    bold_hemi: Optional[str] = None,
    LOG=None,
) -> dict:
    """
    Check that every source of vertex counts agrees, before any mask is built.

    Three independent sources must describe the same surface:
      1. FreeSurfer geometry (?h.white) -- authoritative
      2. neuropythy atlas overlays (wang, benson)
      3. the fMRIPrep BOLD GIFTI being masked

    They agree by construction; a disagreement means the inputs are inconsistent
    (commonly a re-run recon-all against stale neuropythy output) and any result
    would be silently wrong. Sources whose files are absent are skipped -- that is
    handled separately as AtlasNotFoundError when the atlas is actually loaded.

    Returns
    -------
    dict
        Per-hemisphere vertex counts, e.g. ``{"l": 149623, "r": 150894}``.

    Raises
    ------
    GridMismatchError
        If the present sources disagree, or if an fsaverage surface is not the
        canonical density.
    """
    fs_dir = Path(fs_dir)
    sub_dir = _surface_subject_dir(fs_dir, sub, analysis_space, sess)
    atlases = list(atlases or [])
    counts = {}

    for hemi in (HEMI_LEFT, HEMI_RIGHT):
        n_geom, geom_path = _geometry_vertex_count(sub_dir, hemi)
        sources = [("geometry", str(geom_path), n_geom)]

        for atlas in atlases:
            for fname in ATLAS_FILE_VARIANTS.get(atlas, ()):
                path = sub_dir / "surf" / f"{hemi}h.{fname}"
                if path.exists():
                    n = _atlas_vertex_count(path)
                    sources.append((f"atlas:{atlas}", str(path), n))

        bold = _bold_vertex_count(bold_img, hemi, bold_hemi=bold_hemi)
        if bold is not None:
            sources.append(("bold", bold[1], bold[0]))

        distinct = {n for _, _, n in sources}
        if len(distinct) > 1:
            detail = "\n".join(f"  {n:>9}  {name:<16} {path}" for name, path, n in sources)
            raise GridMismatchError(
                f"Vertex count mismatch for subject {sub}, hemisphere {hemi}h "
                f"({analysis_space}):\n{detail}\n"
                "These must all describe the same surface. This usually means the "
                "FreeSurfer surfaces were regenerated without refreshing the neuropythy "
                "atlases, or that a cached ROI pack is stale -- re-run with force to "
                "rebuild."
            )

        if analysis_space == SPACE_FSAVERAGE and n_geom != FSAVERAGE_N_VERTICES:
            raise GridMismatchError(
                f"analysisSpace is '{SPACE_FSAVERAGE}' but the {hemi}h surface has "
                f"{n_geom} vertices, not the expected {FSAVERAGE_N_VERTICES}. This "
                "usually means the data is on a lower-density surface such as "
                "fsaverage6 (40962) or fsaverage5 (10242)."
            )

        counts[hemi] = n_geom

    if LOG is not None:
        LOG.debug(
            f"Vertex counts validated ({analysis_space}): "
            f"lh={counts[HEMI_LEFT]}, rh={counts[HEMI_RIGHT]}"
        )
    return counts


def _load_fullbrain_mask(
    fs_dir: Path | str,
    sub: str,
    hemi: str,
    analysis_space: str,
    sess: str | None = None,
    n_vertices: Optional[int] = None,
) -> dict:
    """
    Build the full_brain mask: every vertex of the hemisphere.

    The vertex count comes from the validated count when available, else directly
    from the surface geometry. It is deliberately not inferred from an atlas
    overlay, which may be absent or stale.
    """
    if n_vertices is None:
        sub_dir = _surface_subject_dir(fs_dir, sub, analysis_space, sess)
        n_vertices, _ = _geometry_vertex_count(sub_dir, hemi)

    return {
        (hemi, ATLAS_FULLBRAIN, ATLAS_FULLBRAIN): np.ones((n_vertices,), dtype=bool)
    }


def _load_atlas_data_and_labels(
    fs_dir: Path | str,
    sub: str,
    hemi: str,
    atlas: str,
    analysis_space: str,
    sess: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Load atlas data and corresponding labels."""
    atlas_path = _build_atlas_path(fs_dir, sub, hemi, atlas, analysis_space, sess)
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
    fs_dir: Path | str,
    sub: str,
    hemi: str,
    atlas: str,
    analysis_space: str,
    sess: str | None = None,
) -> tuple[np.ndarray, dict, str]:
    """Load custom FreeSurfer annotation atlas."""
    if analysis_space == SPACE_VOLUME:
        raise UnsupportedAtlasError("Custom atlas not supported in volume space")
    if f"{hemi}h." not in atlas:
        raise UnsupportedAtlasError(f"Custom atlas {atlas} not for hemisphere {hemi}")

    sub_dir = _resolve_fs_subject_dir(fs_dir, sub, sess)
    annot_path = sub_dir / "customLabel" / atlas
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


def _load_nifti_custom_atlas(
    atlas_info: dict, analysis_space: str
) -> tuple[np.ndarray, dict, str]:
    """Load custom NIfTI atlas with text label file.

    Parameters
    ----------
    atlas_info : dict
        Dict with keys 'name', 'nifti', 'labels' (file paths)
    analysis_space : str
        Analysis space (volume/surface)

    Returns
    -------
    areas : np.ndarray
        Atlas label data
    area_labels : dict
        Mapping of ROI names to label values
    atlas_name : str
        Name of the atlas
    """
    nifti_path = Path(atlas_info["nifti"])
    label_path = Path(atlas_info["labels"])

    if not nifti_path.exists():
        raise AtlasNotFoundError(f"Custom NIfTI atlas missing: {nifti_path}")
    if not label_path.exists():
        raise AtlasNotFoundError(f"Custom atlas labels missing: {label_path}")

    # Load NIfTI data
    areas = nib.load(str(nifti_path)).get_fdata().squeeze()

    # Parse label text file (format: "label_value ROI_name" per line)
    area_labels = {ROI_UNKNOWN: 0}
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                try:
                    label_val = int(parts[0])
                    roi_name = parts[1]
                    area_labels[roi_name] = label_val
                except ValueError:
                    continue

    return areas, area_labels, atlas_info["name"]


# -------------------------------- mask builders --------------------------------


def _surface_masks_for_atlas(
    fs_dir: Path | str,
    sub: str,
    hemi: str,
    atlas: str,
    rois: Iterable[str],
    analysis_space: str,
    sess: str | None = None,
    verbose: bool = False,
    n_vertices: Optional[int] = None,
) -> dict:
    """
    Return dict of boolean masks keyed by (hemi, atlas, roi) for surface spaces.
    """
    try:
        if atlas == ATLAS_FULLBRAIN:
            return _load_fullbrain_mask(
                fs_dir, sub, hemi, analysis_space, sess, n_vertices=n_vertices
            )

        elif atlas in [ATLAS_BENSON, ATLAS_WANG]:
            areas, labels = _load_atlas_data_and_labels(
                fs_dir, sub, hemi, atlas, analysis_space, sess
            )
            return _create_roi_masks(areas, labels, hemi, atlas, rois)

        elif ATLAS_FS_CUSTOM in atlas:
            areas, labels, atlas_name = _load_custom_atlas(
                fs_dir, sub, hemi, atlas, analysis_space, sess
            )
            return _create_roi_masks(areas, labels, hemi, atlas_name, rois)

        else:
            # Unknown atlas (ignore custom NIfTI in surface mode)
            return {}

    except (AtlasNotFoundError, UnsupportedAtlasError) as e:
        if verbose:
            print(f"Warning: Could not load atlas {atlas} for hemisphere {hemi}: {e}")
        return {}


def _same_grid(img, bold_img, atol=1e-5) -> bool:
    """True if img already lies on the BOLD voxel grid (same shape and affine)."""
    return img.shape[:3] == bold_img.shape[:3] and np.allclose(
        img.affine, bold_img.affine, atol=atol
    )


def _resample_images_to_bold(atlas_vol_img, lh_ribbon_img, rh_ribbon_img, bold_img):
    """Resample atlas and ribbon images to BOLD space if needed."""
    # Resample atlas volume
    if not _same_grid(atlas_vol_img, bold_img):
        atlas_vol_img = resample_to_img(
            atlas_vol_img,
            bold_img,
            interpolation="nearest",
            force_resample=True,
        )

    # Resample left hemisphere ribbon
    if not _same_grid(lh_ribbon_img, bold_img):
        lh_ribbon_img = resample_to_img(
            lh_ribbon_img,
            bold_img,
            interpolation="nearest",
            force_resample=True,
        )

    # Resample right hemisphere ribbon
    if not _same_grid(rh_ribbon_img, bold_img):
        rh_ribbon_img = resample_to_img(
            rh_ribbon_img,
            bold_img,
            interpolation="nearest",
            force_resample=True,
        )

    return atlas_vol_img, lh_ribbon_img, rh_ribbon_img


def _dilate_atlas_with_distance_assignment(atlas_base, unique_labels):
    """Dilate atlas labels using distance-based assignment at borders.

    Parameters
    ----------
    atlas_base : np.ndarray
        Original atlas with integer labels
    unique_labels : np.ndarray
        Array of unique label values (excluding 0)

    Returns
    -------
    np.ndarray
        Dilated atlas with disputed voxels assigned to nearest label
    """
    from scipy.ndimage import binary_dilation, distance_transform_edt

    atlas = np.copy(atlas_base)
    if len(unique_labels) == 0:
        return atlas

    # Create union of all dilated masks to find disputed territory
    union_dilated = np.zeros_like(atlas_base, dtype=bool)
    for label in unique_labels:
        label_mask = atlas_base == label
        dilated_mask = binary_dilation(label_mask, structure=np.ones((3, 3, 3)))
        union_dilated |= dilated_mask

    # Find voxels that need assignment (in union but not in original)
    to_assign = union_dilated & (atlas_base == 0)

    if np.any(to_assign):
        # For each label, compute distance transform from original mask
        min_distances = np.full(atlas_base.shape, np.inf)
        nearest_labels = np.zeros_like(atlas_base)

        for label in unique_labels:
            label_mask = atlas_base == label
            distances = distance_transform_edt(~label_mask)
            closer = distances < min_distances
            min_distances[closer] = distances[closer]
            nearest_labels[closer] = label

        # Assign disputed voxels to nearest label
        atlas[to_assign] = nearest_labels[to_assign]

    return atlas


def _create_volume_roi_masks(
    atlas_vol,
    lh_ribbon,
    rh_ribbon,
    labels_l,
    labels_r,
    atlas,
    rois,
    dilate=True,
):
    """Create volume ROI masks for both hemispheres."""
    if dilate:
        atlas_l_base = atlas_vol * lh_ribbon.astype(atlas_vol.dtype)
        atlas_r_base = atlas_vol * rh_ribbon.astype(atlas_vol.dtype)

        # Get unique labels (excluding 0) and dilate using distance-based assignment
        unique_labels_l = np.unique(atlas_l_base)
        unique_labels_l = unique_labels_l[unique_labels_l > 0]
        atlas_l = _dilate_atlas_with_distance_assignment(atlas_l_base, unique_labels_l)

        unique_labels_r = np.unique(atlas_r_base)
        unique_labels_r = unique_labels_r[unique_labels_r > 0]
        atlas_r = _dilate_atlas_with_distance_assignment(atlas_r_base, unique_labels_r)
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
    nifti_custom_atlases: Optional[list] = None,
    sess: str | None = None,
) -> dict:
    """
    Return dict of boolean 3D masks in T1w space keyed by (hemi, atlas, roi),
    or ('both','full_brain','full_brain') for volume-wide masks.
    """
    fs_dir = Path(fs_dir)
    sub_dir = _resolve_fs_subject_dir(fs_dir, sub, sess)

    # Load cortical ribbon files
    lh_ribbon_path = sub_dir / "mri" / "lh.ribbon.mgz"
    rh_ribbon_path = sub_dir / "mri" / "rh.ribbon.mgz"

    if not (lh_ribbon_path.exists() and rh_ribbon_path.exists()):
        raise AtlasNotFoundError(
            f"Missing cortical ribbon files for subject {sub}: {lh_ribbon_path}, {rh_ribbon_path}"
        )

    lh_ribbon_img = nib.load(str(lh_ribbon_path))
    rh_ribbon_img = nib.load(str(rh_ribbon_path))

    if atlas == ATLAS_FULLBRAIN:
        # Resample the ribbon to the BOLD grid so the flat indices are in BOLD
        # voxel order and stay shape-consistent with wang/benson masks in the
        # union (otherwise apply_masks_to_run indexes the wrong grid and
        # _write_union_membership raises GridMismatchError).
        if resample and bold_img is not None:
            if isinstance(bold_img, (str, Path)):
                bold_img = nib.load(str(bold_img))
            if not _same_grid(lh_ribbon_img, bold_img):
                lh_ribbon_img = resample_to_img(
                    lh_ribbon_img,
                    bold_img,
                    interpolation="nearest",
                    force_resample=True,
                )
            if not _same_grid(rh_ribbon_img, bold_img):
                rh_ribbon_img = resample_to_img(
                    rh_ribbon_img,
                    bold_img,
                    interpolation="nearest",
                    force_resample=True,
                )
        lh_ribbon = lh_ribbon_img.get_fdata().astype(bool)
        rh_ribbon = rh_ribbon_img.get_fdata().astype(bool)
        # Store separate per-hemi full_brain masks for per-hemi unions
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
    elif ATLAS_NIFTI_CUSTOM in atlas or (
        nifti_custom_atlases and atlas in [a["name"] for a in nifti_custom_atlases]
    ):
        # Find the matching atlas info
        atlas_info = None
        if nifti_custom_atlases:
            for a in nifti_custom_atlases:
                if a["name"] == atlas:
                    atlas_info = a
                    break
        if not atlas_info:
            raise UnsupportedAtlasError(f"Custom NIfTI atlas '{atlas}' info not found")

        # Load the custom atlas - for volume it should already be in volume space
        areas, labels, atlas_name = _load_nifti_custom_atlas(atlas_info, SPACE_VOLUME)
        # For volume, we assume the NIfTI already has both hemispheres
        labels_l = labels_r = labels

        # Load as volume image
        atlas_vol_img = nib.load(str(atlas_info["nifti"]))

        dilate = False  # Assume already preprocessed
    else:
        # Volume supports only benson/wang/full_brain/custom
        raise UnsupportedAtlasError(
            f"Atlas '{atlas}' not supported in volume space. Use: {ATLAS_BENSON}, {ATLAS_WANG}, {ATLAS_FULLBRAIN}, or custom NIfTI"
        )

    # Load atlas volume path for standard atlases
    if atlas in [ATLAS_BENSON, ATLAS_WANG]:
        atlas_vol_path = sub_dir / "mri" / ATLAS_VOLUME_FILES[atlas]
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
    fs_dir: Path | str,
    sub: str,
    ses: str | None,
    analysis_space: str,
    custom_annots=None,
    LOG=None,
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
    subject = _resolve_fs_subject_dir(fs_dir, sub, ses).name

    # Subject-level Benson maps
    subj_benson = fs_dir / subject / "mri" / "benson14_varea.mgz"
    subj_wang_rh = fs_dir / subject / "surf" / "rh.wang15_mplbl.mgz"
    if not subj_benson.exists() or not subj_wang_rh.exists():
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
        # For custom annots, we need to determine if using new layout
        # Extract session from context if available (passed through ctx)
        sub_dir = _resolve_fs_subject_dir(fs_dir, sub, ses)
        dest_dir = sub_dir / "customLabel"
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
    nifti_custom_atlases: Optional[list] = None,
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
    ses = ctx.get("ses")  # Extract session for new layout support
    LOG = ctx.get("log") or get_logger(
        __file__, verbose=bool(verbose or ctx.get("verbose", False))
    )
    force = ctx.get("force")
    store_as_indices = bool(ctx.get("store_as_indices", False))

    # Validate inputs
    _validate_inputs(analysis_space, atlases, rois)

    # Ensure prerequisites (idempotent)
    _run_neuropythy(
        fs_dir, sub, ses, analysis_space, custom_annots=custom_annots, LOG=LOG
    )

    # Output locations (space lives outside in folder structure as you prefer)
    out_dir = out_base / f"sub-{sub}"
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "all_roi_masks.h5"
    meta_path = out_dir / "all_roi_masks_meta.json"

    # Validate that geometry, atlases and BOLD agree on vertex counts before any
    # mask is built. Runs ahead of the cache check so a stale pack is caught
    # rather than served.
    vertex_counts = None
    if analysis_space in (SPACE_FSNATIVE, SPACE_FSAVERAGE):
        vertex_counts = _validate_surface_vertex_counts(
            fs_dir,
            sub,
            analysis_space,
            atlases,
            sess=ses,
            bold_img=bold_img,
            # The caller knows which hemisphere this BOLD is; an averaged run is an
            # in-memory image whose hemisphere cannot be read from a filename.
            bold_hemi=ctx.get("hemi"),
            LOG=LOG,
        )

    # Canonical meta + key
    meta_canon = _canonical_meta(
        sub, analysis_space, atlases, rois, fs_dir, vertex_counts=vertex_counts
    )
    key = _meta_digest(meta_canon)
    meta = dict(meta_canon)
    meta["key"] = key

    # If cache exists and not forcing, reuse it -- but only when it describes the
    # same inputs. The stored key covers the vertex counts, so a cache built
    # against different surfaces is rebuilt instead of producing indices that no
    # longer match the data.
    if h5_path.exists() and not force:
        cached_key = None
        if meta_path.exists():
            try:
                cached_key = json.loads(meta_path.read_text()).get("key")
            except Exception:
                cached_key = None

        if analysis_space in ("fsnative", "fsaverage"):
            # Verify cache has flat_index datasets (added in newer schema); rebuild if missing
            try:
                with h5py.File(str(h5_path), "r") as _f:
                    _has_flat = (
                        "masked_space/flat_index_l" in _f
                        and "masked_space/flat_index_r" in _f
                    )
            except Exception:
                _has_flat = False

            if not _has_flat:
                LOG.info(
                    "ROI cache missing flat_index datasets (old schema); rebuilding…"
                )
            elif cached_key is None:
                LOG.info("ROI cache has no stored key; rebuilding…")
            elif cached_key != key:
                LOG.info(
                    f"ROI cache key mismatch (cached={cached_key}, expected={key}); "
                    "inputs or surface geometry changed, rebuilding…"
                )
            else:
                meta_path.write_text(json.dumps(meta, indent=2))
                return RoiPack(key, h5_path, meta)
        else:
            meta_path.write_text(json.dumps(meta, indent=2))

    # Compute ROI masks
    roi_masks = {}
    atlases = list(atlases or [])
    rois = list(rois or [])

    if analysis_space in (SPACE_FSNATIVE, SPACE_FSAVERAGE):
        for atlas in atlases:
            for hemi in (HEMI_LEFT, HEMI_RIGHT):
                roi_masks.update(
                    _surface_masks_for_atlas(
                        fs_dir,
                        sub,
                        hemi,
                        atlas,
                        rois,
                        analysis_space,
                        sess=ses,
                        verbose=verbose,
                        n_vertices=(vertex_counts or {}).get(hemi),
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
            # Allow custom nifti atlases in volume space
            is_custom_nifti = nifti_custom_atlases and atlas in [
                a["name"] for a in nifti_custom_atlases
            ]
            if (
                atlas not in (ATLAS_BENSON, ATLAS_WANG, ATLAS_FULLBRAIN)
                and not is_custom_nifti
            ):
                LOG.debug(f"Skipping atlas '{atlas}' in volume space (unsupported).")
                continue
            roi_masks.update(
                _volume_masks_for_atlas(
                    fs_dir,
                    sub,
                    atlas,
                    rois,
                    bold_img=bimg,
                    nifti_custom_atlases=nifti_custom_atlases,
                    sess=ses,
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

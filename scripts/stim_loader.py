# stim_loader.py

import json
from pathlib import Path

import h5py
import numpy as np

# --------------------------- public classes ----------------------------


class StimSpec(object):
    def __init__(
        self,
        kind,
        task,
        tr,
        prescan,
        start_scan,
        n_frames,
        frames,
        aperture_nii,
        meta,
        seqtiming,
    ):
        """
        Initialize a StimSpec instance.

        Parameters
        ----------
        kind : str
            Type of stimulus source (e.g., 'forced_mat', 'h5_auto').
        task : str
            Task label.
        tr : float or None
            Repetition time in seconds.
        prescan : float or None
            Prescan duration in seconds.
        start_scan : float or None
            Start scan time in seconds.
        n_frames : int or None
            Number of frames (-1 if unknown).
        frames : H5Frames or MatFrames or None
            Stimulus frames object.
        aperture_nii : str or Path or None
            Path to precomputed aperture NIfTI file.
        meta : dict or None
            Metadata dictionary.
        seqtiming : array-like or None
            Sequence timing information.
        """
        self.kind = kind
        self.task = task
        self.tr = float(tr) if tr is not None else float("nan")
        self.prescan = float(prescan) if prescan is not None else 0.0
        self.start_scan = float(start_scan) if start_scan is not None else 0.0
        self.n_frames = int(n_frames) if n_frames is not None else -1
        self.frames = frames
        # frames may be None for precomputed apertures
        self.seq = getattr(frames, "seq", None)
        self.seqtiming = seqtiming
        self.aperture_nii = Path(aperture_nii) if aperture_nii else None
        self.meta = meta or {}


class H5Frames:
    def __init__(self, h5_path, key="stimulus"):
        """
        Initialize H5Frames for lazy access to HDF5 stimulus data.

        Parameters
        ----------
        h5_path : str or Path
            Path to the HDF5 file.
        key : str, optional
            Key for the stimulus group in HDF5 (default: 'stimulus').
        """
        self.h5_path = Path(h5_path)
        self.key = key
        with h5py.File(str(self.h5_path), "r") as f:
            if self.key not in f:
                raise KeyError(f"Dataset '{self.key}' not found in {self.h5_path}")
            self._shape = tuple(f[f"{self.key}/images"].shape)
            self._images = np.array(f[f"{self.key}/images"])
            self._seq = np.array(f[f"{self.key}/seq"])

    def __len__(self):
        """
        Return the number of frames in the stimulus.

        Returns
        -------
        int
            Number of frames.
        """
        return int(self._shape[-1])

    @property
    def images(self):
        return self._images

    @property
    def seq(self):
        return self._seq

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, idx):
        """
        Get the stimulus image at the given index.

        Parameters
        ----------
        idx : int
            Frame index.

        Returns
        -------
        np.ndarray
            Stimulus image at the index.
        """
        return self._images[..., int(idx)]


class MatFrames(object):

    def __init__(self, mat_path):
        """
        Initialize MatFrames for lazy access to MATLAB stimulus data.

        Parameters
        ----------
        mat_path : str or Path
            Path to the .mat file.
        """
        from scipy.io import loadmat  # lazy import

        m = loadmat(str(mat_path), simplify_cells=True)
        arr = None
        # common keys
        if "images" in m and isinstance(m["images"], np.ndarray):
            arr = m["images"]
        else:
            for root in ("stimulus", "params"):
                if root in m:
                    obj = m[root]
                    if (
                        isinstance(obj, dict)
                        and "images" in obj
                        and isinstance(obj["images"], np.ndarray)
                    ):
                        arr = obj["images"]
                        break
        if arr is None:
            # Fallback: take the only 3D/4D ndarray present
            cands = [
                v for v in m.values() if isinstance(v, np.ndarray) and v.ndim in (3, 4)
            ]
            if len(cands) == 1:
                arr = cands[0]
        if arr is None:
            raise KeyError("Could not locate stimulus images in %s" % mat_path)
        self._images = arr
        self._shape = tuple(arr.shape)

    def __len__(self):
        """
        Return the number of frames in the stimulus.

        Returns
        -------
        int
            Number of frames.
        """
        return int(self._shape[-1])

    @property
    def shape(self):
        return self._shape

    def __getitem__(self, idx):
        """
        Get the stimulus image at the given index.

        Parameters
        ----------
        idx : int
            Frame index.

        Returns
        -------
        np.ndarray
            Stimulus image at the index.
        """
        return np.asarray(self._images[..., int(idx)])


# ------------------------------ readers --------------------------------


def _read_params_from_mat(mat_path):
    """
    Read stimulus parameters from a MATLAB .mat file.

    Parameters
    ----------
    mat_path : str or Path
        Path to the .mat file.

    Returns
    -------
    dict
        Dictionary containing parameters like 'tr', 'prescan', etc.
    """
    from scipy.io import loadmat

    m = loadmat(str(mat_path), simplify_cells=True)
    p = m.get("params") or {}
    if not isinstance(p, dict):
        p = {}
    tr = float(_dig(p, "tr"))
    prescan = float(_dig(p, "prescanDuration", 0.0))
    start_scan = float(_dig(p, "startScan", 0.0))
    seq = _dig(p, "seq", None)
    seqtiming = _dig(p, "seqtiming", None)[1]
    nimg = _dig(p, "numImages", None)
    nimg = int(nimg) if nimg is not None else -1
    return {
        "tr": tr,
        "seq": seq,
        "seqtiming": seqtiming,
        "prescan": prescan,
        "start_scan": start_scan,
        "numImages": nimg,
    }


def _read_params_from_h5(h5_path):
    """
    Read stimulus parameters from an HDF5 file.

    Parameters
    ----------
    h5_path : str or Path
        Path to the HDF5 file.

    Returns
    -------
    dict
        Dictionary containing parameters from the 'params' group attributes.
    """
    import h5py

    out = {}
    with h5py.File(str(h5_path), "r") as f:
        if "params" not in f:
            raise KeyError(f"No 'params' group in {h5_path}")
        g = f["params"]
        for k, v in g.attrs.items():
            # decode bytes if needed
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            out[k] = v

    # sanity check
    if "tr" not in out:
        raise KeyError(f"'tr' missing in params.attrs of {h5_path}")
    return out


def _read_params_from_json(json_path):
    """
    Read stimulus parameters from a JSON file.

    Parameters
    ----------
    json_path : str or Path
        Path to the JSON file.

    Returns
    -------
    dict
        Dictionary containing parameters like 'tr', 'prescan', etc.
    """
    d = json.loads(Path(json_path).read_text())
    p = d.get("params") or {}
    tr = float(p["tr"])
    prescan = float(p.get("prescanDuration", 0.0))
    start_scan = float(p.get("startScan", 0.0))
    nimg = p.get("numImages", None)
    return {"tr": tr, "prescan": prescan, "start_scan": start_scan, "numImages": nimg}


def _nifti_timepoints(nii_path):
    """
    Get the number of timepoints from a NIfTI file.

    Parameters
    ----------
    nii_path : str or Path
        Path to the NIfTI file.

    Returns
    -------
    int
        Number of timepoints (4th dimension).
    """
    import nibabel as nib

    img = nib.load(str(nii_path))
    shp = img.shape
    if len(shp) < 4:
        raise ValueError("%s is not 4D" % nii_path)
    return int(shp[3])


def _dig(d, key, default=None):
    """
    Get value from dictionary with case-insensitive fallback.

    Parameters
    ----------
    d : dict
        Dictionary to search.
    key : str
        Key to look for.
    default : any, optional
        Default value if key not found.

    Returns
    -------
    any
        Value from dictionary or default.
    """
    if key in d:
        return d[key]
    # case-insensitive fallback
    for k in d.keys():
        if str(k).lower() == str(key).lower():
            return d[k]
    return default


# ----------------------- HDF5 discovery helper -------------------------


def _resolve_h5_candidates(bids_root, sub, ses, task, run):
    """
    Resolve candidate HDF5 stimulus files for the given BIDS identifiers.

    Parameters
    ----------
    bids_root : Path
        Root of the BIDS dataset.
    sub : str
        Subject identifier.
    ses : str
        Session identifier.
    task : str
        Task identifier.
    run : str or None
        Run identifier.

    Returns
    -------
    Path
        Path to the found HDF5 file.

    Raises
    ------
    FileNotFoundError
        If no suitable HDF5 file is found.
    """
    root = Path(bids_root)
    # run-scoped
    if run is not None:
        cands = [
            root
            / "sourcedata"
            / "stimuli"
            / f"sub-{sub}"
            / f"ses-{ses}"
            / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_stim.h5",
            root
            / "sourcedata"
            / "stimuli"
            / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_stim.h5",
        ]
        for p in cands:
            if p.is_file():
                return p
    # task-scoped
    cands = [
        root / "sourcedata" / "stimuli" / f"task-{task}_stim.h5",
        root
        / "sourcedata"
        / "stimuli"
        / f"sub-{sub}"
        / f"ses-{ses}"
        / f"task-{task}_stim.h5",
    ]
    for p in cands:
        if p.is_file():
            return p

    raise FileNotFoundError(
        f"No suitable .h5 stimulus found for sub-{sub} ses-{ses} task-{task} run-{run} in {bids_root}"
    )


# ----------------------- main resolution function ----------------------


def resolve_stim_source(ctx, bids_root, run, force_params):
    """
    Resolve and return a StimSpec for the stimulus source.

    Attempts to find stimulus data in order of precedence:
    1. Forced .mat file from config["force_params"]
    2. HDF5 files (run-scoped, then task-scoped)
    3. Vistadisp .mat files (run-level)
    4. Precomputed aperture NIfTI files

    Parameters
    ----------
    ctx : dict
        Context dictionary with sub, ses, task, etc.
    bids_root : Path
        Root of the BIDS dataset.
    run : str or None
        Run identifier.
    force_params : list or tuple or None
        Forced parameters for .mat file.

    Returns
    -------
    StimSpec or None
        StimSpec instance if found, None otherwise.
    """
    sub = ctx.get("sub")
    ses = ctx.get("ses")
    task = ctx.get("task")

    # ---- 1) forced .mat (accept list/tuple or JSON string) ----
    if force_params:
        fp = list(force_params)
        if len(fp) >= 2:
            forced_matfile, forced_task = fp[0], fp[1]

            if not str(forced_matfile).endswith(".mat"):
                forced_matfile = str(forced_matfile) + ".mat"

            mat_path = (
                bids_root / "sourcedata" / "vistadisplog" / forced_matfile
            ).resolve()
            if not mat_path.is_file():
                raise FileNotFoundError("force_params mat not found: %s" % mat_path)

            p = _read_params_from_mat(mat_path)
            frames = MatFrames(mat_path)
            return StimSpec(
                kind="forced_mat",
                task=str(forced_task),
                tr=p["tr"],
                prescan=p["prescan"],
                start_scan=p["start_scan"],
                n_frames=frames.shape[-1],
                frames=frames,
                aperture_nii=None,
                meta={"mat": str(mat_path)},
            )
        else:
            raise ValueError(
                "force_params must be a list/tuple of (matfile, task), got: %s"
                % force_params
            )

    # ----------- 2) auto-discover .h5 (task-matching) -----------------
    # If any .h5 matches this task (prefer run-scoped), use it.
    h5_path = None
    try:
        h5_path = _resolve_h5_candidates(bids_root, sub, ses, task, run)
    except FileNotFoundError:
        h5_path = None

    if h5_path:
        h = _read_params_from_h5(h5_path)
        frames = H5Frames(h5_path, key="stimulus")
        tf = h.get("tempFreq", None)
        seqtiming = (1.0 / float(tf)) if tf else None
        return StimSpec(
            kind="h5_auto",
            task=str(task),
            tr=h.get("tr"),
            prescan=h.get("prescan", 0.0),
            start_scan=h.get("start_scan", 0.0),
            seqtiming=seqtiming,
            n_frames=len(frames),
            frames=frames,
            aperture_nii=None,
            meta={"h5": str(h5_path)},
        )

    # ---- 3) vistadisplog run-level _params.mat ----
    if run is not None:
        cand = [
            bids_root
            / "sourcedata"
            / "vistadisplog"
            / f"sub-{sub}"
            / f"ses-{ses}"
            / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_params.mat",
            bids_root
            / "sourcedata"
            / "vistadisplog"
            / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_params.mat",
        ]
        for mp in cand:
            if mp.is_file():
                p = _read_params_from_mat(mp)
                frames = MatFrames(mp)
                return StimSpec(
                    kind="vistadisp_mat",
                    task=str(task),
                    tr=p["tr"],
                    prescan=p["prescan"],
                    start_scan=p["start_scan"],
                    n_frames=len(frames),
                    frames=frames,
                    aperture_nii=None,
                    meta={"mat": str(mp)},
                )

    # ---- 4) precomputed apertures (+ optional task JSON) ----
    ap = bids_root / "sourcedata" / "stimuli" / f"task-{task}_apertures.nii.gz"
    if ap.is_file():
        timing = None
        jp = bids_root / "sourcedata" / "stimuli" / f"task-{task}_params.json"
        if jp.is_file():
            timing = _read_params_from_json(jp)
        if timing is None:
            timing = {"tr": float("nan"), "prescan": 0.0, "start_scan": 0.0}
        nT = _nifti_timepoints(ap)
        return StimSpec(
            kind="precomp_aperture",
            task=str(task),
            tr=timing["tr"],
            prescan=timing["prescan"],
            start_scan=timing["start_scan"],
            n_frames=nT,
            frames=None,
            aperture_nii=ap,
            meta={
                "aperture": str(ap),
                "params_json": str(jp) if jp.is_file() else None,
            },
        )

    # nothing usable
    return None

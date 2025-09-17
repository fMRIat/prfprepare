# nii_to_surfNii.py

from pathlib import Path

import nibabel as nib
import numpy as np
from prfprepare_logging import get_logger


def _load_bold_data(in_path):
    """
    Load BOLD data from NIfTI or GIFTI file into a numpy array.

    Parameters
    ----------
    in_path : Path
        Path to the BOLD data file (.nii, .nii.gz, .gii, or .func.gii).

    Returns
    -------
    tuple
        (data, img, kind) where:
        - data : np.ndarray, BOLD data array
        - img : nibabel image object or None
        - kind : str, "nifti" or "gifti"

    Raises
    ------
    ValueError
        If NIfTI file is not 4D.
    """

    if isinstance(in_path, (Path, str)):
        suffixes = "".join(in_path.suffixes)
        img = nib.load(str(in_path))

        if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
            data = img.get_fdata(dtype=np.float32)
            if data.ndim != 4:
                raise ValueError(f"Expected 4D BOLD NIfTI; got shape {data.shape}")

        elif suffixes.endswith(".gii") or suffixes.endswith(".func.gii"):
            data = img.agg_data()
    else:
        img = in_path
        data = img.get_fdata()

    return data, img


def apply_masks_to_run(
    ctx,
    stim,
    roi_pack,
    output_only_average: bool = False,
    force: bool = False,
) -> dict:
    """
    Apply ROI masks from roi_pack to a 4D BOLD NIfTI and save masked output.

    Parameters
    ----------
    ctx : dict
        Context dictionary containing:
        - sub : str, subject identifier
        - ses : str, session identifier
        - task : str, task identifier
        - run : str, run identifier
        - hemi : str, hemisphere ('l' or 'r')
        - bold_path : Path, path to input BOLD file or nibabel image
        - func_out : Path, output directory
        - log : logger instance (optional)
        - verbose : bool for verbose logging (optional)
    stim : object
        Stimulus object with 'tr' attribute (repetition time).
    roi_pack : RoiPack
        ROI package handle for accessing masks.
    output_only_average : bool, optional
        If True, skip writing masked NIfTI output. Default is False.
    force : bool, optional
        If True, overwrite existing output files. Default is False.

    Returns
    -------
    dict or None
        Summary dictionary with processing results, or None if output_only_average=True.
        Contains information about applied masks and output files.
    """
    sub = ctx.get("sub")
    ses = ctx.get("ses")
    task = ctx.get("task")
    run = ctx.get("run")
    hemi = ctx.get("hemi")
    in_path = ctx["bold_path"]
    out_dir = ctx["func_out"]
    tr = float(stim.tr)

    LOG = ctx.get("log") or get_logger(
        __file__, verbose=bool(ctx.get("verbose", False))
    )

    nii_path = (
        out_dir
        / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_hemi-{hemi.upper()}_bold.nii.gz"
    )
    if force or not nii_path.exists():
        # mask NIfTI
        if output_only_average:
            return

        # --- load BOLD
        data, img = _load_bold_data(in_path)

        # Select correct grid (surface clears selection; volume picks grid by shape+affine)
        try:
            roi_pack.select_grid_for_input(img)
        except Exception:
            # If selection fails (legacy file), proceed with legacy union computation
            pass

        # Use union index; fallback only if union group missing
        u = roi_pack.get_masked_space_flat_index(hemi=hemi)
        flat = data.reshape(-1, data.shape[-1])
        masked = flat[u].astype(np.float32)

        newNii = nib.Nifti2Image(masked[:, None, None, :], affine=np.eye(4))
        newNii.header["pixdim"] = [1.0, 1.0, 1.0, tr, 1.0, 1.0, 1.0, 1.0]
        newNii.header["qform_code"] = 0
        newNii.header["sform_code"] = 0
        newNii.header["qoffset_x"] = 1.0
        newNii.header["qoffset_y"] = 1.0
        newNii.header["qoffset_z"] = 1.0
        newNii.header["cal_max"] = 1.0
        newNii.header["xyzt_units"] = 10
        nib.save(newNii, str(nii_path))
        LOG.info(f"Wrote masked BOLD: {nii_path} (T={data.shape[-1]})")

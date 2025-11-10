# stim_as_nii.py

from pathlib import Path
from shutil import copy2
from typing import Optional

import nibabel as nib
import numpy as np
from prfprepare_logging import get_logger


def _infer_output_name(task: Optional[str]) -> str:
    """
    Infer the output filename for stimulus apertures based on task name.

    Parameters
    ----------
    task : str or None
        Task identifier. If None or empty, uses default name.

    Returns
    -------
    str
        Output filename for the NIfTI file.
    """
    if task is None or str(task).strip() == "":
        return "stim_apertures.nii.gz"
    return f"task-{task}_apertures.nii.gz"


def generate_aperture_nii(
    ctx: dict,
    stim,
    use_numImages: bool = False,
) -> tuple[str, int]:
    """
    Generate a 4D NIfTI file of binary stimulus apertures from stimulus data.

    Creates binary aperture masks from stimulus frames, either by copying a
    precomputed aperture or by constructing one from stimulus frames. The
    resulting NIfTI contains binary masks where 1 indicates stimulus presence.

    Parameters
    ----------
    ctx : dict
        Context dictionary containing configuration parameters:
        - log : logger instance (optional)
        - verbose : bool for verbose logging (optional)
        - etcorrection : bool for eye-tracking correction (optional)
        - force : bool to overwrite existing files (optional)
        - stim_out : Path to output directory
    stim : object
        Stimulus object with attributes:
        - tr : float, repetition time in seconds
        - frames : object with images and seq attributes
        - task : str, task identifier (optional)
        - aperture_nii : str or Path, precomputed aperture file (optional)
        - prescan : float, prescan duration (optional)
        - seqtiming : array-like, sequence timing (optional)
        - numImages : int, number of images (optional)
    use_numImages : bool, optional
        If True, use stim.numImages to determine frame count.
        Default is False.

    Returns
    -------
    tuple[str, int]
        Path to the created NIfTI file and number of timepoints.

    Raises
    ------
    FileNotFoundError
        If precomputed aperture file is specified but not found.
    ValueError
        If required stimulus data is missing or invalid.
    NotImplementedError
        If eye-tracking correction is requested (not implemented).
    """
    LOG = ctx.get("log") or get_logger(
        __file__, verbose=bool(ctx.get("verbose", False))
    )
    etcorrection = ctx.get("etcorrection")
    force = ctx.get("force")
    out_dir = ctx.get("stim_out")

    if not etcorrection:
        # Target filename
        out_nii = out_dir / _infer_output_name(getattr(stim, "task", None))
        if out_nii.exists() and not force:
            out_nii_nT = nib.load(str(out_nii)).shape[-1]
            return (out_nii, out_nii_nT)

        # If a precomputed aperture is given, just copy it here (idempotent).
        precomp = getattr(stim, "aperture_nii", None)
        if precomp:
            LOG.debug(f"Using precomputed aperture: {precomp}")
            src = Path(precomp)
            if not src.exists():
                raise FileNotFoundError(f"Precomputed aperture not found: {src}")
            if force or not out_nii.exists():
                copy2(src, out_nii)
            out_nii_nT = nib.load(str(out_nii)).shape[-1]
            LOG.debug(f"Copied precomputed aperture to {out_nii} (T={out_nii_nT})")
            return (out_nii, out_nii_nT)

        # Otherwise, we must construct from frames
        frames = getattr(stim, "frames", None)
        if frames is None:
            raise ValueError(
                "stim.frames is None and no precomputed aperture available."
            )

        images = getattr(getattr(stim, "frames", None), "images", None)
        seq = getattr(getattr(stim, "frames", None), "seq", None)
        tr = getattr(stim, "tr", None)
        prescan = getattr(stim, "prescan", 0.0)

        # get the seqtiming and compute shift
        seqTiming = getattr(stim, "seqtiming", None)
        shift = tr / 2 / seqTiming if seqTiming is not None else 0

        # Resolve time dimension limit
        if use_numImages:
            numImages = getattr(stim, "numImages", None)
            if numImages is None:
                raise ValueError("stim.numImages is not set but use_numImages=True")
        else:
            LOG.debug("Using seqtiming from stim")
            numImages = len(seq) * seqTiming / tr

        # add prescan frames
        n_with_prescan = int(round(numImages + prescan / tr))

        # Compute frame indices to pick
        idx = (
            np.linspace(0, len(seq) - 1, n_with_prescan, dtype=int, endpoint=False)
            + int(shift)
        )

        if idx.size == 0:
            raise ValueError(
                "Computed zero frames to export (idx.size == 0). Check params/seqtiming/tr."
            )

        LOG.debug("Constructing apertures from frames...")
        # Gather frames: picked_images shape (H, W, T_with_prescan)
        picked_images = images[:, :, seq[idx]]

        # Remove prescan-duration from the beginning
        if prescan > 0:
            n_drop = int(round(prescan / tr))
            if n_drop >= picked_images.shape[-1]:
                raise ValueError(
                    f"Prescan removal would drop all frames: prescan={prescan}, tr={tr}, T={picked_images.shape[-1]}."
                )
            picked_images = picked_images[:, :, n_drop:]
            LOG.debug(
                f"Prescan={prescan:.3f}, removing first {n_drop} volumes; new shape: {picked_images.shape}",
            )

        # Binarize → uint8 apertures, reshape to (H, W, 1, T)
        H, W, T = picked_images.shape
        a, a_count = np.unique(picked_images, return_counts=True)
        apertures = (picked_images != a[np.argmax(a_count)]).astype(np.uint8)
        apertures = apertures.reshape(H, W, 1, T)

        # Write NIfTI
        affine = np.eye(4, dtype=np.float32)
        img = nib.Nifti1Image(apertures, affine)
        hdr = img.header
        hdr["pixdim"][1:5] = (1.0, 1.0, 1.0, float(tr))
        hdr["qoffset_x"] = 1.0
        hdr["qoffset_y"] = 1.0
        hdr["qoffset_z"] = 1.0
        hdr["qform_code"] = 0
        hdr["sform_code"] = 0
        hdr["cal_max"] = 1.0
        hdr["xyzt_units"] = 10  # mm and seconds
        nib.save(img, str(out_nii))
        LOG.info(f"Wrote apertures NIfTI: {out_nii} (T={T}, TR={tr})")

        return (out_nii, T)

    else:
        raise NotImplementedError(
            "etcorrection=True is not implemented in generate_aperture_nii()."
        )


""" old stuff, eyetracker correction currently not implemented
%% do the shifting for ET corr
if etcorr and not forceParams:
note("etcorr is true and it will do it now")
# base paths
outPET = outP.replace("/sub-", "_ET/sub-")

# create the output folders
makedirs(outPET, exist_ok=True)

for sesI, ses in enumerate(sess):
    logPs = np.array(
        glob(
            path.join(
                bidsDir,
                "sourcedata",
                "vistadisplog",
                f"sub-{sub}",
                f"ses-{ses}",
                "*_params.mat",
            )
        )
    )

    for logP in logPs:
        try:
            stim = path.basename(
                loadmat(logP, simplify_cells=True)["params"]["loadMatrix"]
            )
        except TypeError:
            stimP = glob(
                path.join(bidsDir, "sourcedata", "stimuli", "*.mat")
            )
            if len(stimP) == 1:
                stimP = stimP[0]
                print(
                    "There is no stimulus file defined in the params file (params.loadMatrix)!"
                )
                print(
                    f"We will use the only stimulus file we found: {stimP}!"
                )
            elif len(stimP) > 1:
                print(
                    "There is no stimulus file defined in the params file (params.loadMatrix)!"
                )
                print(
                    "We found more than one stimulus file in the stimuli folder, please define one in the prams or remove all but one in the stimuli folder!"
                )
        else:
            stimP = path.join(bidsDir, "sourcedata", "stimuli", stim)

        if forceParams:
            task = forceTask
        else:
            task = logP.split("task-")[-1].split("_run")[0]
        run = logP.split("run-")[-1].split("_")[0]

        makedirs(path.join(outPET, "stimuli"), exist_ok=True)
        oFname = path.join(
            outPET,
            "stimuli",
            f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_apertures.nii.gz",
        )
        gazeFile = path.join(
            bidsDir,
            "sourcedata",
            "etdata",
            f"sub-{sub}",
            f"ses-{ses}",
            f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_gaze.mat",
        )

        if not path.isfile(gazeFile):
            print(f"Gaze file not found at {gazeFile}")
            print("switching off eyetracker correction!")
            etcorr = False

        if not path.isfile(oFname) or force:
            if not path.isfile(stimP):
                raise Warning(f"Did not find stim File: {stimP}")

            # loat the mat files defining the stimulus
            imagesFile = loadmat(stimP, simplify_cells=True)
            params = loadmat(logP, simplify_cells=True)

            # get all values necessary
            seq = imagesFile["stimulus"]["seq"]
            seqTiming = imagesFile["stimulus"]["seqtiming"]
            images = imagesFile["stimulus"]["images"]
            tr = params["params"]["tr"]

            # build and binarise the stimulus
            stimImagesU, stimImagesUC = np.unique(
                images, return_counts=True
            )
            images[images != stimImagesU[np.argmax(stimImagesUC)]] = 1
            images[images == stimImagesU[np.argmax(stimImagesUC)]] = 0

            picked_images = images[:, :, seq[:: int(1 / seqTiming[1] * tr)] - 1]

            # load the gaze file and do the gaze correction
            gaze = loadmat(gazeFile, simplify_cells=True)

            # get rid of out of image data (loss of tracking)
            gaze["x"][np.any((gaze["x"] == 0, gaze["x"] == 1280), 0)] = (
                1280 / 2
            )  # 1280 comes from resolution of screen
            gaze["y"][np.any((gaze["y"] == 0, gaze["y"] == 1024), 0)] = (
                1024 / 2
            )  # 1024 comes from resolution of screen
            # TODO: load the resolution from the _params.mat file?

            # resamplet to TR
            x = np.array(
                [
                    np.mean(f)
                    for f in np.array_split(gaze["x"], picked_images.shape[2])
                ]
            )
            y = np.array(
                [
                    np.mean(f)
                    for f in np.array_split(gaze["y"], picked_images.shape[2])
                ]
            )

            # demean the ET data
            x -= x.mean()
            y -= y.mean()
            y = (
                -y
            )  # there is a flip between ET and fixation dot sequece (pixel coordinates),
            # with this the ET data is in the same space as fixation dot seq.

            # TODO: we problably shoud make a border around the actual stim and then
            #       place the original stim in the center before shifting it so that
            #       more peripheral regions could also be stimulated.
            #       for the analysis the new width (zB 8° radius)
            # border = 33 for +1° radius?
            # shiftStim = np.zeros((picked_images.shape[0]+2*border,picked_images.shape[1]+2*border,picked_images.shape[2]))

            # shift the stimulus opposite of gaze direction
            for i in range(len(x)):
                picked_images[..., i] = shift(
                    picked_images[..., i],
                    (-y[i], -x[i]),
                    mode="constant",
                    cval=0,
                )

            # save the stimulus as nifti
            img = nib.Nifti1Image(
                picked_images[:, :, None, :].astype(float), np.eye(4)
            )
            img.header["pixdim"][1:5] = [1, 1, 1, tr]
            img.header["qoffset_x"] = img.header["qoffset_y"] = img.header[
                "qoffset_z"
            ] = 1
            img.header["cal_max"] = 1
            img.header["xyzt_units"] = 10
            nib.save(img, oFname)

return etcorr
"""

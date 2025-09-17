# link_stimuli.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
from prfprepare_logging import get_logger


def write_events(
    ctx, stim, timepoints: int, stim_name: str, output_only_average: bool
) -> Path:
    """
    Write BIDS events.tsv file for a given run and task.

    Creates a BIDS-compliant events file containing onset times, durations,
    stimulus file names, and indices for each timepoint in the run.

    Parameters
    ----------
    ctx : dict
        Context dictionary containing:
        - sub : str, subject identifier
        - ses : str, session identifier
        - task : str, task identifier
        - run : str, run identifier
        - func_out : Path, output directory for functional data
        - force : bool, overwrite existing files (optional)
        - log : logger instance (optional)
        - verbose : bool for verbose logging (optional)
    stim : object
        Stimulus object with 'tr' attribute (repetition time in seconds).
    timepoints : int
        Number of timepoints in the run.
    stim_name : str
        Name of the stimulus file for the events.
    output_only_average : bool
        If True, skip creating events file.

    Returns
    -------
    Path or None
        Path to the created events.tsv file, or None if output_only_average=True.
    """
    sub = ctx.get("sub")
    ses = ctx.get("ses")
    task = ctx.get("task")
    run = ctx.get("run")
    out_dir = ctx.get("func_out")
    force = ctx.get("force", False)

    LOG = ctx.get("log") or get_logger(
        __file__, verbose=bool(ctx.get("verbose", False))
    )

    if output_only_average:
        LOG.debug("Skipping events.tsv creation due to output_only_average=True")
        return

    events_path = out_dir / f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_events.tsv"

    if events_path.exists() and not force:
        LOG.debug(f"Events already exist, skipping: {events_path}")
        return events_path

    tr = float(getattr(stim, "tr"))

    with open(events_path, "w") as f:
        f.write("onset\tduration\tstim_file\tstim_file_index\n")
        for i in range(timepoints):
            f.write(f"{i * tr:.3f}\t{tr:.3f}\t{stim_name}\t{i + 1}\n")
    LOG.debug(f"Wrote events.tsv: {events_path} (T={timepoints}, TR={tr})")
    return events_path


""" old stuff, eyetracker correction currently not implemented
                # create events.tsv for ET corr
                if etcorr:
                    if average and len(runs) > 1:
                        die("We can not do eyetracker correction on averaged runs!")

                    outPET = outP.replace("/sub-", "_ET/sub-")
                    if path.isdir(outPET):
                        newTSV = path.join(
                            outPET,
                            f"ses-{ses}",
                            "func",
                            f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_events.tsv",
                        )

                        stimNameET = path.join(
                            outPET,
                            "stimuli",
                            f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_apertures.nii.gz",
                        )

                        if not path.isfile(newTSV) or force:
                            if not path.isfile(stimNameET):
                                print(f"did not find {stimNameET}!")
                                continue

                            nii = nib.load(stimNameET)
                            TR = nii.header["pixdim"][4]
                            nT = nii.shape[3]

                            # fill the events file
                            with open(newTSV, "w") as file:
                                file.writelines(
                                    "onset\tduration\tstim_file\tstim_file_index\n"
                                )

                                for i in range(nT):
                                    file.write(
                                        f'{i*TR:.3f}\t{TR:.3f}\t{stimNameET.split("/")[-1]}\t{i+1}\n'
                                    )

                    else:
                        print("No eyetracker analysis-XX folder found!")
"""

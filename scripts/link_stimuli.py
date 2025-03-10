# %%
import nibabel as nib
from os import path
import numpy as np
from glob import glob
import sys


def link_stimuli(
    sub,
    sess,
    tasks,
    layout,
    bidsDir,
    outP,
    etcorr,
    forceParams,
    average,
    output_only_average,
    force,
    verbose,
):
    """
    Here we link the _bold.nii.gz files to the corresponding stimulus files
    by creating _events.tsv files which contain the stimulus information.
    Those files are specific for sub,ses,task and runs;
    but not specific for the area or atlas.
    """

    def die(*args):
        print(*args)
        sys.exit(1)

    def note(*args):
        if verbose:
            print(*args)
        return None

    if forceParams:
        forceParamsFile, forceTask = forceParams

    for sesI, ses in enumerate(sess):
        if not tasks:
            tasks = layout.get(subject=sub, session=ses, return_type="id", target="task")
        if forceParams:
            tasks = [forceTask]

        for task in tasks:
            apertures = np.array(glob(path.join(outP, "stimuli", "task-*.nii.gz")))
            try:
                stimName = apertures[[f"task-{task}_" in ap for ap in apertures]].item()
            except:
                print(f"Did not find task-{task} in {apertures}.")
                continue

            runs = layout.get(
                subject=sub, session=ses, task=task, return_type="id", target="run"
            )
            # adapt for averaged runs
            if average and len(runs) > 1:
                if output_only_average:
                    runs = ["".join(map(str, runs)) + "avg"]
                else:
                    runs.append("".join(map(str, runs)) + "avg")

            for run in runs:
                # create events.tsv
                newTSV = path.join(
                    outP,
                    f"ses-{ses}",
                    "func",
                    f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_events.tsv",
                )

                if not path.isfile(newTSV) or force:
                    with open(newTSV, "w") as file:
                        nii = nib.load(stimName)
                        TR = nii.header["pixdim"][4]
                        nT = nii.shape[3]

                        # fill the events file
                        file.writelines("onset\tduration\tstim_file\tstim_file_index\n")

                        for i in range(nT):
                            file.write(
                                f'{i*TR:.3f}\t{TR:.3f}\t{stimName.split("/")[-1]}\t{i+1}\n'
                            )

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

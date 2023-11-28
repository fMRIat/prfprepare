# %%
import copy
import json
import sys
from glob import glob
from os import makedirs, path

import bids
import neuropythy as ny
import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.io import loadmat
from scipy.ndimage import grey_dilation
from nibabel.funcs import four_to_three


###############################################################################
def nii_to_surfNii(
    sub,
    sess,
    layout,
    bidsDir,
    subInDir,
    outP,
    fsDir,
    forceParams,
    fmriprepLegacyLayout,
    average,
    output_only_average,
    atlases,
    roisIn,
    analysisSpace,
    force,
    verbose,
):
    """
    This function converts the surface _bold.func.gz files to 2D nifti2 files
    where every pixel contains one vertex timecourse. Different ROIs specified
    are merged into one mask and we output one nifti2 file containing all
    voxel data within any of the defined ROIs. This allows for minimising
    computation time since vertices contained in e.g. wang-V1 and benson-V1
    only have to be analysed once! Further, for ever atals and ROI we output
    one sidecar .json giving all information about which indices wihin the
    nifti file belongs to the ROI and which indicices in fs space correspond.

    Additionally the first timepoints are removed as defined in PrescanDuration
    as well as startScan in the _params.mat file.
    """

    def note(*args):
        if verbose:
            print(*args)
        return None

    def die(*args):
        print(*args)
        sys.exit(1)

    if forceParams:
        forceParamsFile, forceTask = forceParams

    for hemi in ["l", "r"]:
        # first get the total number of vertices

        if analysisSpace == "fsnative":
            nVertices = len(
                nib.freesurfer.io.read_geometry(
                    path.join(fsDir, f"sub-{sub}", "surf", f"{hemi.lower()}h.pial")
                )[0]
            )

            # define the empty mask
            allROImask = np.zeros(nVertices)

            # loop over all defined atlases
            for atlas in atlases:
                if atlas != "fullBrain":
                    allROImask = getAllROImask(
                        sub,
                        fsDir,
                        atlas,
                        roisIn,
                        hemi,
                        allROImask,
                        analysisSpace,
                        verbose,
                    )

            # define the json files for the found mask
            # loop over all defined atlases
            for atlas in atlases:
                # load in the atlas
                if atlas == "fullBrain":
                    atlasName = "fullBrain"
                    areaLabels = {"fullBrain": -1}
                    rois = ["fullBrain"]
                    areas = [-1]
                else:
                    # load in the atlas
                    areas, areaLabels, rois, atlasName = load_atlas(
                        atlas, fsDir, sub, hemi, roisIn, analysisSpace, verbose
                    )

                # go for all given ROIs
                for roi in rois:
                    # if we want fullBrain change the mask to all ones
                    hemi_str = hemi

                    if roi == "fullBrain":
                        thisROImask = np.ones(allROImask.shape)
                        allROImask = np.ones(allROImask.shape)

                    else:
                        # else we adapt the mask for the roi
                        # get labels associated with ROI
                        roiLabels = [
                            value for key, value in areaLabels.items() if roi in key
                        ]

                        if not roiLabels:
                            continue

                        thisROImask = np.any([areas == lab for lab in roiLabels], 0)

                    # define a list of all appliccable boldFiles
                    for sesI, ses in enumerate(sess):
                        boldFiles = []
                        funcOutP = path.join(outP, f"ses-{ses}", "func")
                        makedirs(funcOutP, exist_ok=True)

                        jsonP = path.join(
                            funcOutP,
                            f"sub-{sub}_ses-{ses}_hemi-{hemi_str.upper()}_desc-{roi}-{atlasName}_maskinfo.json",
                        )
                        if not path.isfile(jsonP):
                            if forceParams:
                                tasks = [forceTask]
                            else:
                                tasks = layout.get_tasks(subject=sub, session=ses)

                            for task in tasks:
                                runs = layout.get_runs(
                                    subject=sub, session=ses, task=task
                                )

                                # adapt for averaged runs
                                if average and len(runs) > 1:
                                    if output_only_average:
                                        runs = ["".join(map(str, runs)) + "avg"]
                                    else:
                                        runs.append("".join(map(str, runs)) + "avg")
                                for run in runs:
                                    boldFiles.append(
                                        f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_hemi-{hemi_str.upper()}_bold.nii.gz"
                                    )

                            # define the json  for this specific atlas-roi combi for one subject and session
                            jsonP = path.join(
                                funcOutP,
                                f"sub-{sub}_ses-{ses}_hemi-{hemi_str.upper()}_desc-{roi}-{atlasName}_maskinfo.json",
                            )
                            jsonI = {
                                "atlas": atlasName,
                                "roi": roi,
                                "hemisphere": hemi,
                                "thisHemiSize": int(allROImask.sum()),
                                "boldFiles": boldFiles,
                                "roiIndFsnative": np.where(thisROImask)[0].tolist(),
                                "roiIndBold": np.where(
                                    thisROImask[allROImask.astype(bool)]
                                )[0].tolist(),
                            }
                            if len(jsonI["roiIndFsnative"]) != len(jsonI["roiIndBold"]):
                                die("Something wrong with the Indices!!")
                            with open(jsonP, "w") as fl:
                                json.dump(jsonI, fl, indent=4)

        elif analysisSpace == "volume":
            # load the GM mask from freesurfer for the receptive hemi
            hemiRibbon = nib.load(
                path.join(fsDir, f"sub-{sub}", "mri", f"{hemi}h.ribbon.mgz")
            )

            # load an example bold image
            if forceParams:
                task = forceTask
            else:
                task = layout.get_tasks(subject=sub, session=sess[0])[0]

            boldref4d = nib.load(
                glob(
                    path.join(
                        subInDir,
                        f"ses-{sess[0]}",
                        "func",
                        f"sub-{sub}_ses-{sess[0]}_task-{task}_run-*_space-T1w_desc-preproc_bold.nii*",
                    )
                )[0]
            )

            boldref = four_to_three(boldref4d)[0]

            allROImask = np.zeros(boldref.shape)

            for atlas in atlases:
                if atlas not in ["benson", "wang", "fullBrain"]:
                    print(
                        "With analysisSpace==volume you can only use atlases [benson, wang, fullBrain]!"
                    )
                    atlases.remove(atlas)
                    continue

                if atlas != "fullBrain":
                    resDilRibbonShape = reslice_atlas(
                        atlas, sub, hemi, fsDir, hemiRibbon, boldref
                    )

                    resDilRibbonNum = 0
                    while resDilRibbonShape != allROImask.shape:
                        resDilRibbonNum += 1
                        resDilRibbonShape = reslice_atlas(
                            atlas,
                            sub,
                            hemi,
                            fsDir,
                            hemiRibbon,
                            boldref,
                            resDilRibbonNum,
                        )

                    allROImask = getAllROImask(
                        sub,
                        fsDir,
                        atlas,
                        roisIn,
                        hemi,
                        allROImask,
                        analysisSpace,
                        verbose,
                        resDilRibbonNum,
                    )

            # define the json files for the found mask
            # loop over all defined atlases
            for atlas in atlases:
                # load in the atlas
                if atlas == "fullBrain":
                    atlasName = "fullBrain"
                    areaLabels = {"fullBrain": -1}
                    rois = ["fullBrain"]
                    areas = [-1]
                else:
                    areas, areaLabels, rois, atlasName = load_atlas(
                        atlas,
                        fsDir,
                        sub,
                        hemi,
                        roisIn,
                        analysisSpace,
                        verbose,
                        resDilRibbonNum,
                    )

                # go for all given ROIs
                for roi in rois:
                    # if we want fullBrain change the mask to all ones
                    if roi == "fullBrain":
                        if hemi == "r":
                            continue
                        hemi_str = "both"
                        thisROImask = np.ones(allROImask.shape)
                        allROImask = np.ones(allROImask.shape)
                    else:
                        hemi_str = hemi

                        # else we adapt the mask for the roi
                        # get labels associated with ROI
                        roiLabels = [
                            value for key, value in areaLabels.items() if roi in key
                        ]

                        if not roiLabels:
                            continue

                        thisROImask = np.any([areas == lab for lab in roiLabels], 0)

                    # define a list of all appliccable boldFiles
                    for sesI, ses in enumerate(sess):
                        boldFiles = []
                        funcOutP = path.join(outP, f"ses-{ses}", "func")
                        makedirs(funcOutP, exist_ok=True)

                        jsonP = path.join(
                            funcOutP,
                            f"sub-{sub}_ses-{ses}_hemi-{hemi_str.upper()}_desc-{roi}-{atlasName}_maskinfo.json",
                        )
                        if not path.isfile(jsonP):
                            if forceParams:
                                tasks = [forceTask]
                            else:
                                tasks = layout.get_tasks(subject=sub, session=ses)

                            for task in tasks:
                                runs = layout.get_runs(
                                    subject=sub, session=ses, task=task
                                )

                                # adapt for averaged runs
                                if average and len(runs) > 1:
                                    if output_only_average:
                                        runs = ["".join(map(str, runs)) + "avg"]
                                    else:
                                        runs.append("".join(map(str, runs)) + "avg")
                                for run in runs:
                                    boldFiles.append(
                                        f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_hemi-{hemi_str.upper()}_bold.nii.gz"
                                    )

                            # define the json  for this specific atlas-roi combi for one subject and session
                            jsonI = {
                                "atlas": atlasName,
                                "roi": roi,
                                "hemisphere": hemi,
                                "thisHemiSize": int(allROImask.sum()),
                                "boldFiles": boldFiles,
                                "origImageSize": list(allROImask.shape),
                                "roiPos3D": np.array(np.where(thisROImask)).T.tolist(),
                                "roiIndBold": np.where(
                                    thisROImask[allROImask.astype(bool)]
                                )[0].tolist(),
                            }
                            with open(jsonP, "w") as fl:
                                json.dump(jsonI, fl, indent=4)

        else:
            die(
                f"Your analysisSpace {analysisSpace} is not supported! "
                "Please choose from [fsaverage, volume]"
            )

        # now lets apply the merged mask to all bold files
        for sesI, ses in enumerate(sess):
            # note(f'[nii_to_sufNii.py] Working on sub-{sub} ses-{ses} hemi-{hemi_str.upper()}')
            funcInP = path.join(subInDir, f"ses-{ses}", "func")
            funcOutP = path.join(outP, f"ses-{ses}", "func")

            if forceParams:
                tasks = [forceTask]
            else:
                tasks = layout.get_tasks(subject=sub, session=ses)

            for task in tasks:
                runs = layout.get_runs(subject=sub, session=ses, task=task)
                # adapt for averaged runs
                if average and len(runs) > 1:
                    runsOrig = copy.copy(runs)
                    if output_only_average:
                        runs = ["".join(map(str, runs)) + "avg"]
                    else:
                        runs.append("".join(map(str, runs)) + "avg")
                for run in runs:
                    # check if already exists, if not force skip
                    # if not path.exists(newNiiP) or force:

                    # name the output files
                    newNiiP = path.join(
                        funcOutP,
                        f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_hemi-{hemi_str.upper()}_bold.nii.gz",
                    )
                    if not path.isfile(newNiiP):
                        note(
                            f"[nii_to_sufNii.py] Working on {path.basename(newNiiP)}..."
                        )
                        skip = False

                        if "av" not in str(run):
                            # load the .gii in fsnative
                            if analysisSpace == "fsnative":
                                if fmriprepLegacyLayout:
                                    giiP = path.join(
                                        funcInP,
                                        f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-fsnative_hemi-{hemi_str.upper()}_bold.func.gii",
                                    )
                                else:
                                    giiP = path.join(
                                        funcInP,
                                        f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_hemi-{hemi_str.upper()}_space-fsnative_bold.func.gii",
                                    )

                                # get the data data
                                try:
                                    data = nib.load(giiP).agg_data()
                                except IndexError as error:
                                    print(f"File {giiP} not found, skipping...")
                                    continue

                            # or volume file in T1 space
                            elif analysisSpace == "volume":
                                # get the data
                                try:
                                    niiP = glob(
                                        path.join(
                                            funcInP,
                                            f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-T1w_desc-preproc_bold.nii*",
                                        )
                                    )[0]
                                    data = np.asarray(nib.load(niiP).get_fdata())
                                except FileNotFoundError:
                                    print(f"File {niiP} not found, skipping...")
                                    continue
                                except IndexError:
                                    print(f"File not found, skipping...")
                                    continue

                        else:
                            datas = []
                            for r in runsOrig:
                                if analysisSpace == "fsnative":
                                    if fmriprepLegacyLayout:
                                        giiP = path.join(
                                            funcInP,
                                            f"sub-{sub}_ses-{ses}_task-{task}_run-{r}_space-fsnative_hemi-{hemi_str.upper()}_bold.func.gii",
                                        )
                                    else:
                                        giiP = path.join(
                                            funcInP,
                                            f"sub-{sub}_ses-{ses}_task-{task}_run-{r}_hemi-{hemi_str.upper()}_space-fsnative_bold.func.gii",
                                        )
                                    try:
                                        datas.append(nib.load(giiP).agg_data())
                                    except FileNotFoundError:
                                        continue

                                # or volume file in T1 space
                                elif analysisSpace == "volume":
                                    # get the data data
                                    try:
                                        niiP = glob(
                                            path.join(
                                                funcInP,
                                                f"sub-{sub}_ses-{ses}_task-{task}_run-{r}_space-T1w_desc-preproc_bold.nii*",
                                            )
                                        )[0]
                                        datas.append(
                                            np.asarray(nib.load(niiP).get_fdata())
                                        )
                                    except FileNotFoundError:
                                        print(f"File {niiP} not found, skipping...")
                                        continue
                                    except IndexError:
                                        print(f"File not found, skipping...")
                                        continue

                            if len(datas) > 1:
                                # crop them to the same length for averaging
                                giiMinLength = min([g.shape[-1] for g in datas])
                                gii = [g[..., :giiMinLength] for g in datas]
                                # average the runs
                                data = np.mean(gii, 0)
                            else:
                                try:
                                    data = np.array(datas[0])
                                except IndexError as error:
                                    print(f"No data found to average, skipping...")
                                    skip = True

                        if not skip:
                            # apply the combined ROI mask
                            data = data[allROImask.astype(bool), :]

                            # get rid of volumes where the stimulus showed only blank (prescanDuration)
                            if forceParams:
                                params_path = path.join(
                                    bidsDir,
                                    "sourcedata",
                                    "vistadisplog",
                                    forceParamsFile,
                                )
                            else:
                                if "av" not in str(run):
                                    params_path = path.join(
                                        bidsDir,
                                        "sourcedata",
                                        "vistadisplog",
                                        f"sub-{sub}",
                                        f"ses-{ses}",
                                        f"sub-{sub}_ses-{ses}_task-{task}_run-{run}_params.mat",
                                    )
                                else:
                                    params_path = path.join(
                                        bidsDir,
                                        "sourcedata",
                                        "vistadisplog",
                                        f"sub-{sub}",
                                        f"ses-{ses}",
                                        f"sub-{sub}_ses-{ses}_task-{task}_run-01_params.mat",
                                    )

                            if path.isfile(params_path):
                                params = loadmat(params_path, simplify_cells=True)
                            else:
                                params_path = path.join(
                                    bidsDir,
                                    "sourcedata",
                                    "stimuli",
                                    f"task-{task}_params.json",
                                )
                                note(
                                    "we used the params file that comes with the precomputed stimulus!"
                                )
                                note(params_path)
                                with open(params_path, "r") as fl:
                                    params = json.load(fl)

                            tr = params["params"]["tr"]

                            if "prescanDuration" in params["params"].keys():
                                prescan = params["params"]["prescanDuration"]

                                if prescan > 0:
                                    note(
                                        f"Removing {prescan // tr} volumes from the beginning due to prescan"
                                    )
                                    data = data[:, prescan // tr :]
                            else:
                                prescan = 0

                            # remove volumes the stimulus was wating to start (startScan)
                            if "startScan" in params["params"].keys():
                                startScan = params["params"]["startScan"]

                                if startScan > 0:
                                    note(
                                        f"Removing {startScan // tr} volumes from the beginning due to startScan"
                                    )
                                    data = data[:, startScan // tr :]
                            else:
                                startScan = 0

                            # create and save new nii img
                            try:
                                apertures = np.array(
                                    glob(
                                        path.join(
                                            outP, "stimuli", "task-*_apertures.nii.gz"
                                        )
                                    )
                                )
                                stimNii = nib.load(
                                    apertures[
                                        [f"task-{task}_" in ap for ap in apertures]
                                    ].item()
                                )
                            except:
                                print(
                                    f'could not find task-{task} in {path.join(outP, "stimuli")}!'
                                )
                                continue

                            # trim data to stimulus length, gets rid of volumes when the
                            # scanner was running for longer than the task and is topped manually
                            stimLength = stimNii.shape[-1]
                            if data.shape[1] < stimLength:
                                die(
                                    f"For {path.basename(newNiiP)} the data is shorter than "
                                    f"the simulus file ({data.shape[1]}<{stimLength})"
                                )
                            elif data.shape[1] > stimLength:
                                data = data[:, :stimLength]
                            else:
                                pass

                            # save the new nifti
                            newNii = nib.Nifti2Image(
                                data[:, None, None, :].astype("float32"),
                                affine=np.eye(4),
                            )
                            newNii.header["pixdim"] = stimNii.header["pixdim"]
                            newNii.header["qoffset_x"] = 1
                            newNii.header["qoffset_y"] = 1
                            newNii.header["qoffset_z"] = 1
                            newNii.header["cal_max"] = 1
                            newNii.header["xyzt_units"] = 10
                            nib.save(newNii, newNiiP)


def getAllROImask(
    sub,
    fsDir,
    atlas,
    roisIn,
    hemi,
    allROImask,
    analysisSpace,
    verbose,
    resDilRibbonNum=0,
):
    # find the merged mask
    def note(*args):
        if verbose:
            print(*args)
        return None

    def die(*args):
        print(*args)
        sys.exit(1)

    # load in the atlas
    areas, areaLabels, rois, atlasName = load_atlas(
        atlas, fsDir, sub, hemi, roisIn, analysisSpace, verbose, resDilRibbonNum
    )

    # go for all given ROIs
    for roi in rois:
        # get labels associated with ROI
        roiLabels = [value for key, value in areaLabels.items() if roi in key]

        if not roiLabels:
            note(f"We could not find {roi} in atlas {atlas}, continue...")
            continue

        thisROImask = np.any([areas == lab for lab in roiLabels], 0)
        allROImask = np.any((allROImask, thisROImask), 0)
    return allROImask


def load_atlas(
    atlas, fsDir, sub, hemi, rois, analysisSpace, verbose, resDilRibbonNum=0
):
    def note(*args):
        if verbose:
            print(*args)
        return None

    def die(*args):
        print(*args)
        sys.exit(1)

    if analysisSpace == "fsnative":
        atlasF = "surf"
        atlasPre = f"{hemi}h."
    elif analysisSpace == "volume":
        atlasF = "mri"
        atlasPre = f"{hemi}h.res_dil_{resDilRibbonNum}_"

    if atlas == "benson":
        areasP = path.join(fsDir, f"sub-{sub}", atlasF, f"{atlasPre}benson14_varea.mgz")
        if not path.exists(areasP):
            die(f"We could not find the benson atlas fiel: {areasP}")

        # load the label files
        areas = nib.load(areasP).get_fdata().squeeze()

        # load the label area dependency
        mdl = ny.vision.retinotopy_model("benson17", f"{hemi}h")
        areaLabels = dict(mdl.area_id_to_name)
        areaLabels = {areaLabels[k]: k for k in areaLabels}
        labelNames = list(areaLabels.keys())

        if rois[0] == "all":
            rois = labelNames

        atlasName = atlas

    elif atlas == "wang":
        areasP = path.join(fsDir, f"sub-{sub}", atlasF, f"{atlasPre}wang15_mplbl.mgz")
        if not path.exists(areasP):
            die(f"We could not find the wang atlas file: {areasP}")

        # load the label files
        areas = nib.load(areasP).get_fdata().squeeze()

        labelNames = [
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
        areaLabels = {labelNames[k]: k for k in range(len(labelNames))}

        if rois[0] == "all":
            rois = labelNames[1:]

        atlasName = atlas

    elif "annot" in atlas:
        if analysisSpace == "volume":
            return [], [], [], []

        if hemi + "h." in atlas:
            annotP = path.join(fsDir, f"sub-{sub}", "customLabel", atlas)

            a, c, l = nib.freesurfer.io.read_annot(annotP)
            areas = a + 1
            areaLabels = {"Unknown": 0} | {
                l[k].decode("utf-8"): k + 1 for k in range(len(l))
            }

            if rois[0] == "all":
                rois = list(areaLabels.keys())[1:]

            atlasName = atlas.split(".")[1]
        else:
            return [], [], [], []

    else:
        die(
            f"You specified a wrong atlas ({atlas}), please choose from [benson, wang, fs_custom]!"
        )

    return areas, areaLabels, rois, atlasName


def reslice_atlas(atlas, sub, hemi, fsDir, hemiRibbon, boldref, num=0):
    # we get the atlases as volumes
    if atlas == "benson":
        atlasName = "benson14_varea.mgz"
    elif atlas == "wang":
        atlasName = "wang15_mplbl.mgz"

    res_dil_name = path.join(
        fsDir, f"sub-{sub}", "mri", f"{hemi}h.res_dil_{num}_{atlasName}"
    )
    if not path.isfile(res_dil_name):
        atlasRibbon = nib.load(path.join(fsDir, f"sub-{sub}", "mri", atlasName))

        hemiAtlasRibbonDat = atlasRibbon.get_fdata() * hemiRibbon.get_fdata().astype(
            bool
        )

        dilHemiAtlasRibbonDat = grey_dilation(hemiAtlasRibbonDat, size=(2, 2, 2))
        dilHemiAtlasRibbon = nib.Nifti1Image(
            dilHemiAtlasRibbonDat, header=atlasRibbon.header, affine=atlasRibbon.affine
        )
        nib.save(
            dilHemiAtlasRibbon,
            path.join(fsDir, f"sub-{sub}", "mri", f"{hemi}h.dil_{atlasName}"),
        )

        # resample the mask to bold space
        resDilRibbon = resample_from_to(dilHemiAtlasRibbon, boldref, order=0)
        nib.save(resDilRibbon, res_dil_name)

        return resDilRibbon.shape


if __name__ == "__main__":
    sub = "01"
    ses = ["01"]
    baseP = "/z/fmri/data/coimbra23"  #'/local/dlinhardt/temp/helen'
    bidsDir = path.join(baseP, "BIDS")
    layout = bids.BIDSLayout(bidsDir)
    subInDir = path.join(baseP, "derivatives", "fmriprep", "analysis-01", f"sub-{sub}")
    outP = path.join(baseP, "derivatives", "prfprepare", "analysis-01", f"sub-{sub}")
    fsDir = path.join(
        baseP, "derivatives", "fmriprep", "analysis-01", "sourcedata", "freesurfer"
    )
    forceParams = ""  # ['mini_params','prf']
    fmriprepLegacyLayout = False
    average = True
    output_only_average = False
    atlases = [
        "benson",
        "wang",
        "lh.LOTS.annot",
        "lh.litVWFA.annot",
        "lh.motspots.annot",
        "rh.LOTS.annot",
        "rh.motspots.annot",
    ]
    roisIn = ["V1", "V2", "V3"]
    analysisSpace = "volume"
    force = False
    verbose = True
    nii_to_surfNii(
        sub,
        ses,
        layout,
        bidsDir,
        subInDir,
        outP,
        fsDir,
        forceParams,
        fmriprepLegacyLayout,
        average,
        output_only_average,
        atlases,
        roisIn,
        analysisSpace,
        force,
        verbose,
    )

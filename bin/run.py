# %%
from link_stimuli import link_stimuli
from nii_to_surfNii import nii_to_surfNii
from stim_as_nii import stim_as_nii
import json
import os
import sys
from os import path
from neuropythy.commands import atlas
import bids
import collections
from glob import glob

# for the annots part
# import nibabel as nib
import subprocess as sp
from zipfile import ZipFile

# get all needed functions
flywheelBase = "/flywheel/v0"
sys.path.insert(0, flywheelBase)

configFile = path.join(flywheelBase, "config.json")
bidsDir = path.join(flywheelBase, "BIDS")


# updates nested dicts
def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


# turns a list within a string to a list of strings
def listFromStr(s):
    return s.split("]")[0].split("[")[-1].split(",")


#  turns the config entry to a loopable list
def config2list(c, b=None):
    if b is not None:
        if "all" in c:
            l = sorted(b)
        else:
            if isinstance(c, list):
                l = c
            else:
                l = listFromStr(c)
    else:
        if isinstance(c, str):
            try:
                l = [float(a) for a in listFromStr(c)]
            except:
                l = listFromStr(c)
        elif isinstance(c, list):
            try:
                l = [float(a) for a in c]
            except:
                l = c
        elif isinstance(c, float) or isinstance(c, bool) or isinstance(c, int):
            l = [c]

    return l


################################################
# define de default config
defaultConfig = {
    "subjects": "all",
    "sessions": "all",
    "etcorrection": False,
    "tasks": "all",
    "force": False,
    "custom_output_name": "",
    "fmriprep_legacy_layout": False,
    "forceParams": "",
    "use_numImages": False,
    "config": {
        "analysisSpace": "fsnative",
        "average_runs": False,
        "output_only_average": False,
        "rois": "all",
        "atlases": "all",
        "fmriprep_bids_layout": False,
        "fmriprep_analysis": "01",
    },
    "verbose": True,
}


def die(*args):
    print(*args)
    sys.exit(1)


################################################
# load in the config json and update the config dict
try:
    with open(configFile, "r") as fl:
        jsonConfig = json.load(fl)
    config = update(defaultConfig, jsonConfig)
except Exception:
    die("Could not read config.json!")

verbose = config["verbose"]
force = config["force"]


def note(*args):
    if verbose:
        print(*args)
    return None


note("Following configuration is used:")
note(json.dumps(config, indent=4))


################################################
# everything not subject-specific
# define input direcotry

# get the BIDS layout
layout = bids.BIDSLayout(bidsDir, validate=False, derivatives=True)
fmriprep_bids_layout = config["config"]["fmriprep_bids_layout"]
fmriprepAnalysis = config["config"]["fmriprep_analysis"]
if not fmriprep_bids_layout:
    inDir = path.join(flywheelBase, "input", f"analysis-{fmriprepAnalysis}")
else:
    # check if it is bids valid
    specific_laypout = layout.derivatives[f'derivatives/fmriprep-{fmriprepAnalysis}']
    name=specific_laypout.root.split('/')[-1]
    inDir = path.join(flywheelBase, "output", f"{name}")

note(f"Loading data from {inDir}")

# check the BIDS directory
if not path.isdir(bidsDir):
    die("no BIDS directory found!")

# define and check subject and freesurfer dir
if config["fmriprep_legacy_layout"] is True:
    fsDir = path.join(inDir, "freesurfer")
else:
    fsDir = path.join(inDir, "sourcedata", "freesurfer")

if path.isdir(fsDir):
    note(f"Freesurfer dir found at {fsDir}!")
else:
    die(f"No freesurfer dir found at {fsDir}!")

# ROIs and atlases from config
areas = config2list(config["config"]["rois"])
atlases = config2list(config["config"]["atlases"])
if atlases[0] == "all":
    atlases = ["benson", "wang", "fs_custom"]

if areas[0] == "fullBrain":
    atlases = ["fullBrain"]

# check if there is the custom.zip, if yes unzip
if "fs_custom" in atlases:
    fsAvgLabelCustom = path.join(fsDir, "fsaverage", "customLabel")
    fsAvgLabelCustomZip = path.join(fsAvgLabelCustom, "custom.zip")

    if not path.isfile(fsAvgLabelCustomZip):
        print(f"We could not find a custom.zip in {fsAvgLabelCustom}!")
        print("Removing fs_custom atlas from list.")
        atlases.remove("fs_custom")
        customAnnots = []
    else:
        note(f"Found custom.zip in {fsAvgLabelCustom}!")
        if not path.isfile(path.join(fsAvgLabelCustom, "DONE")):
            # Unzip the annotations
            with ZipFile(fsAvgLabelCustomZip, "r") as zipObj:
                zipObj.extractall(fsAvgLabelCustom)

            # create a check file
            with open(fna := path.join(fsAvgLabelCustom, "DONE"), "a"):
                os.utime(fna, None)

        # Read all the annotations
        customAnnots = glob(path.join(fsAvgLabelCustom, "*.annot"))
        atlases.remove("fs_custom")
        atlases += [path.basename(a) for a in customAnnots]
else:
    customAnnots = []

# get additional prams from config.json

tasks = config2list(config["tasks"])
if tasks == [""]:
    tasks = False

customName = config2list(config["custom_output_name"])[0]
if customName == "":
    customName = False

etcorr = config2list(config["etcorrection"])[0]

use_numImages = config2list(config["use_numImages"])[0]
note(f"[run.py] use_numImages is: {use_numImages}")

fmriprepLegacyLayout = config2list(config["fmriprep_legacy_layout"])[0]

forceParams = config2list(config["forceParams"])
if forceParams == [""]:
    forceParams = False

average = config2list(config["config"]["average_runs"])[0]

output_only_average = config2list(config["config"]["output_only_average"])[0]

analysisSpace = config2list(config["config"]["analysisSpace"])[0]

###############################################################################
# define the output directory automatically
# start a new one when the config part in the .json is different
analysis_number = 0
found_outbids_dir = False

while not found_outbids_dir and analysis_number < 100:
    analysis_number += 1

    outDir = path.join(
        flywheelBase, "output", "prfprepare", f"analysis-{analysis_number:02d}"
    )
    optsFile = path.join(outDir, "options.json")

    # if the analyis-XX directory exists check for the config file
    if path.isdir(outDir) and path.isfile(optsFile):
        with open(optsFile, "r") as fl:
            opts = json.load(fl)

        # check for the options file equal to the config
        if sorted(opts.items()) == sorted(config["config"].items()):
            found_outbids_dir = True

    # when we could not find a fitting analysis-XX forlder we make a new one
    else:
        if not path.isdir(outDir):
            os.makedirs(outDir, exist_ok=True)

        # dump the options file in the output directory
        with open(optsFile, "w") as fl:
            json.dump(config["config"], fl, indent=4)
        found_outbids_dir = True

note(f"Output directory: {outDir}")


# subject from config and check
BIDSsubs = layout.get_subjects()
subs = config2list(config["subjects"], BIDSsubs)


################################################
# loop over subjects
for sub in subs:
    if sub not in BIDSsubs:
        die(f"We did not find given subject {sub} in BIDS dir!")

    # define and check subject dir
    if config["fmriprep_legacy_layout"] is True:
        subInDir = path.join(inDir, "fmriprep", f"sub-{sub}")
    else:
        subInDir = path.join(inDir, f"sub-{sub}")

    if path.isdir(subInDir):
        note(f"Subject in-dir found at {subInDir}!")
    else:
        die(f"No Subject in-dir found at {subInDir}!")

    # session if given otherwise it will loop through sessions from BIDS
    BIDSsess = layout.get_sessions(subject=sub)
    sess = config2list(config["sessions"], BIDSsess)

    # define the subject output dir
    subOutDir = path.join(outDir, f"sub-{sub}")

    ###############################################################################
    # run neuropythy if not existing yet
    if not path.isfile(path.join(fsDir, f"sub-{sub}", "mri", "benson14_varea.mgz")):
        try:
            print("Letting Neuropythy work...")
            os.chdir(fsDir)
            atlas.main(f"sub-{sub}", "-v", "-S")
            os.chdir(path.expanduser("~"))
        except BaseException as error:
            print("An exception occurred: {}".format(error))
            die("Neuropythy failed!")

    ###############################################################################
    # Convert the annots to indivisual subject space using surf2surf
    if customAnnots:
        os.environ["SUBJECTS_DIR"] = fsDir

        sublbl = path.join(fsDir, f"sub-{sub}", "customLabel")
        if not path.isdir(sublbl):
            os.mkdir(sublbl)
        for annot in customAnnots:
            if not path.isfile(path.join(sublbl, path.basename(annot))):
                he = path.basename(annot).split(".")[0]
                cmd = (
                    f"mri_surf2surf --srcsubject fsaverage --trgsubject sub-{sub} --hemi {he} "
                    f"--sval-annot {annot} --tval {path.join(sublbl, path.basename(annot))}"
                )
                sp.call(cmd, shell=True)

    ###############################################################################
    # do the actual work
    os.chdir(flywheelBase)

    print("Converting Stimuli to .nii.gz...")
    etcorr = stim_as_nii(
        sub,
        sess,
        bidsDir,
        subOutDir,
        etcorr,
        forceParams,
        use_numImages,
        force,
        verbose,
    )

    print("Masking data with visual areas and save them to 2D nifti...")
    nii_to_surfNii(
        sub,
        sess,
        layout,
        bidsDir,
        subInDir,
        subOutDir,
        fsDir,
        forceParams,
        fmriprepLegacyLayout,
        average,
        output_only_average,
        atlases,
        areas,
        analysisSpace,
        force,
        verbose,
    )

    # run this again with the eyetracker correction if applicable
    if etcorr:
        nii_to_surfNii(
            sub,
            sess,
            layout,
            bidsDir,
            subInDir,
            subOutDir.replace(
                f"analysis-{analysis_number:02d}", f"analysis-{analysis_number:02d}_ET"
            ),
            fsDir,
            forceParams,
            fmriprepLegacyLayout,
            average,
            output_only_average,
            atlases,
            areas,
            analysisSpace,
            force,
            verbose,
        )
    # we could add some option for smoothing here?

    print("Creating events.tsv for the data containing the correct stimulus...")
    link_stimuli(
        sub,
        sess,
        tasks,
        layout,
        bidsDir,
        subOutDir,
        etcorr,
        forceParams,
        average,
        output_only_average,
        force,
        verbose,
    )

# copy the dataset_discription from fmriprep
sp.call(f'cp {path.join(inDir,"dataset_description.json")} {outDir}', shell=True)

# if defined write link for custom output folder name
if customName:
    try:
        os.chdir(path.join(flywheelBase, "output", "prfprepare"))
        if not path.islink(f"analysis-{customName}"):
            os.symlink(f"analysis-{analysis_number:02d}", f"analysis-{customName}")
    except:
        print(f"Could not create the custom_output_name analysis-{customName}")

os.chdir(path.expanduser("~"))
# exit happily
sys.exit(0)


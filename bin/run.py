#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:49:37 2022

@author: dlinhardt
"""

import json, os, sys
from neuropythy.commands import atlas
import bids
import glob as glob

# for the annots part
import nibabel as nib
import subprocess as sp
from zipFile import ZipFile
import numpy as np

# get all needed functions
sys.path.insert(0, '../scripts/')
from stim_as_nii    import stim_as_nii
from nii_to_surfNii import nii_to_surfNii
from link_stimuli   import link_stimuli

flywheelBase  = '/flywheel/v0'

configFile = os.path.join(flywheelBase, 'config.json')
bidsDir = os.path.join(flywheelBase, 'BIDS')

def die(*args):
    print(*args)
    sys.exit(1)
def note(*args):
    if verbose: print(*args)
    return None

verbose = os.environ.get('VERBOSE', '0').strip() == '1'
force   = os.environ.get('FORCE', '0').strip() == '1'

try:
    with open(configFile, 'r') as fl:
        conf = json.load(fl)
except Exception:
    die("Could not read config.json!")
    
# subject from config and check
sub   = conf['subject']

layout = bids.BIDSLayout(bidsDir)
BIDSsubs = layout.get(return_type='id', target='subject')

if not sub in BIDSsubs:
    die(f'We did not find given subject {sub} in BIDS dir!')
note(f'Working on Subject: {sub}')

# session if given otherwise it will loop through sessions from BIDS
if hasattr('conf', 'session'):
    if conf['sessionName'][0] != 'all':
        sess = conf['sessionName']
    else:
        sess = layout.get(subject=sub, return_type='id', target='session')  
else:
    sess = layout.get(subject=sub, return_type='id', target='session')  

# ROIs and atlases from config
areas   = conf['rois'].split(']')[0].split('[')[-1].split(',')
atlases = conf['atlases'].split(']')[0].split('[')[-1].split(',')
if atlases[0] == 'all':
    atlases = ['benson', 'wang']

# Annotations
fs_annot     = conf["fs_annot"]
fs_annot_ids = conf['fs_annot_ids'].split(']')[0].split('[')[-1].split(',')

if fs_annot=="custom.zip":
    # If it is custom, it meand that there will be a file called custom.zip in fsaverage/label folder that we will need to convert to  individual subject space first. This means like Neuropythy, this step will update the subject's label folder with new files. 
    # Do it below after the dirs have been defined
    convert_custom_annot = True



# get additional prams from config.json
etcorr = conf['etcorrection']
fmriprepLegacyLayout = conf['fmriprep_legacy_layout']
if hasattr('conf', 'forceParams'):
    forceParams = (conf['forceParams'].split(']')[0].split('[')[-1].split(','))
else:
    forceParams = False

# define input direcotry
inDir = os.path.join(flywheelBase, 'input', f'analysis-{conf["config"]["fmriprep_analysis"]}')

# check the BIDS directory
if not os.path.isdir(bidsDir):
    die('no BIDS directory found!')

# define and check subject and freesurfer dir
if conf['fmriprep_legacy_layout'] == True:
    subInDir = os.path.join(inDir, 'fmriprep', f'sub-{sub}')
    fsDir    = os.path.join(inDir, 'freesurfer')
else:
    subInDir = os.path.join(inDir, f'sub-{sub}')
    fsDir    = os.path.join(inDir, 'sourcedata', 'freesurfer')

if os.path.isdir(fsDir):
    note(f'Freesurfer dir found at {fsDir}!')
else:
    die(f'No freesurfer dir found at {fsDir}!')
    

###############################################################################    
# define the output directory automatically
# start a new one when the config part in the .json is different
analysis_number = 0
found_outbids_dir = False

while not found_outbids_dir and analysis_number<100:
    analysis_number += 1
    outDir=os.path.join(flywheelBase, 'output', 'prfprepare', f'analysis-{analysis_number:02d}')
    optsFile = os.path.join(outDir, 'options.json')  
    # if the analyis-XX directory exists check for the config file
    if os.path.isdir(outDir) and os.path.isfile(optsFile): 
        with open(optsFile, 'r') as fl: 
            opts = json.load(fl)
        # check for the options file equal to the config            
        if sorted(opts.items()) == sorted(conf['config'].items()): 
            found_outbids_dir = True
    # when we could not find a fitting analysis-XX forlder we make a new one
    else:
        if not os.path.isdir(outDir):
            os.makedirs(outDir, exist_ok=True)
            # dump the options file in the output directory
        with open(optsFile, 'w') as fl:
            json.dump(conf['config'], fl)    
        found_outbids_dir = True
note(f'Output directory: {outDir}')

# define the subject output dir
subOutDir = os.path.join(outDir, f'sub-{sub}')


###############################################################################
# run neuropythy if not existing yet
if not os.path.isfile(os.path.join(fsDir, f'sub-{sub}', 'surf', 'lh.benson14_varea.mgz')):
    try:
        print('Letting Neuropythy work...')
        os.chdir(fsDir)
        atlas.main(f'sub-{sub}')
        os.chdir(os.path.expanduser("~"))
    except:
        print('Neuropythy failed!')
        die()

#######################
# Obtain the vertex numbers by combining the required Atlases and ROIs
# Convert the custom annot from fsaverage to individual space as well
if convert_custom_annot:
   # Unzip the annotations
   fsavglbl = os.path.join(fsDir,"fsaverage","label")
   fsavglblcustom = os.path.join(fsavglbl, "custom")
   if not os.path.isdir(fsavglblcustom): os.mkdir(fsavglblcustom)  
   with ZipFile(os.path.join(fsavglbl,"custom.zip"),'r') as zipObj:
       zipObj.extractall(fsavglblcustom)

   # Read all the annotations
   os.chdir(fsavglblcustom)
   annots = glob.glob("*.annot")
  
   # Create a dictionary to store all the annots, labels and its vertices
   customAnnotDict = dict()

   # Convert the annots to indivisual subject space using surf2surf
   os.environ["SUBJECTS_DIR"]=fsDir 
   # for ses in sess: # I think we need to fix the sess thing... we will talk about it :)
   # sublbl = os.path.join(fsDir,f'sub-{sub}',f'ses-{ses}','label')
   sublbl = os.path.join(fsDir,f'sub-{sub}','label')
   for annot in annots:
       hemi = annot.split(".")[0]
       cmd = (f'mri_surf2surf --srcsubject fsaverage --trgsubject sub-{sub} --hemi {hemi} '  
              f'--sval-annot {os.path.join(fsavglblcustom,annot)} --tval {sublbl}/{annot}')
       sp.call(cmd, shell=True)   
       # Now we need to extact every label from evey annot so that we can read the vertices to create the big ROI for the analysis
       # I think we do not need to extract the labels, it is enough reading the vertices (I had to extract them for DWI, not here)
       # I leave the extraction code here just in case, delete later. 
            # Extract the individual labels
            # --labelbase ${ANNOTname}
            # mri_annotation2label --subject ${SUBJECT_ID} --hemi ${hemi}  \
            #                      --annotation ${labeldir}/${ANNOTname}.annot --outdir $labeldirtmp
       # Read the new annot:
       customAnnotDict[annot] = nib.freesurfer.read_annot(f'{sublbl}/{annot}')
            


# David, here you have the code to read the labels and vertices:
# annots == list(customAnnotDict.keys())
for annot in annots:
    subannot = customAnnotDict[annot]
    for i,lab in enumerate(subannot[2]):
        label          = lab.decode('UTF-8').strip("'")
        vertices0based = np.argwhere(subannot[0]==i)

# Here you have all teh annot > Label > vertices
# I assume that you will sum up all the vertices so that we only analyze one vertex once right? 



###############################################################################
# do the actual work
os.chdir(flywheelBase)

print('Converting Stimuli to .nii.gz...')
etcorr = stim_as_nii(sub, sess, bidsDir, subOutDir, etcorr, forceParams, force, verbose)

print('Masking data with visual areas and save them to 2D nifti...')
nii_to_surfNii(sub, sess, layout, bidsDir, subInDir, subOutDir, fsDir, forceParams,
               fmriprepLegacyLayout, atlases, areas, etcorr, force, verbose)
# we could add some option for smoothing here?

print('Creating events.tsv for the data containing the correct stimulus...')
link_stimuli(sub, sess, layout, bidsDir, subOutDir, etcorr, force, verbose)

os.chdir(os.path.expanduser("~"))


# exit happily
sys.exit(0)

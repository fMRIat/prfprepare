#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:49:37 2022

@author: dlinhardt
"""

import json, os, sys
from neuropythy.commands import atlas
import bids

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
    
    outDir  = os.path.join(flywheelBase, 'output', 'prfprepare', f'analysis-{analysis_number:02d}')
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
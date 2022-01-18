#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 10:51:30 2021

@author: dlinhardt
"""
import argparse
import numpy as np
from os import path, makedirs
import bids
import nibabel as nib
from scipy.io import loadmat
from scipy.ndimage import shift
from glob import glob

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

# parser
parser = argparse.ArgumentParser(description='parser for script converting mrVista stimulus files to nii')

parser.add_argument('sub',         type=str, help='subject name')
parser.add_argument('bids_in_dir', type=str, help='input directory before fmriprep for BIDS layout')
parser.add_argument('--etcorr',    type=str, help='perform an eyetracker correction [default: False]', default='False')
parser.add_argument('--force',     type=str, help='force a new run [default: False]', default='False')

args = parser.parse_args()

etcorr = str2bool(args.etcorr)
force  = str2bool(args.force)

# base paths
inP  = '/flywheel/v0/input'
outP = '/flywheel/v0/output/BIDS'

# create the output folders
makedirs(outP, exist_ok=True)
        
# get the bids layout fur given subject
layout = bids.BIDSLayout(args.bids_in_dir)

subs = layout.get(return_type='id', target='subject')

if args.sub in subs:
    sub = args.sub
else:
    exit(3)
    

#%% convert stimulus from matlab to nii.gz if not done yet
stims = np.array(glob(path.join(args.bids_in_dir, 'stimuli', '*.mat')))
stimsBase = np.array([stims[i].split('task-')[-1].split('_presentation.mat')[0] for i in range(len(stims))])

for stimI,stim in enumerate(stimsBase):
    makedirs(path.join(outP, 'stimuli'), exist_ok=True)
    oFname = path.join(outP, 'stimuli', f'task-{stim}_apertures.nii.gz')
    
    if not path.exists(oFname) or force:
        # load stimulus from mrVista stimulation file
        origStimF = stims[stimI]
                    
        origStim = loadmat(origStimF, simplify_cells=True)
        stimSeq  = origStim['stimulus']['seq']
        oImages  = origStim['stimulus']['images']
                    
        stimImagesU, stimImagesUC = np.unique(oImages, return_counts=True)
        oImages[oImages!=stimImagesU[np.argmax(stimImagesUC)]] = 1
        oImages[oImages==stimImagesU[np.argmax(stimImagesUC)]] = 0
        oStimVid  = oImages[:,:,stimSeq[::int(1/origStim['stimulus']['seqtiming'][1]*origStim['params']['tr'])]-1] # 8Hz flickering * 2s TR
                    
        img = nib.Nifti1Image(oStimVid[:,:,None,:].astype('float64'), np.eye(4))
        img.header['pixdim'][1:5] = [1,1,1,2]
        img.header['qoffset_x']=img.header['qoffset_y']=img.header['qoffset_z'] = 1
        img.header['cal_max'] = 1
        img.header['xyzt_units'] = 10
        nib.save(img, oFname)



#%% do the shifting for ET corr
if etcorr:    
    
    sess = layout.get(subject=sub, return_type='id', target='session')
    
    for sesI,ses in enumerate(sess):
    
        tasks = layout.get(subject=sub, session=ses, return_type='id', target='task')
        
        for taskI,task in enumerate(tasks):
            
            origStimF = stims[[task[:3] in ap for ap in stimsBase]].item()
            
            runs = layout.get(subject=sub, session=ses, task=task, return_type='id', target='run')
            
            for runI,run in enumerate(runs):
            
                oFname = path.join(outP, 'stimuli', f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_apertures.nii.gz')
                gazeFile = path.join(args.bids_in_dir, f'sub-{sub}', f'ses-{ses}', 'etdata', 
                                     f'sub-{sub}_ses-{ses}_task-{task}_run-{run:02d}_gaze.mat')
                
                if not path.exists(gazeFile):
                    print(f'No gaze file is found at {gazeFile}!')
                    print( 'Therefore, we skip eyetracker correction...')
                    exit(7)
                
                if not path.exists(oFname) or force:
                        
                    # load stimulus from mrVista stimulation file
                    origStim = loadmat(origStimF, simplify_cells=True)
                    stimSeq  = origStim['stimulus']['seq']
                    oImages  = origStim['stimulus']['images']
                    
                    stimImagesU, stimImagesUC = np.unique(oImages, return_counts=True)
                    oImages[oImages!=stimImagesU[np.argmax(stimImagesUC)]] = 1
                    oImages[oImages==stimImagesU[np.argmax(stimImagesUC)]] = 0
                    oStimVid  = oImages[:,:,stimSeq[::int(1/origStim['stimulus']['seqtiming'][1]*origStim['params']['tr'])]-1] # 8Hz flickering * 2s TR
                    
                    # load the jitter file
                    gaze = loadmat(gazeFile, simplify_cells=True)
                    
                    # get rid of out of image data (loss of tracking)
                    gaze['x'][np.any((gaze['x']==0, gaze['x']==1280),0)] = 1280/2 # 1280 comes from resolution of screen
                    gaze['y'][np.any((gaze['y']==0, gaze['y']==1024),0)] = 1024/2 # 1024 comes from resolution of screen
                    
                    # resamplet to TR
                    x = np.array([np.mean(f) for f in np.array_split(gaze['x'], oStimVid.shape[2])])
                    y = np.array([np.mean(f) for f in np.array_split(gaze['y'], oStimVid.shape[2])])
                    
                    
                    # demean the ET data
                    x -= x.mean()
                    y -= y.mean()
                    y = -y # there is a flip between ET and fixation dot sequece (pixel coordinates), 
                             # with this the ET data is in the same space as fixation dot seq.
                    
                    
                    # TODO: we problably shoud make a border around the actual stim and then 
                    #       place the original stim in the center before shifting it so that
                    #       more peripheral regions could also be stimulated.
                    #       for the analysis the new width (zB 8° radius)
                    # border = 33 for +1° radius?
                    # shiftStim = np.zeros((oStimVid.shape[0]+2*border,oStimVid.shape[1]+2*border,oStimVid.shape[2]))
                    
                    # shift the stimulus opposite of gaze direction
                    for i in range(len(x)):
                        oStimVid[...,i] = shift(oStimVid[...,i], (-y[i], -x[i]), mode='constant', cval=0)
                        
                        
                    img = nib.Nifti1Image(oStimVid[:,:,None,:].astype('float64'), np.eye(4))
                    img.header['pixdim'][1:5] = [1,1,1,2]
                    img.header['qoffset_x']=img.header['qoffset_y']=img.header['qoffset_z'] = 1
                    img.header['cal_max'] = 1
                    img.header['xyzt_units'] = 10
                    nib.save(img, oFname)

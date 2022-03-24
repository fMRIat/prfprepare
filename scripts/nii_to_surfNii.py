#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 13:28:36 2021

@author: dlinhardt
"""

# from nilearn import image
# from nilearn.surface import vol_to_surf
import nibabel as nib
from os import path, makedirs
import bids
import numpy as np
# from shutil import copy2 as copy
import neuropythy as ny
import argparse
from glob import glob

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

# parser
parser = argparse.ArgumentParser(description='parser for script converting mrVista stimulus files to nii')

parser.add_argument('sub',         type=str, help='subject name')
parser.add_argument('bids_in_dir', type=str, help='input directory before fmriprep for BIDS layout')
parser.add_argument('output_dir',  type=str, help='output subject directory')
parser.add_argument('subjectPath', type=str, help='input subject directory')
parser.add_argument('freesurferPath',    type=str, help='input freesurfer directory')
parser.add_argument('--fmriprep_legacy', type=str, help='if fMRIPrep output layout is legacy', default='False')
parser.add_argument('--etcorr',    type=str, help='perform an eyetracker correction [default: False]', default='False')
parser.add_argument('--atlas',     type=str, help='which atlas to use for the region comparison [default: benson]', default='benson')
parser.add_argument('--areas',     type=str, help='which atlas to use for the region comparison [default: benson]', default='[V1]')
parser.add_argument('--force',     type=str, help='force a new run [default: False]', default='False')

args = parser.parse_args()

etcorr = str2bool(args.etcorr)
force  = str2bool(args.force)
fmriprep_legacy  = str2bool(args.fmriprep_legacy)

# base paths
outP = args.output_dir

# get the bids layout fur given subject
layout = bids.BIDSLayout(args.bids_in_dir)

subs = layout.get(return_type='id', target='subject')

if args.sub in subs:
    sub = args.sub
else:
    exit(3)
    
# extract the ROIs
rois = args.areas.split(']')[0].split('[')[-1].split(',')

###############################################################################
# load in the atlas
if args.atlas=='benson':
    lh_areasP = path.join(args.freesurferPath, f'sub-{sub}', 'surf', 'lh.benson14_varea.mgz')
    rh_areasP = path.join(args.freesurferPath, f'sub-{sub}', 'surf', 'rh.benson14_varea.mgz')
    if not path.exists(lh_areasP) or not path.exists(rh_areasP): exit(4)
    
    # load the label files
    lh_areas = nib.load(lh_areasP).get_fdata()[0,0,:]
    rh_areas = nib.load(rh_areasP).get_fdata()[0,0,:]
    
    # load the label area dependency
    mdl = ny.vision.retinotopy_model('benson17', 'lh')
    areaLabels = dict(mdl.area_id_to_name)
    areaLabels = { areaLabels[k]:k for k in areaLabels }
    
elif args.atlas=='wang':
    lh_areasP = path.join(args.freesurferPath, f'sub-{sub}', 'surf', 'lh.wang15_mplbl.mgz')
    rh_areasP = path.join(args.freesurferPath, f'sub-{sub}', 'surf', 'rh.wang15_mplbl.mgz')
    if not path.exists(lh_areasP) or not path.exists(rh_areasP): exit(4)
    
    # load the label files
    lh_areas = nib.load(lh_areasP).get_fdata()[0,0,:]
    rh_areas = nib.load(rh_areasP).get_fdata()[0,0,:]
    
    labelNames = ["Unknown",
         "V1v", "V1d", "V2v", "V2d", "V3v", "V3d",
         "hV4", "VO1", "VO2", "PHC1", "PHC2",
         "V3A", "V3B", "LO1", "LO2", "TO1", "TO2",
         "IPS0", "IPS1", "IPS2", "IPS3", "IPS4", "IPS5",
         "SPL1", "hFEF"]
    
    areaLabels = {labelNames[k]:k for k in range(len(labelNames))}


# go for all given ROIs
for roi in rois:
    # get labels associated with ROI
    roiLabels = [value for key, value in areaLabels.items() if roi in key]
                
    sess = layout.get(subject=sub, return_type='id', target='session')  
    
    for sesI,ses in enumerate(sess):
        funcInP  = path.join(args.subjectPath, f'ses-{ses}', 'func')
        funcOutP = path.join(outP, f'ses-{ses}', 'func')
        makedirs(funcOutP, exist_ok=True)
               
        lh_mask = np.array([ l in roiLabels for l in lh_areas ])
        rh_mask = np.array([ l in roiLabels for l in rh_areas ])
        
        tasks = layout.get(subject=sub, session=ses, return_type='id', target='task')
        
        for task in tasks:
            
            runs = layout.get(subject=sub, session=ses, task=task, return_type='id', target='run')
            
            for run in runs:
                
                newNiiP = path.join(funcOutP, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-{roi}_bold.nii.gz')

                if not path.exists(newNiiP) or force:
                    
                    # load the .gii in fsnative
                    if fmriprep_legacy:
                        # print('Reading fmriprep legacy files, space then hemi')
                        giiPL = path.join(funcInP, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-fsnative_hemi-L_bold.func.gii')
                        giiPR = path.join(funcInP, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_space-fsnative_hemi-R_bold.func.gii')
                    else:
                        # print('Reading fmriprep NON-legacy files, hemi then space')
                        giiPL = path.join(funcInP, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_hemi-L_space-fsnative_bold.func.gii')
                        giiPR = path.join(funcInP, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_hemi-R_space-fsnative_bold.func.gii')
                    giiImgL = nib.load(giiPL).agg_data()
                    giiImgR = nib.load(giiPR).agg_data()
                    
                    # apply V1 mask and stack left and right
                    vertices = np.vstack((giiImgL[lh_mask,:], giiImgR[rh_mask,:]))
                    
                    # get rid of vertices without data
                    vertices = vertices[vertices.var(1)>1e-6,:]
                    
                    # create and save new nii img
                    apertures = np.array(glob(path.join(outP, 'stimuli', 'task-*.nii.gz')))
                    stimNii = nib.load(apertures[[task[:3] in ap for ap in apertures]].item())
                    
                    newNii = nib.Nifti2Image(vertices[:,None,None,:].astype('float64'), affine=np.eye(4))
                    newNii.header['pixdim']     = stimNii.header['pixdim']
                    newNii.header['qoffset_x']  = 1
                    newNii.header['qoffset_y']  = 1
                    newNii.header['qoffset_z']  = 1
                    newNii.header['cal_max']    = 1
                    newNii.header['xyzt_units'] = 10
                    nib.save(newNii, newNiiP)
                    
                    if etcorr:
                        funcOutPET = path.join(outP.replace('/sub-', '_ET/sub-'), f'ses-{ses}', 'func')
                        makedirs(funcOutPET, exist_ok=True)
                        newNiiPET = path.join(funcOutPET, f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-{roi}_bold.nii.gz')
                        nib.save(newNii, newNiiPET)
                


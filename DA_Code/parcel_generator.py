#my_functions.py
import os
import glob
import numpy as np
import nibabel as nib
from scipy import ndimage
import sys
import time


def arg_maker(xcluster,xnodes,xresolution,xhemi):

    xargs = {'cluster': xcluster,
            'nodes': xnodes,
            'resolution':int(xresolution),
            'hemisphere': xhemi,
            'func':'denoised_func_Subject',
            'sphere_r':5,
            'feat_sels':0.7,
            'sl_sphere_r':21,
            'sl_feat_sel':0.7,
            'sl_space':3,
            'mask_expend':False,
            'mask_expend_iter':1,
            'train_TR':2.5,
            'hyp_ref_id':0}

    xargs['mask'] = 'harvardoxford_cortical_%dmm_mask' % xargs['resolution']
    xargs['node_mask'] = 'niftiNORMS_%dmm_mask' % xargs['resolution']

    xargs['node_type']='parcel'

    #xsub_num = int(xsub_num)
    return xargs


def creat_parcel(xargs, dirs, xlog, logs):
    
    start = time.time()


    xparcel_file = dirs['in']
    xparcel = nib.load(xparcel_file)
    xhdr = xparcel.header

    xparcel_mx = xparcel.get_fdata()
    n_pars = int(np.max(xparcel_mx))
    # xcent = xparcel.shape[0]/2
    if xargs['resolution']==2:
        xcent = 45
    elif xargs['resolution']==3:
        xcent = 30
    elif xargs['resolution']==1:
        xcent = 91

    print('* create roi from %d parcels defined in %s' % (n_pars, xparcel_file))
    print('... center x coordinates: %s' % xcent)

    nl = 0
    nr = 0
    roi_selection = {}
    it_rois_nl = []
    it_rois_nr = []
    n_vox_in_sphere_nl = []
    n_vox_in_sphere_nr = []
    n_novox_rois_nl = []
    n_novox_rois_nr = []

    for xv in range(n_pars):
        print('... parcel: %d' % (xv+1))
        xdata = np.zeros(xparcel.shape)
        xdata[np.where(xparcel_mx==(xv+1))]=1
        xpar_img = nib.Nifti1Image(xdata, affine=xparcel.affine)
        xpar_img.set_data_dtype(xparcel_mx.dtype)
        
        if np.max(np.where(xdata==1)[0]) > xcent:
            xhemi = 'RH'
            nl = nl+1
            xpar = 'node_%s_%0.3d_%dmm_%s' % (xargs['nodes'], nl, xargs['resolution'], xhemi)
        else:
            xhemi = 'LH'
            nr = nr+1
            xpar = 'node_%s_%0.3d_%dmm_%s' % (xargs['nodes'], nr, xargs['resolution'], xhemi)

        xvox = np.array(np.where(xparcel_mx==(xv+1)))
        print('..... %d voxel was found for %s before filtering' % (xvox.shape[1], xpar))

        xpar_file = os.path.join(dirs['deriv'], xpar)
        xmask = '%s.nii.gz' % xargs['node_mask']
        xmask_file = os.path.join(dirs['deriv'], xmask)
        nib.save(xpar_img, xpar_file)

        os.system('fslmaths %s -bin %s_bin' % (xpar_file, xpar_file))
        os.system('fslmaths %s_bin -mas %s %s_bin' % (xpar_file, xmask_file, xpar_file))
        os.remove('%s.nii' % xpar_file)

        ## check if the mask is empty
        xtmp_par = nib.load('%s_bin.nii.gz' % xpar_file)
        xtmp_par_mx = xtmp_par.get_fdata()
        xvox_filtered = np.array(np.where(xtmp_par_mx==1))
        xmax = np.max(xtmp_par_mx)

        if xmax==0:
            print('..... no voxel was found for %s' % xpar)
            if xhemi=='RH':
                n_novox_rois_nl.append(nl)
            else:
                n_novox_rois_nr.append(nr)

            os.remove('%s_bin.nii.gz' % xpar_file)
        else:
            logs('..... %d voxel was found for %s after filtering' % (xvox_filtered.shape[1], xpar))
            if xhemi=='RH':
                it_rois_nl.append(nl)
                n_vox_in_sphere_nl.append(xvox_filtered.shape[1])
            else:
                it_rois_nr.append(nr)
                n_vox_in_sphere_nr.append(xvox_filtered.shape[1])

            os.system('gunzip %s_bin.nii.gz' % xpar_file)
    
    roi_selection['n_all_pars'] = n_pars
    roi_selection['n_rois_LH'] = nl
    roi_selection['n_rois_RH'] = nr
    roi_selection['n_sel_rois_LH'] = len(it_rois_nl)
    roi_selection['n_sel_rois_RH'] = len(it_rois_nr)
    roi_selection['it_rois_LH'] = it_rois_nl
    roi_selection['it_rois_RH'] = it_rois_nr
    roi_selection['n_vox_in_sphere_LH'] = n_vox_in_sphere_nl
    roi_selection['n_vox_in_sphere_RH'] = n_vox_in_sphere_nr
    roi_selection['n_novox_rois_LH'] = n_novox_rois_nl
    roi_selection['n_novox_rois_RH'] = n_novox_rois_nr

    outfile = os.path.join(dirs['deriv'], 'rois_%s_%dmm.npy' 
        % (xargs['node_mask'], xargs['resolution']))
    np.save(outfile, roi_selection)

    ##collapse
    for xhemi in ['LH','RH']:
        all_pars = os.path.join(dirs['deriv'], 'all_pars_MNI_%dmm_%s.nii.gz' 
            % (xargs['resolution'], xhemi))
        xfiles = glob.glob(os.path.join(dirs['deriv'], 'node*%s*' % xhemi))

        for xv in range(len(xfiles)):
            xpar_file = xfiles[xv]
            if xv==0:
                os.system('cp %s %s' % (xpar_file, all_pars))
            else:
                os.system('fslmaths %s -add %s %s' % (all_pars, xpar_file, all_pars))

    end = time.time()
    total_time = end - start
    m, s = divmod(total_time, 60)
    print('Time elapsed: %s minutes %s seconds' % (round(m), round(s)))


def redefine_parcel(xargs, dirs, xlog, logs):
    #For Glasser
    print('(+) redefine parcel ids: %s' % xargs['nodes'])
    xtmpdir = os.path.join(dirs['in'], 'tmp')
    if (os.path.isdir(xtmpdir)==False):
        os.mkdir(xtmpdir)

    xparcel_file =dirs['in']
    xparcel = nib.load(xparcel_file)
    xhdr = xparcel.header
    xparcel_mx = xparcel.get_fdata()
    xparcel_mx = np.round(xparcel_mx)
    n_pars = int(np.max(xparcel_mx))
    
    if xargs['resolution']==2:
        xcent = 45
    elif xargs['resolution']==3:
        xcent = 30
    elif xargs['resolution']==1:
        xcent = 91

    npar = 0
    for xv in range(n_pars):
        print('* parcel %0.3d' % (xv+1))
        xdata = np.zeros(xparcel.shape)
        xdata[np.where(xparcel_mx==(xv+1))]=1
        xpar_img = nib.Nifti1Image(xdata, affine=xparcel.affine)
        xpar_img.set_data_dtype(xparcel_mx.dtype)
        
        xvox = np.array(np.where(xparcel_mx==(xv+1)))
        xpar = 'tmp_node_%s_%0.3d_%dmm' % (xargs['nodes'], xv+1, xargs['resolution'])
        xpar_file = os.path.join(xtmpdir, xpar)
        xpar_bin = '%s_bin' % xpar_file

        nib.save(xpar_img, '%s.nii.gz' % xpar_file)
        os.system('fslmaths %s -bin %s' % (xpar_file, xpar_bin))
        os.remove('%s.nii.gz' % xpar_file)

        ## split to LH/RH 
        xparcel_new = nib.load('%s.nii.gz' % xpar_bin)
        xparcel_new_mx = xparcel_new.get_fdata()
        xvox_bin = np.array(np.where(xparcel_new_mx==1))

        n_new_vox = 0
        if (np.min(xvox_bin[0]) < xcent) and (np.max(xvox_bin[0]) > xcent):
            # left hemisphere
            npar = npar+1
            xtmp_new_l = os.path.join(xtmpdir, 
                'tmp_l_node_%s_%0.3d_%dmm' % (xargs['nodes'], npar, xargs['resolution']))
            os.system('fslmaths %s -bin -roi %d -1 -1 -1 -1 -1 -1 -1 %s' 
                % (xpar_bin, xcent, xtmp_new_l))
            
            ## save it as the parcel id
            xhemi_par = nib.load('%s.nii.gz' % xtmp_new_l)
            xhemi_par_mx = xhemi_par.get_fdata()
            xdata = np.zeros(xhemi_par.shape)
            xdata[np.where(xhemi_par_mx==1)]=npar
            xpar_img = nib.Nifti1Image(xdata, affine=xhemi_par.affine)
            xpar_img.set_data_dtype(xhemi_par_mx.dtype)

            xnew_l = os.path.join(xtmpdir, 
                'node_%s_%0.3d_%dmm' % (xargs['nodes'], npar, xargs['resolution']))
            nib.save(xpar_img, '%s.nii.gz' % xnew_l)

            xvox_new = np.array(np.where(xdata==npar))
            n_new_vox += xvox_new.shape[1]
            print('... changing intensity of %d voxels to the LH parcel id %0.3d' 
                % (xvox_new.shape[1], npar))

            # right hemisphere
            npar = npar+1
            xtmp_new_r = os.path.join(xtmpdir, 
                'tmp_r_node_%s_%0.3d_%dmm' % (xargs['nodes'], npar, xargs['resolution']))
            os.system('fslmaths %s -sub %s %s' 
                % (xpar_bin, xnew_l, xtmp_new_r))
            
            ## save it as the parcel id
            xhemi_par = nib.load('%s.nii.gz' % xtmp_new_r)
            xhemi_par_mx = xhemi_par.get_fdata()
            xdata = np.zeros(xhemi_par.shape)
            xdata[np.where(xhemi_par_mx==1)]=npar
            xpar_img = nib.Nifti1Image(xdata, affine=xhemi_par.affine)
            xpar_img.set_data_dtype(xhemi_par_mx.dtype)

            xnew_r = os.path.join(xtmpdir, 
                'node_%s_%0.3d_%dmm' % (xargs['nodes'], npar, xargs['resolution']))
            nib.save(xpar_img, '%s.nii.gz' % xnew_r)
            os.remove('%s.nii.gz' % xtmp_new_l)
            os.remove('%s.nii.gz' % xtmp_new_r)
            os.remove('%s.nii.gz' % xpar_bin)

            xvox_new = np.array(np.where(xdata==npar))
            n_new_vox += xvox_new.shape[1]
            print('... changing intensity of %d voxels to the RH parcel id %0.3d' 
                % (xvox_new.shape[1], npar))
        else:
            if (np.min(xvox_bin[0]) < xcent):
                print('... parcel only in LH')
                xhemi = 'LH'
            elif (np.max(xvox_bin[0]) > xcent):
                print('... parcel only in RH')
                xhemi = 'RH'
            
            ###########################################
            # hemisphere
            npar = npar+1
            xtmp_new_h = os.path.join(xtmpdir, 
                'tmp_h_node_%s_%0.3d_%dmm' % (xargs['nodes'], npar, xargs['resolution']))
            os.system('mv %s.nii.gz %s.nii.gz' % (xpar_bin, xtmp_new_h))
            
            ## save it as the parcel id
            xhemi_par = nib.load('%s.nii.gz' % xtmp_new_h)
            xhemi_par_mx = xhemi_par.get_fdata()
            xdata = np.zeros(xhemi_par.shape)
            xdata[np.where(xhemi_par_mx==1)]=npar
            xpar_img = nib.Nifti1Image(xdata, affine=xhemi_par.affine)
            xpar_img.set_data_dtype(xhemi_par_mx.dtype)

            xnew_h = os.path.join(xtmpdir, 
                'node_%s_%0.3d_%dmm' % (xargs['nodes'], npar, xargs['resolution']))
            nib.save(xpar_img, '%s.nii.gz' % xnew_h)
            os.remove('%s.nii.gz' % xtmp_new_h)

            xvox_new = np.array(np.where(xdata==npar))
            n_new_vox += xvox_new.shape[1]
            print('... changing intensity of %d voxels to the %s parcel id %0.3d' 
                % (xvox_new.shape[1], xhemi, npar))

        print('... changing %d voxels (0.5mm) to %d voxels (%dmm)' 
            % (xvox_bin.shape[1], n_new_vox, xargs['resolution']))

    xpars = glob.glob(os.path.join(xtmpdir, 'node*'))
    xpars_all = os.path.join('Glasser_Parcels_MNI_%s' % xargs['resolution'])

    for xv in range(len(xpars)):
        if xv==0:
            os.system('cp %s %s.nii.gz' % (xpars[xv], xpars_all))
        else:
            os.system('fslmaths %s -add %s %s' % (xpars_all, xpars[xv], xpars_all))

    os.system('gunzip %s.nii.gz' % xpars_all)
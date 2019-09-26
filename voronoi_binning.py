"""

Code for binning big images based on the Voronoi binning method by Cappellari & Copin (2003)


NAME:
      VORONOI_BINNING

AUTHOR:
      Gabriel Brammer, University of Copenhagen
      gabriel.brammer_at_nbi.ku.dk

PURPOSE:
      Perform adaptive spatial binning of Integral-Field Spectroscopic
      (IFS) data to reach a chosen constant signal-to-noise ratio per bin.
      The Voronoi method from Cappellari is here applied on big images,
      so blocking of the images is necessary since it cannot be applied on
      the whole image. The binning is also then applied to the target image
      obtained with different filters.

EXPLANATION:
      Further information on VORONOI_2D_BINNING algorithm can be found in
      Cappellari M., Copin Y., 2003, MNRAS, 342, 345
      http://adsabs.harvard.edu/abs/2003MNRAS.342..345C

CALLING SEQUENCE:
        binned_flux,binned_err = voronoi_binning(image, quiet=True, plot=True)

INPUTS:
       image: The target with the filter where S/N is highest.

KEYWORDS:
        PLOT: Set this keyword to produce a plot of the two-dimensional
              bins of the target image.
       QUIET: Set this keyword to avoid printing progress results.


OUTPUTS:
   binned_flux: Flux of the input image for every Voronoi bin.
   binned_err: Error of the input image for every Voronoi bin.


PROCEDURES USED:
          VORONOI_BINNING   -- Main program to bin the target and other filter images.
          BIN_IMAGE         -- Apply 2D bins to a new image

"""

import numpy as np
import astropy.io.fits as pyfits
from grizli import utils,prep
import matplotlib.pyplot as plt
import glob
from astropy.table import Table

#----------------------------------------------------------------------------


def voronoi_binning(image, quiet=True, plot=True):
    
    """
    Function to bin images using the Voronoi binning method by Cappellari & Copin (2003), and implement
    the binning to the same target observed with different filters.
    
    Input as 'image' the target with the filter where S/N is highest
    
    """
    import numpy as np
    import astropy.io.fits as pyfits
    from grizli import utils
    from grizli import prep
    import matplotlib.pyplot as plt

    im = pyfits.open(image)
    
    sci = np.cast[np.float32](im['SCI'].data)
    sh = sci.shape
    
    ivar = np.cast[np.float32](im['WHT'].data)
    var = 1/ivar
    
    orig_mask = (ivar > 0)
    sci[~orig_mask] = 0
    var[~orig_mask] = 0
    orig_var = var*1
    
    # Simple background
    bkg = np.median(sci[orig_mask])
    sci -= bkg*orig_mask
    
    cps = sci*im[0].header['EXPTIME']
    shot_err = np.sqrt(np.maximum(cps, 4))/im[0].header['EXPTIME']
    var2 = var + shot_err**2
    var = var2
    
    full_bin_seg = np.zeros(sh, dtype=np.int)-1
    
    
    yp, xp = np.indices(sci.shape)
    
    xpf = xp.flatten()
    ypf = yp.flatten()
    
    original_bin = 5 # Factor of 2**original_bin
    
    # Initialize mask
    mask = orig_mask & True
    bin_min = 1
    full_image = sci*0.
    full_err = sci*0.
    
    targetSN = 50
    
    # Don't include original pixels that satisfy targetSN
    if False:
        orig_target = (var > 0) & (sci/np.sqrt(var) > targetSN)
        full_image[orig_target] = sci[orig_target]
        full_err[orig_target] = np.sqrt(var)[orig_target]
        
        # update mask
        mask &= (~orig_target)
    else:
        orig_target = xp < -10
        
    idx = np.arange(sci.size, dtype=np.int) 
    full_image = full_image.flatten()
    full_err = full_err.flatten()
        
    # id, bin, xmin, xmax, ymin, ymax, npix
    full_bin_data = []
    
    SKIP_LAST = False
    
    bin_iter = 0
    bin_factor = original_bin
    
    NO_NEWLINE = '\x1b[1A\x1b[1M'
    
    for bin_iter, bin_factor in enumerate(range(original_bin+1)[::-1]):
        bin = 2**bin_factor
        
        if (bin_factor == 0) & SKIP_LAST:
            continue
            
        ypb = yp[mask] // bin
        xpb = xp[mask] // bin
            
        if bin_factor > 0:
            binned_sci = np.zeros((sh[0]//bin+1, sh[1]//bin+1))
            binned_npix = binned_sci*0
            binned_var = binned_sci*0

            ypi, xpi = np.indices(binned_sci.shape)
            
            # Only consider unmasked bins
            ij = np.unique(xpb + sh[0]//bin*ypb)
            yarr = ij // (sh[0]//bin)
            xarr = ij - (sh[0]//bin)*yarr
            
            for xi, yi in zip(xarr, yarr):
                if not quiet:
                    print(NO_NEWLINE+'{0} {1}/{2} {3}/{4}'.format(bin_factor, 
                                               xi, xarr.max(), 
                                               yi, yarr.max()))
                
                slx = slice(xi*bin, xi*bin+bin)
                sly = slice(yi*bin, yi*bin+bin)
                mslice = mask[sly, slx]
                # Straight average
                binned_sci[yi, xi] = sci[sly, slx][mslice].sum()
                binned_npix[yi, xi] = mslice.sum()
                binned_var[yi, xi] = var[sly, slx][mslice].sum()
            
            binned_err = np.sqrt(binned_var)/binned_npix       
            binned_avg = binned_sci / binned_npix
            
            mask_i = (binned_npix > 0) & (binned_avg/binned_err >7)
            xpi = xpi[mask_i]
            ypi = ypi[mask_i]
            binned_avg = binned_avg[mask_i]
            binned_err = binned_err[mask_i]
            binned_npix = binned_npix[mask_i]
            
        else:
            xpi = xp[mask]
            ypi = yp[mask]
            binned_avg = sci[mask]
            binned_err = np.sqrt(var)[mask]
            binned_npix = mask[mask]*1
        
        if True:
            # Mask pixels in largest binning that don't satisfy S/N cutoff
            clip_mask = mask < 0
            for xi, yi in zip(xpi, ypi):
                slx = slice(xi*bin, xi*bin+bin)
                sly = slice(yi*bin, yi*bin+bin)
                clip_mask[sly, slx] = True
            
            mask &= clip_mask
            
            if bin_factor == original_bin:
                # Only consider the largest blob
                from skimage.morphology import label, closing, square
                label_image = label(mask)
                label_ids = np.unique(label_image)[1:]
                label_sizes = np.array([(label_image == id_i).sum() for id_i in label_ids])
                
                keep_ids = label_ids[label_sizes > 0.2*label_sizes.max()]
                keep_mask = mask < -1
                for i in keep_ids:
                    keep_mask |= label_image == i
                
                mask &= keep_mask
                
            ypb = yp[mask] // bin
            xpb = xp[mask] // bin
                  
        # Cappellari's code
        from vorbin.voronoi_2d_binning import voronoi_2d_binning
        
        print('Run voronoi_2d_binning, bin_factor={0}'.format(bin_factor))
                
        res = voronoi_2d_binning(xpi, ypi, binned_avg, binned_err, targetSN, quiet=True, plot=False, pixelsize=0.1*bin, cvt=True, wvt=True)
            
        binNum, xBin, yBin, xBar, yBar, sn, nPixels, scale = res
    
        # Put Voronoi bins with nPixels > 1 back in the original image
        large_bins = nPixels > 1
        NBINS = len(nPixels)
    
        large_bin_ids = np.arange(NBINS)[large_bins]
        
        # Put bin in original 2D array and store info
        for b0, bin_id in enumerate(large_bin_ids):
            m_i = binNum == bin_id
            xi_bin = xpi[m_i]
            yi_bin = ypi[m_i]
            for xi, yi in zip(xi_bin, yi_bin):
                slx = slice(xi*bin, xi*bin+bin)
                sly = slice(yi*bin, yi*bin+bin)
                mslice = mask[sly, slx]
                full_bin_seg[sly, slx][mslice] = b0+bin_min
            
            # Bin properties
            id_i = b0+bin_min
            xmin = xi_bin.min()*bin-1
            xmax = xi_bin.max()*bin+1
            ymin = yi_bin.min()*bin-1
            ymax = yi_bin.max()*bin+1
            npix = m_i.sum()*bin**2
            bin_data_i = [id_i, bin, xmin, xmax, ymin, ymax, npix]
            full_bin_data.append(bin_data_i)
            
        # Update the mask
        not_in_a_bin = full_bin_seg == -1
        mask &= not_in_a_bin
        
        bin_min = full_bin_seg.max()+1
        if not quiet:
            print('\n\n\n\n\n bin_factor: {0}, bin_min: {1}\n\n\n\n'.format(bin_factor, bin_min))
    
    ## Bin information
    bin_data = np.array(full_bin_data)
    # bin_data_i = [id_i, bin, xmin, xmax, ymin, ymax, npix]
    tab = utils.GTable()
    for i, c in enumerate(['id', 'bin', 'xmin', 'xmax', 'ymin', 'ymax', 'npix']):
        tab[c] = bin_data[:,i]
        if 'min' in c:
            tab[c] -= tab['bin']
        elif 'max' in c:
            tab[c] += tab['bin']  
    
    tab['flux'], tab['err'], tab['area'] = prep.get_seg_iso_flux(sci, full_bin_seg, tab, err=np.sqrt(var))

    binned_flux = prep.get_seg_iso_flux(sci, full_bin_seg, tab, fill=tab['flux']/tab['area'])
    binned_err = prep.get_seg_iso_flux(sci, full_bin_seg, tab, fill=tab['err']/tab['area'])
    binned_area = prep.get_seg_iso_flux(sci, full_bin_seg, tab, fill=tab['area'])
    binned_bin = prep.get_seg_iso_flux(sci, full_bin_seg, tab, fill=tab['bin'])
    
    binned_flux[mask] = sci[mask]
    binned_err[mask] = np.sqrt(var)[mask]
    binned_area[mask] = 1
    binned_bin[mask] = 1
    
    if plot:
        plt.figure(figsize=(12,12))
        plt.imshow(binned_flux,vmin=-0.5,vmax=2)
        
    # Save output into image fits file
    primary_extn = pyfits.PrimaryHDU()
    sci_extn = pyfits.ImageHDU(data=binned_flux,name='SCI')
    err_extn = pyfits.ImageHDU(data=binned_err,name='ERR') 
    pyfits.HDUList([primary_extn, sci_extn, err_extn]).writeto('binned_image.fits', output_verify='fix', overwrite=True)
    

    # Bin other filter images for the target using the obtained Voronoi 2D bins
    import glob
    from astropy.table import Table
    files = glob.glob('*fits.gz')
    files.sort()
    res = {}
    for file in files:
        im = pyfits.open(file)
        f = utils.get_hst_filter(im[0].header).lower()
        print(file, f)
        res[f] = bin_image(im, tab, full_bin_seg, mask)
        
        primary_extn = pyfits.PrimaryHDU()
        sci_extn = pyfits.ImageHDU(data=res[f]['image_flux'],name='SCI')
        err_extn = pyfits.ImageHDU(data=res[f]['image_err'],name='ERR')
        pyfits.HDUList([primary_extn, sci_extn, err_extn]).writeto('binned_{0}_image.fits'.format(f), output_verify='fix',overwrite=True)
        
        # pop out from res dict
        res[f].pop('image_flux')
        res[f].pop('image_err')
        
        t = Table([res[f]], names=(res[f].keys()))
        t.write('binned_{0}_table.fits'.format(f), overwrite=True)

    return binned_flux, binned_err


#----------------------------------------------------------------------

def bin_image(im, tab_in, seg_in, mask_in, bkg_in=None, bg_mask_in=None):
    
    """
    Apply 2D bins specified in "seg_in" to a new image
    
    """ 
    
    from grizli import utils
    from grizli import prep
    import numpy as np
    
    sci_data = im['SCI'].data*1
    var_data = 1/im['WHT'].data
    wht_mask = im['WHT'].data > 0
    
    IS_ACS = sci_data.shape[0] == 2*seg_in.shape[0]
    
    if IS_ACS:
        # ACS
        to_flam = im[1].header['PHOTFLAM']
        to_fnu = 1. #im[1].header['PHOTFNU']
        tab = utils.GTable()
        for c in tab_in.colnames:
            if c[:2] in ['xm', 'ym']:
                tab[c] = 2*tab_in[c]
            else:
                tab[c] = tab_in[c]
        
        sh = sci_data.shape
        mask = np.zeros(sh, dtype=bool)
        seg = np.zeros(sh, dtype=np.int)
        for i in [0,1]:
            for j in [0,1]:
                mask[j::2, i::2] = mask_in
                seg[j::2, i::2] = seg_in
        
    else:
        to_flam = im[0].header['PHOTFLAM']
        to_fnu = im[0].header['PHOTFNU']
        
        tab = tab_in
        mask = mask_in
        seg = seg_in

    if bg_mask_in is None:
        bg_mask = (~mask) & wht_mask & (seg <= 0)
    else:
        bg_mask = bg_mask_in
        
    if bkg_in is None:
        bkg = np.median(sci_data[bg_mask])
    else:
        bkg = bkg_in
        
    sci_data -= bkg
    
    var_data[~wht_mask] = 0
    
    bin_flux, bin_err, bin_area = prep.get_seg_iso_flux(sci_data, seg, tab, err=np.sqrt(var_data))

    image_flux = prep.get_seg_iso_flux(sci_data, seg, tab, fill=bin_flux/bin_area)
    image_err = prep.get_seg_iso_flux(sci_data, seg, tab, fill=bin_err/bin_area)

    image_flux[mask] = sci_data[mask]*1
    image_err[mask] = np.sqrt(var_data)[mask]
    
    res = {}
    res['bin_flux'] = bin_flux
    res['bin_err'] = bin_err
    res['bin_area'] = bin_area
    
    if IS_ACS:
        res['to_flam'] = to_flam
        res['to_fnu'] = to_fnu
        res['image_flux'] = image_flux[0::2, 0::2]*4
        res['image_err'] = image_err[0::2, 0::2]*4
    else:
        res['to_flam'] = to_flam
        res['to_fnu'] = to_fnu
        res['image_flux'] = image_flux
        res['image_err'] = image_err

    res['bkg'] = bkg
    res['bg_mask'] = bg_mask
    
    return res


#--------------------------------------------------------------------------------



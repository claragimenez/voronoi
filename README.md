# voronoi
Code for binning big images based on the Voronoi binning method by Cappellari &amp; Copin (2003)

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

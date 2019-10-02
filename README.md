# voronoi
Code for binning big images based on the Voronoi binning method by Cappellari &amp; Copin (2003)

AUTHOR:
      Gabriel Brammer, University of Copenhagen, gabriel.brammer_at_nbi.ku.dk
      Clara Giménez Arteaga, University of Copenhagen, clara.arteaga_at_nbi.ku.dk

PURPOSE:
      Perform adaptive spatial binning of Integral-Field Spectroscopic
      (IFS) data to reach a chosen constant signal-to-noise ratio per bin.
      The Voronoi method from Cappellari is here applied on big images,
      which are sliced in order to be able to run the method. The binning
      is then applied to the target image obtained with different filters.

EXPLANATION:
      Further information on VORONOI_2D_BINNING algorithm can be found in
      Cappellari M., Copin Y., 2003, MNRAS, 342, 345
      http://adsabs.harvard.edu/abs/2003MNRAS.342..345C

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata.mixins.ndslicing import NDSlicingMixin
from astropy.coordinates import SkyCoord
import astropy.units as u
from sunpy.coordinates import Helioprojective
from crispy.utils import ObjDict
import datetime
from reproject import reproject_exact
from tqdm import tqdm

class RHESSISlicingMixin(NDSlicingMixin):
    """
    This is the class that the ``RHESSI`` object inherits that allows us to slice the object easily.
    """
    def __getitem__(self, item):
        kwargs = self._slice(item)
        cl = self.__class__(**kwargs)
        cl.ind = item
        cl.aux = self.aux
        
        return cl
    
    def _slice(self, item):
        kwargs = {}
        kwargs["filename"] = ObjDict({})
        kwargs["filename"]["data"] = self.data[item]
        kwargs["uncertainty"] = self._slice_uncertainty(item)
        kwargs["mask"] = self._slice_mask(item)
        kwargs["wcs"] = self._slice_wcs(item)
        kwargs["filename"]["header"] = self.header
        
        return kwargs

class RHESSI(RHESSISlicingMixin):
    """
    This is the baseclass for RHESSI imaging spectroscopy data. (Specifically that which is prepped by P.J.A.S. as I have no idea if there's a standard for this kind of thing). RHESSI imaging spectroscopy data consists of a 2D field-of-view integrated in time for a certain number of energy ranges. All of this can be found in the object.

    Parameters
    ----------
    filename : str or ObjDict
        The path to the file.
    wcs : astropy.wcs.WCS, optional
        The WCS to be used for the observations. Due to the unconventional energy axis in the imaging spectroscopy data I would recommend leaving this as None to allow the hidden methods of this class to construct the WCS for you. Default is None.
    uncertainty : numpy.ndarray, optional
        If an uncertainty on the measurements exists, it can be used here. Default is None.
    mask : numpy.ndarray, optional
        A mask array to use for the observations. Default is None.
    """
    def __init__(self, filename, wcs=None, uncertainty=None, mask=None):
        if type(filename) == str:
            self.file = fits.open(filename)[0]
            self.aux = fits.open(filename)[1:]
        elif type(filename) == ObjDict:
            self.file = filename
            
        if wcs == None:
            self.wcs = self._make_rhessi_wcs(self.file)
        else:
            self.wcs = wcs
            
        self.uncertainty = uncertainty
        self.mask = mask
        
    @property
    def data(self):
        return self.file.data
    
    @property
    def header(self):
        return self.file.header

    @property
    def shape(self):
        return self.file.data.shape
    
    @property
    def E(self):
        return self.aux[1].data["IM_EBAND_USED"]
    
    @property
    def E_lower(self):
        return self.header["ENERGY_L"]
    
    @property
    def E_upper(self):
        return self.header["ENERGY_H"]
    
    @property
    def start_time(self):
        return self.header["DATE_OBS"]
    
    @property
    def end_time(self):
        return self.header["DATE_END"]
    
    @property
    def integration_time(self):
        et = datetime.time.fromisoformat(self.end_time[11:])
        st = datetime.time.fromisoformat(self.start_time[11:])
        
        et = datetime.timedelta(hours=et.hour, minutes=et.minute, seconds=et.second)
        st = datetime.timedelta(hours=st.hour, minutes=st.minute, seconds=st.second)
        
        return (et - st).seconds
    
    def _make_rhessi_wcs(self, fn):
        h = fn.header
        rwcs = WCS(h)

        if len(self.shape) == 4:
            rwcs.wcs.ctype = ["HPLN-TAN", "HPLT-TAN", "Energy", "Time"]
            rwcs.wcs.cunit = ["arcsec", "arcsec", "keV", "UTC"]
            rwcs.wcs.crval = [h["XCEN"], h["YCEN"], h["NAXIS3"]//2, h["NAXIS4"]//2]
            rwcs.wcs.crpix = [h["NAXIS1"]//2, h["NAXIS2"]//2, h["NAXIS3"]//2, h["NAXIS4"]//2]
            rwcs.wcs.cdelt = [h["CDELT1"], h["CDELT2"], 1.0, 1.0]
        else:
            rwcs.wcs.ctype = ["HPLN-TAN", "HPLT-TAN", "Energy"]
            rwcs.wcs.cunit = ["arcsec", "arcsec", "keV"]
            rwcs.wcs.crval = [h["XCEN"], h["YCEN"], h["NAXIS3"]//2]
            rwcs.wcs.crpix = [h["NAXIS1"]//2, h["NAXIS2"]//2, h["NAXIS3"]//2]
            rwcs.wcs.cdelt = [h["CDELT1"], h["CDELT2"], 1.0]

        return rwcs
    
    def _stringy_energies(self):
        sE = []
        for er in self.E:
            sE.append(f"{int(er[0])}-{int(er[1])}")
            
        return sE
    
    def energy_range(self, idx):
        return self.E[idx]
    
    def image(self):
        if type(self.ind) == int:
            idx = self.ind
        else:
            idx = self.ind[0]
            
        er = self.energy_range(idx)
        
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1, projection=self.wcs.low_level_wcs)
        im1 = ax1.imshow(self.data, cmap="sunset")
        ax1.set_ylabel("Helioprojective Latitude [arcsec]")
        ax1.set_xlabel("Helioprojective Longitude [arcsec]")
        ax1.set_title(f"{er[0]}-{er[1]} keV")
        fig.colorbar(im1, ax=ax1, orientation="horizontal", label=r"photons cm$^{-2}$ s$^{-1}$ arcsec$^{-2}$")
        fig.show()
        
    def spectrum(self):
        plt.figure()
        plt.bar(self._stringy_energies(), height=self.data, color="k")
        plt.yscale("log")
        plt.xticks(rotation=30)
        plt.xlabel("Energy Range [keV]")
        plt.ylabel(r"HXR Flux [photons cm$^{-2}$ s$^{-1}$ arcsec$^{-2}$]")
        plt.show()
        
    def from_lonlat(self, lon, lat):
        lon, lat = lon << u.arcsec, lat << u.arcsec
        sc = SkyCoord(lon, lat, frame=Helioprojective)
        if len(self.wcs.low_level_wcs.array_shape) == 3:
            if hasattr(self, "ind"):
                if type(self.ind[-2]) == slice and type(self.ind[-1]) == slice:
                    return self.wcs.low_level_wcs._wcs[0,self.ind[-2],self.ind[-1]].world_to_array_index(sc)
                elif type(self.ind[-2]) == slice and type(self.ind[-1]) != slice:
                    return self.wcs.low_level_wcs._wcs[0,self.ind[-2]].world_to_array_index(sc)
                elif type(self.ind[-2]) != slice and type(self.ind[-1]) == slice:
                    return self.wcs.low_level_wcs._wcs[0,:,self.ind[-1]].world_to_array_index(sc)
                else:
                    return self.wcs.low_level_wcs._wcs[0].world_to_array_index(sc)
            else:
                return self.wcs[0].world_to_array_index(sc)
        elif len(self.wcs.low_level_wcs.array_shape) == 2:
            return self.wcs.world_to_array_index(sc)
        else:
            raise NotImplementedError("Too many or too little dimensions.")
            
    def to_lonlat(self, y, x, coord=False, unit=False):
        if coord:
            if len(self.wcs.low_level_wcs.array_shape) == 3:
                if hasattr(self, "ind"):
                    if type(self.ind[-2]) == slice and type(self.ind[-1]) == slice:
                        return self.wcs.low_level_wcs._wcs[0,self.ind[-2],self.ind[-1]].array_index_to_world(y,x)
                    elif type(self.ind[-2]) == slice and type(self.ind[-1]) != slice:
                        return self.wcs.low_level_wcs._wcs[0,self.ind[-2]].array_index_to_world(y,x)
                    elif type(self.ind[-2]) != slice and type(self.ind[-1]) == slice:
                        return self.wcs.low_level_wcs._wcs[0,:,self.ind[-1]].array_index_to_world(y,x)
                    else:
                        return self.wcs.low_level_wcs._wcs[0].array_index_to_world(y,x)
                else:
                    return self.wcs[0].array_index_to_world(y,x)
            elif len(self.wcs.low_level_wcs.array_shape) == 2:
                return self.wcs.array_index_to_world(y,x) 
            else:
                raise NotImplementedError("Too many or too little dimensions.")
        else:
            if unit:
                if len(self.wcs.low_level_wcs.array_shape) == 3:
                    if hasattr(self, "ind"):
                        if type(self.ind[-2]) == slice and type(self.ind[-1]) == slice:
                            sc = self.wcs.low_level_wcs._wcs[0,self.ind[-2],self.ind[-1]].array_index_to_world(y,x)
                            return sc.Tx, sc.Ty
                        elif type(self.ind[-2]) == slice and type(self.ind[-1]) != slice:
                            sc = self.wcs.low_level_wcs._wcs[0,self.ind[-2]].array_index_to_world(y,x)
                            return sc.Tx, sc.Ty
                        elif type(self.ind[-2]) != slice and type(self.ind[-1]) == slice:
                            sc = self.wcs.low_level_wcs._wcs[0,:,self.ind[-1]].array_index_to_world(y,x)
                            return sc.Tx, sc.Ty
                        else:
                            sc = self.wcs.low_level_wcs._wcs[0].array_index_to_world(y,x)
                            return sc.Tx, sc.Ty
                    else:
                        sc = self.wcs[0].array_index_to_world(y,x)
                        return sc.Tx, sc.Ty
                elif len(self.wcs.low_level_wcs.array_shape) == 2:
                    sc = self.wcs.array_index_to_world(y,x) 
                    return sc.Tx, sc.Ty
                else:
                    raise NotImplementedError("Too many or too little dimensions.")
            else:
                if len(self.wcs.low_level_wcs.array_shape) == 3:
                    if hasattr(self, "ind"):
                        if type(self.ind[-2]) == slice and type(self.ind[-1]) == slice:
                            sc = self.wcs.low_level_wcs._wcs[0,self.ind[-2],self.ind[-1]].array_index_to_world(y,x)
                            return sc.Tx.value, sc.Ty.value
                        elif type(self.ind[-2]) == slice and type(self.ind[-1]) != slice:
                            sc = self.wcs.low_level_wcs._wcs[0,self.ind[-2]].array_index_to_world(y,x)
                            return sc.Tx.value, sc.Ty.value
                        elif type(self.ind[-2]) != slice and type(self.ind[-1]) == slice:
                            sc = self.wcs.low_level_wcs._wcs[0,:,self.ind[-1]].array_index_to_world(y,x)
                            return sc.Tx.value, sc.Ty.value
                        else:
                            sc = self.wcs.low_level_wcs._wcs[0].array_index_to_world(y,x)
                            return sc.Tx.value, sc.Ty.value
                    else:
                        sc = self.wcs[0].array_index_to_world(y,x)
                        return sc.Tx.value, sc.Ty.value
                elif len(self.wcs.low_level_wcs.array_shape) == 2:
                    sc = self.wcs.array_index_to_world(y,x) 
                    return sc.Tx.value, sc.Ty.value
                else:
                    raise NotImplementedError("Too many or too little dimensions.")

def rhessi_reproject(rhessi, crisp, derot=False):
    """
    This function is used to reproject the RHESSI data onto the CRISP frame using an exact method of `flux-conserving spherical polygon intersection <http://montage.ipac.caltech.edu/docs/algorithms.html>`_.

    Parameters
    ----------
    rhessi : RHESSI
        The object of the RHESSI observations to reproject.
    crisp : crispy.crisp.CRISP
        The CRISP observsations containing the WCS to reproject the RHESSI observations onto.
    derot : bool, optional
        Whether or not the crisp data needs to be derotated before the reprojection is performed.

    Returns
    -------
    rhessi_new : numpy.ndarray
        The RHESSI data reprojected onto the CRISP WCS.
    """
    if derot:
        crisp.reconstruct_full_frame()

    rhessi_new = np.zeros((rhessi.shape[-3], crisp.shape[-2], crisp.shape[-1]))
    for j, r in enumerate(tqdm(rhessi.data)):
        rhessi_new[j], _ = reproject_exact((r, rhessi[j].wcs), crisp[0].wcs, shape_out=crisp[0].shape)

    return rhessi_new
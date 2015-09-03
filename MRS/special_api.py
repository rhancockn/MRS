"""
MRS.api
-------

Functions and classes for representation and analysis of MRS data. This is the
main module to use when performing routine analysis of MRS data.

"""
import numpy as np
import scipy.stats as stats
import nibabel as nib
import warnings

import MRS.analysis as ana
import MRS.utils as ut
import MRS.api as api

try:
    import MRS.freesurfer as fs
except ImportError:
    warnings.warn("Nipype is not installed. Some functions might not work")
    
import nitime.timeseries as nts
import scipy.fftpack as fft
import scipy.stats as stats
import numpy.ma as ma
from scipy.optimize import curve_fit
from scipy.optimize import leastsq

class GABA(api.GABA):
    """
    Class for analysis of GABA MRS from the MEGA-SPECIAL sequence.
    """

    def __init__(self,
                 in_data,
                 line_broadening=2.,
                 zerofill=4096,
                 filt_method=None,
                 spect_method=dict(NFFT=1024, n_overlap=1023, BW=2),
                 min_ppm=0.0,
                 max_ppm=4.0,
                 sampling_rate=2500.):
        """
        Parameters
        ----------

        in_data : str
            Path to a nifti file containing MRS data.

        line_broadening : float (optional)
           How much to broaden the spectral line-widths (Hz). Default: 5
           
        zerofill : int (optional)
           How many zeros to add to the spectrum for additional spectral
           resolution. Default: 2048
           We want a fair amount of increased resolution to make frequency adjustments

        filt_method : dict (optional)
            How/whether to filter the data. Default: None (#nofilter)

        spect_method: dict (optional)
            How to derive spectra. Per default, a simple Fourier transform will
            be derived from apodized time-series, but other methods can also be
            used (see `nitime` documentation for details)
            
        min_ppm, max_ppm : float
           The limits of the spectra that are represented

        sampling_rate : float
           The sampling rate in Hz.
        
        """
        if isinstance(in_data, str):
            # The nifti files follow the strange nifti convention, but we want
            # to use our own logic, which is transients on dim 0 and time on
            # dim -1:
            self.raw_data = self.read_nifti(in_data, sampling_rate)
        elif isinstance(in_data, np.ndarray):
            self.raw_data = in_data

        
        self.preprocess(line_broadening, zerofill, filt_method, sampling_rate, min_ppm, max_ppm)

    def preprocess(self, line_broadening, zerofill, filt_method, sampling_rate, min_ppm, max_ppm):
        """
            Preprocess the water supressed FIDs
        """
        self.data = self.raw_data
        self.data = self.raw_data*np.exp(-np.pi*np.arange(self.data.shape[-1])*line_broadening/sampling_rate)


        self.data, params = self.align_fids_each()

        f_hz, spectra = ana.get_spectra(self.data, line_broadening=None, zerofill=zerofill, filt_method=filt_method, sampling_rate=sampling_rate)
        ppm=ut.freq_to_ppm(f_hz)
        edit_on, edit_off, ppm_sig = self.wsvd_combine(spectra,ppm)
        self.dbg_edit_off =np.copy(edit_off)
        self.dbg_edit_on = np.copy(edit_on)
        
        
        ii=np.ones(edit_on.shape[0],dtype='bool')
        jj=np.arange(edit_on.shape[0])

        model_off, signal, params_off, resid_off = ana.fit_cho_cr_model(edit_off, ppm_sig)
        ii = self._reject_cho_cr(params_off,resid_off,model_off,ii)

        model_on, signal, params_on, resid_on = ana.fit_cho_cr_model(edit_on, ppm_sig)
        ii = self._reject_cho_cr(params_on,resid_on,model_on,ii)
       
        for i in jj[ii]:
            edit_off[i,:]=ut.phase_correct_zero(edit_off[i,:],params_off[i,4])
            edit_on[i,:]=ut.phase_correct_zero(edit_on[i,:],params_off[i,4])
            edit_off[i,:]=np.roll(edit_off[i,:],int(np.round((params_off[i,0]-3.02)/(ppm_sig[0]-ppm_sig[1]))))
            edit_on[i,:]=np.roll(edit_on[i,:],int(np.round((params_off[i,0]-3.02)/(ppm_sig[0]-ppm_sig[1]))))

       
        self.echo_off = edit_off[ii,:]
        self.echo_on = edit_on[ii,:]
        sum_spectra = self.echo_off
        #sum_spectra=self.echo_off
        self.n_kept=sum_spectra.shape[0]
        print('Keeping %d' % self.n_kept)
        sum_spectra = np.array(np.mean(sum_spectra,0), ndmin=2)


        

        
        #average here so we fit to the average. The other functions will treat this as a single transient
        #TODO: Clean this up
        self.diff_spectra = np.array(np.mean(self.echo_on-self.echo_off,0),ndmin=2)
        #self.sum_spectra = np.array(sum_spectra - baseline,ndmin=2)
        self.sum_spectra=sum_spectra
        self.f_ppm = ppm_sig
        self.idx = ut.make_idx(self.f_ppm, 0, 4)

    def read_nifti(self, fname, sampling_rate=2500.):
        """
            Read MEGA-SPECIAL data from a NIMS NIfTI file
        """
        
        sq_n_tran = lambda this: np.transpose(this.squeeze(), [1,2,0])
        nii = nib.load(fname)
        ts = nii.get_data()
        on_isis1 = sq_n_tran(ts[:,:, :,::8])
        on_isis2 = sq_n_tran(ts[:,:, :, 2::8])
        off_isis1 = sq_n_tran(ts[:,:,:, 4::8])
        off_isis2 = sq_n_tran(ts[:,:,:, 6::8])
        fid = (on_isis1-on_isis2, off_isis1-off_isis2)
        pre_ts = np.concatenate([aa[None] for aa in fid], 0)
        return nts.TimeSeries(np.transpose(pre_ts, [1,0,2, 3]), sampling_rate=sampling_rate).data


    def align_fids_each(self, sampling_rate=2500.):
        """
            Do preliminary phase and frequency alignment on each fid (Near et al, 2014 MRM)
        """
        def fid_delta(params,t,fid,fid_ref):
            f,theta = params
            G = fid * np.exp(-1j*np.pi*(2*t*f+theta/180))
            G_component = np.concatenate((np.real(G),np.imag(G)))
            fid_ref_component = np.concatenate((np.real(fid_ref),np.imag(fid_ref)))
            return fid_ref_component-G_component
            
        #zero order phase and frequency correction on each fid
        #separately for edit on and off
        #do phase and frequency alignment in the time domain (Near et al, 2014 MRM)

        #cut out residual water


        dt=1/sampling_rate
        t0=np.arange(0,.2,dt)
        t=np.arange(0,self.data.shape[-1]*dt,dt)
        
        max_channel=np.argmax(np.mean(np.abs(self.data[:,1,:,0]),0).squeeze())
        max_fid_transient=np.argmax(self.data[:,1,max_channel,0].squeeze())
        fid_ref=self.data[max_fid_transient,1,max_channel,0:len(t0)]
        #fid_ref = self.data[1,:,1,0:len(t0)]
        #a=np.angle(fid_ref[:,0])
        a=np.angle(fid_ref[0])
        fid_ref=fid_ref*np.exp(-1j*a) #align the reference phase to 0
        #fid_ref=fid_ref*np.exp(-1j*a)
        fid_aligned = np.zeros(self.data.shape,dtype='complex128')
        params = np.zeros(self.data.shape[:-1] + (2,))*np.nan
        for chan in range(32):
            #max_channel=np.argmax(np.mean(np.abs(self.data[ex,1,:,0]),0).squeeze())
            #fid_ref=self.data[1,1,chan,0:len(t0)]
            #fid_ref = fid_ref*np.exp(-1j*np.angle(fid_ref[0]))
            for ex in range(64):
                for echo in range(2):

                    sig=self.data[ex,echo,chan,:]
                    p,it=leastsq(fid_delta,[0,0],args=(t0,sig[0:len(t0)],fid_ref))
                    fid_aligned[ex,echo,chan,:]=sig * np.exp(-1j*np.pi*(2*t*p[0]+p[1]/180))
                    #fid_aligned[ex,echo,chan,:]=sig*np.exp(-1j*np.angle(sig[0]))
                    params[ex,echo,chan] = p
        return fid_aligned,params
        

    #implement WSVD, using 8-14ppm for a noise estimate (idx: 26-339)
    #a simple SVD fails on memory requirements for individual acquistions, so need to do this on the individual spectra, with the full noise matrix after above alignment
    #not the ideal way to do this
    def wsvd_combine(self, spectra,ppm):
        
        
        noise_ppm_idx=ut.make_idx(ppm, -4.8,0)
        signal_ppm_idx=ut.make_idx(ppm, 0,4.0)
        
        noise_spectra=spectra[...,noise_ppm_idx]
        signal_spectra = spectra[..., signal_ppm_idx]
        #signal_spectra=np.mean(signal_spectra,0)
        #signal_spectra=np.array(np.mean(signal_spectra,0),ndmin=4)
        signal_len =signal_spectra.shape[-1]
        n_transients = signal_spectra.shape[0]
        combined_spectra=np.array(np.zeros([n_transients,2,signal_len],dtype='complex128'))

        #signal_spectra=np.matrix(np.hstack((spectra_on[...,signal_ppm_idx],spectra_off[...,signal_ppm_idx])))
        
        noise = np.matrix(np.zeros([32,2*64*noise_spectra.shape[-1]],dtype='complex128'))
        
        for i in range(32):
            noise[i,:]=noise_spectra[:,:,i,:].flatten()

        #noise transform (eqn 4)
        #sigma=np.matrix(np.cov(noise))
        sigma=noise*noise.H
        d,x=np.linalg.eigh(sigma)
        winv = x*(1/np.sqrt(2)*np.matrix(np.diag(1/np.sqrt(d))))
        

        for i in range(n_transients):
            sig = np.matrix(np.hstack((signal_spectra[i,0,:,:].squeeze(),signal_spectra[i,1,:,:].squeeze())))
            #sig = np.matrix(np.hstack((signal_spectra[0,:,:].squeeze(),signal_spectra[1,:,:].squeeze())))
            #whiten
            whitened=winv*sig
            #signal decomposition (eqn 14)
            u,psi,v = np.linalg.svd(sig)
                
            #reconstructed
            max_a_idx=np.argmax(np.abs(u[:,0]))
            phi=np.exp(-1j*np.angle(u[max_a_idx,0]))
            
            u=u[:,0]/np.dot(u[:,0].H, u[:,0])
            s=(1/np.sqrt(2)*np.matrix(np.diag(1/np.sqrt(d))))*u*(psi[0]*v[0,:])
            combined_spectra[i,0,:]=np.sum(s,0).T[0:signal_len].squeeze()
            combined_spectra[i,1,:]=np.sum(s,0).T[signal_len:(signal_len*2)].squeeze()
            #combined_spectra[0,:]=np.sum(s,0).T[0:signal_len].squeeze()
            #combined_spectra[1,:]=np.sum(s,0).T[signal_len:(signal_len*2)].squeeze()

        edit_on=np.array(combined_spectra[:,0,:].squeeze(),ndmin=2)
        edit_off=np.array(combined_spectra[:,1,:].squeeze(),ndmin=2)
        #edit_on=np.array(combined_spectra[0,:].squeeze(),ndmin=2)
        #edit_off=np.array(combined_spectra[1,:].squeeze(),ndmin=2)
        return edit_on, edit_off, ppm[signal_ppm_idx]
    #TODO: refine rejection (e.g. on params)
    #TODO: get residual variance    
    def _reject_cho_cr(self, params, resid, model, ii):
        """
        Helper function to reject outliers based on mean amplitude
        """
        maxamps = np.nanmax(np.abs(model),-1)
        z_score = (maxamps - np.nanmean(maxamps,0))/np.nanstd(maxamps,0)
        z_area = (params[1,:] - np.nanmean(params[1,:]))/np.nanstd(params[1,:])
        z_phase = (params[4,:] - np.nanmean(params[4,:]))/np.nanstd(params[4,:])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outlier_idx = np.where((np.abs(z_score)>3.0))[0]
            outlier_area_idx = np.where((np.abs(z_area)>3.0))[0]
            outlier_phase_idx = np.where((np.abs(z_phase)>3.0))[0]
            extreme_idx = np.where(maxamps>1)[0]
            nan_idx = np.where(np.isnan(resid))[0]
            fit_idx = np.where(resid>.1)[0]

            outlier_idx = np.unique(np.hstack([nan_idx, fit_idx, outlier_idx,extreme_idx,outlier_area_idx,outlier_phase_idx]))
            ii[outlier_idx] = 0
            

        return ii

    def fit_gaba(self, reject_outliers=False, fit_lb=2.8, fit_ub=3.2,
                 phase_correct=False, fit_func=(ana.fit_gaba_doublet, ut.gaba_doublet)):
        """
        Fit either a single Gaussian, or a two-Gaussian to the GABA 3 PPM
        peak.

        Parameters
        ----------
        reject_outliers : float
            Z-score criterion for rejection of outliers, based on their model
            parameter

        fit_lb, fit_ub : float
            Frequency bounds (in ppm) for the region of the spectrum to be
            fit.

        phase_correct : bool
            Where to perform zero-order phase correction based on the fit of
            the creatine peaks in the sum spectra
            Off by default, as we phase correct each transient in the constructor

        fit_func : None or callable (default None).
            If this is set to `False`, an automatic selection will take place,
            choosing between a two-Gaussian and a single Gaussian, based on a
            split-half cross-validation procedure. Otherwise, the requested
            callable function will be fit. Needs to conform to the conventions
            of `fit_gaussian`/`fit_two_gaussian` and
            `ut.gaussian`/`ut.two_gaussian`.

        """
        self.gaba_idx = ut.make_idx(self.f_ppm, fit_lb, fit_ub)

        self.gaba_model, self.gaba_signal, self.gaba_params,self.gaba_resid = ana.fit_gaba_doublet(self.diff_spectra,self.f_ppm,fit_lb,fit_ub)
        self.gaba_auc, self.gaba_integral = self._calc_auc(ut.gaba_doublet, self.gaba_params, self.gaba_signal, self.gaba_idx,amp_idx=[4,5])

    def fit_glx(self, fit_lb=3.6, fit_ub=3.9):
        self.glx_idx = ut.make_idx(self.f_ppm, fit_lb, fit_ub)
        self.glx_model, self.glx_signal, self.glx_params, self.glx_resid = ana.fit_glx_doublet(self.diff_spectra, self.f_ppm, fit_lb, fit_ub)
        self.glx_auc, self.glx_integral = self._calc_auc(ut.gaba_doublet, self.glx_params, self.glx_signal, self.glx_idx, amp_idx=[4,5])


    def fit_naa(self, fit_lb=1.8, fit_ub=2.4):
        self.naa_idx = ut.make_idx(self.f_ppm, fit_lb, fit_ub)

        self.naa_model, self.naa_signal, self.naa_params = ana.fit_lorentzian(np.array(np.mean(self.echo_off,0),ndmin=2),self.f_ppm,fit_lb,fit_ub)
        self.naa_auc, self.naa_integral = self._calc_auc(ut.lorentzian, self.naa_params, self.naa_signal, self.naa_idx,amp_idx=1)

    def fit_cho_cr(self, fit_lb=2.7, fit_ub=3.5):

        self.cho_cr_idx = ut.make_idx(self.f_ppm, fit_lb, fit_ub)
        self.cho_cr_model, self.cho_cr_signal, self.cho_cr_params, self.cho_cr_resid = ana.fit_cho_cr_model(np.array(np.mean(self.echo_off,0),ndmin=2),self.f_ppm, fit_lb, fit_ub)
        cho_params = np.copy(self.cho_cr_params[0])
        cr_params = np.copy(self.cho_cr_params[0])
        cho_params = cho_params[[0,2,3,4,5,6]]
        cho_params[0] = cho_params[0] +.18
        cr_params = cr_params[[0,1,3,4,5,6]]
        
       

        self.cho_model = ut.lorentzian(self.f_ppm[self.cho_cr_idx], *cho_params )
        self.cr_model = ut.lorentzian(self.f_ppm[self.cho_cr_idx], *cr_params)
        cho_params=np.array(cho_params,ndmin=2)
        cr_params=np.array(cr_params,ndmin=2)
        self.cho_auc, _ = self._calc_auc(ut.lorentzian, cho_params, self.cho_cr_signal, self.cho_cr_idx,amp_idx=1)
        self.cr_auc, _ = self._calc_auc(ut.lorentzian, cr_params, self.cho_cr_signal, self.cho_cr_idx,amp_idx=1)

    def fit_all(self):
        self.fit_naa()
        self.fit_gaba()
        self.fit_glx()
        self.fit_cho_cr()

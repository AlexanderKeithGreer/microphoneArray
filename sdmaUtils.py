import numpy as np
import scipy.signal as sig
import scipy.linalg as la

import utility as ut

def find_corr_from_source(lnm,theta,phi):
    """ Finds the sensor correlation based on given given information

    azimuth     = phi
    elevation   = theta
    """
    if (np.shape(lnm) != np.shape(np.zeros(3))):
        print("Error, parameter 'lmn'  wrong size!")

    if (f_use > fs):
        print("Error, f_use > fs")

    #idk where the 2 comes from in the whole cancellation expression
    omega = 2*np.pi*(f_use/fs)

    sth_cph = np.sin(theta)*np.cos(phi)
    sth_sph = np.sin(theta)*np.sin(phi)
    c_th    = np.cos(phi)

    tau_mn = (lnm[0]*sth_cph + lnm[1]*sth_sph + lnm[2]*c_th)
    Gamma_partial = np.exp(tau_mn)
    return Gamma_partial

def find_Gamma_vv(elements,n,m,fs,c):
#def find_Gamma_vv():
    """
    This requires integrating across many possible direction vectors:

    Gamma_nm = exp(j*Omega*Tau_nm)
    Tau_nm = (fs/c)*(l_xmn*sin(theta)*cos(phi) )

    """
    lnm = np.zeros(3)
    lnm[0] = np.abs(elements[n,0] - elements[m,0])
    lnm[1] = np.abs(elements[n,1] - elements[m,1])
    lnm[2] = np.abs(elements[n,2] - elements[m,2])

    f_use = 3e3

    # We need to have the density of elevation angle samples correct
    #   this requires scaling the number by cos(theta)
    # Where theta is the elevation angle, and phi the azimuth
    #Let aNGLE_unIT be np.pi
    a_un = np.pi
    theta_range = np.linspace(-a_un/2,a_un/2,1800)
    integral = 0
    for theta in range(len(theta_range)):
        phi_range = np.linspace(a_un/2,3*a_un/2,np.int(np.abs(np.round(1800*np.cos(theta_range[theta])))))
        for phi in range(len(phi_range)):
            # Do the integral. We need the d_theta and d_phi
            d_theta = theta_range[1] - theta_range[0]
            d_phi = phi_range[1] - phi_range[0]
            integral += find_corr_from_source(lnm,theta,phi)*d_theta*d_phi
    Gamma_vv = integral/(np.pi**2)
    return Gamma_vv

def invert_Gamma_vv_at_f(freqs,Gamma_vv,mu):
    """
    This section actually forms the matrix we care about

    """

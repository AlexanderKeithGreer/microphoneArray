import numpy as np
import numpy.fft as nf
import scipy.signal as sig
import scipy.linalg as la
import utility as ut

## This section requires significant refactoring.
##  I'm going to abandon it and copy-paste in other code for use in here!

def find_corr_from_source(lnm,theta,phi,c,fs,f_use):
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

    sth_sph = np.cos(phi)*np.cos(theta)
    sth_cph = np.cos(phi)*np.sin(theta)
    c_th    = np.sin(phi)

    tau_mn = (fs/c)*(lnm[0]*sth_cph + lnm[1]*sth_sph + lnm[2]*c_th)
    Gamma_partial = np.exp(1j*tau_mn*omega)
    return Gamma_partial


def find_Gamma_vv(elements,n,m,fs,c,f_use):
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

    # We need to have the density of elevation angle samples correct
    #   this requires scaling the number by cos(theta)
    # Where theta is the elevation angle, and phi the azimuth
    #Let aNGLE_unIT be np.pi
    a_un = np.pi
    theta_range = np.linspace(-a_un/2,a_un/2,180)
    integral = 0
    for theta in range(len(theta_range)):
        phi_range = np.linspace(a_un/2,3*a_un/2,np.int(np.abs(np.round(180*np.cos(theta_range[theta])))))
        for phi in range(len(phi_range)):
            # Do the integral. We need the d_theta and d_phi
            d_theta = theta_range[1] - theta_range[0]
            d_phi = phi_range[1] - phi_range[0]
            integral += find_corr_from_source(lnm,theta,phi,c,fs,f_use)*d_theta*d_phi
    Gamma_vv = integral/(np.pi**2)
    return Gamma_vv

def generate_Gamma(elements,fs,c,f_use):
    """
    Please no. No. No. No.
    """

    Gamma = np.zeros((len(elements),len(elements)),dtype=np.complex128)
    for n in range(len(elements)):
        for m in range(len(elements)):
            Gamma[n,m] = find_Gamma_vv(elements,n,m,fs,c,f_use)

    Gamma = Gamma/Gamma[0,0]
    print("Gamma=\n",Gamma)
    return Gamma

def solve_MVDR_SDB(elements,fs,c,mu,f_use,d):
    """
    Derive the coef for a single frequency
    """

    #Find spatial correlation matrix
    sc = generate_Gamma(elements,fs,c,f_use)

    #Solve the equation I wrote down from Microphone DSP, pg ???
    in_brack = la.inv(sc + mu*np.eye(len(sc)))
    print(np.shape(d))
    top_line = np.matmul(in_brack,d)
    bottom_line = np.matmul((d.T).conj(),np.matmul(in_brack,d))

    coef = top_line / bottom_line
    return coef

def get_sdb_coef(d,elements,fs,c,freqs,freq_max_index):
    """The big kahuna"""
    Hw = np.zeros(np.shape(d),dtype=np.complex128)
    smooth_len = 4
    n_filter_coef = len(freqs)

    for freq in range(freq_max_index):
        Hw[freq,:] = solve_MVDR_SDB(elements,fs,c,0.01,freqs[freq],d[freq])

    for freq in range(smooth_len):
        Hw[freq+freq_max_index,:] = Hw[freq_max_index,:]*0.8

    filter_end = np.int64(smooth_len+freq_max_index)
    filter_start = n_filter_coef - filter_end + 1
    Hw[filter_start:,:] = np.flip(np.conj(Hw[1:filter_end,:]),axis=0)

    print("\nThis is what we want:\n",Hw)
    hw = nf.ifft(Hw.T)
    return hw

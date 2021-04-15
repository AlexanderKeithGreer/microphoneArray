import numpy as np
import numpy.fft as nf
import scipy.signal as sig
import scipy.linalg as la
import utility as ut
import matplotlib.pyplot as plt

def find_Gamma_vv(elements,f_use,c):
    """
    Find the coherence matrix for a set of elements
    This is assuming diffuse noise.
    This function should be blazing fast compared to the other one

    INPUTS:
    elements    - Nx3 matrix of x,y,z values
    fs          - Sampling frequency
    c           - Propagation speed
    OUTPUTS:
    gamma_vv    - Coherence matrix (unregularised)
    """

    n_elements = len(elements)

    distances_per_element = np.zeros((n_elements,n_elements))

    #Find the l_nm matrix
    for n in range(n_elements):
        for m in range(n_elements):
            distances_per_element[n,m] = la.norm(elements[n]-elements[m])

    #Convert to gamma_vv
    #The 2*np.pi is implicit. Don't add it!
    Gamma_vv = np.sinc(distances_per_element*(f_use/c))
    return Gamma_vv

def solve_MVDR_SDB(elements,c,mu,f_use,d):
    """
    Derive the coef for a single frequency
    """

    #Find spatial correlation matrix
    sc = find_Gamma_vv(elements,f_use,c)

    #Solve the equation I wrote down from Microphone DSP, pg ???
    in_brack = la.inv(sc + mu*np.eye(len(sc)))
    print("shape_d = ",np.shape(d))
    top_line = np.matmul(in_brack,d)
    bottom_line = np.matmul((d.T).conj(),np.matmul(in_brack,d))

    coef = top_line / bottom_line
    return coef

def get_sdb_coef(d,elements,fs,c,freqs,freq_max_index):
    """The big kahuna"""
    smooth_len = 4
    n_filter_coef = len(freqs)
    Hw = np.zeros(np.shape(d),dtype=np.complex128)

    for freq in range(freq_max_index):
        print(Hw.shape)
        Hw[freq,:] = solve_MVDR_SDB(elements,c,0.001,freqs[freq],d[freq])

    for freq in range(smooth_len):
        Hw[freq+freq_max_index,:] = Hw[freq_max_index,:]*0.8

    filter_end = np.int64(smooth_len+freq_max_index)
    filter_start = n_filter_coef - filter_end + 1
    Hw[filter_start:,:] = np.flip(np.conj(Hw[1:filter_end,:]),axis=0)

    print("\nThis is what we want:\n",Hw)
    #hw = np.float64(nf.ifft(Hw.T))
    plt.plot()
    hw = nf.fftshift((nf.ifft(Hw.T)))
    print("Hopefully Real!\n", hw)
    return hw

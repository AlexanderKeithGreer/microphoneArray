import numpy as np
import numpy.fft as nf
import scipy.signal as sig
import scipy.linalg as la
import utility as ut
import matplotlib.pyplot as plt


def solve_cdma_sys_6(freq,r,c,null_angles,array_th):
    """ For a single frequency, solve the cd linear system """
    A = np.zeros((6,6),dtype=np.complex128)

    #These values are fixed, and will always be like this for this to work
    c1 = np.array([0,1,0,0,0,1]) # H2(omega) == H6(omega)
    c2 = np.array([0,0,1,0,1,0]) # H3(omega) == H5(omega)
    b = np.array([[1],[0],[0],[0],[0],[0]])

    A[0,:] = np.conj(ut.find_steering_vector_t(freq,r,c,array_th,0))
    A[1,:] = np.conj(ut.find_steering_vector_t(freq,r,c,array_th,null_angles[0]))
    A[2,:] = np.conj(ut.find_steering_vector_t(freq,r,c,array_th,null_angles[1]))
    A[3,:] = np.conj(ut.find_steering_vector_t(freq,r,c,array_th,null_angles[2]))
    A[4,:] = c1
    A[5,:] = c2

    H = (np.matmul(la.inv(A),b))
    H = H.flatten()
    return H


#This method is based on solving a system of linear equations over a range of frequencies
# But, it's really hard to just define this system for an arbitrary no of elements
# Thus, I define it for a random number of
def get_cd_coef_6(r,c,array_th,null_angles,freqs,freq_min,freq_max):
    """ Broadband differential microphone array - frequency sampling method

    INPUTS
    r           :   Array radius in metres
    c           :   Speed of propagation
    array_th:   Angles associated with the UCA elements from centre
    null_angles :   Look angles associated with a null. length must be 3
    freqs       :   Frequencies associated with the frequency sampling
    freq_min &
    freq_max    :   Frequency to start and stop the sampling


    OUTPUTS
    hw          : Time domain filter coef
    """

    if len(null_angles) != 3:
        print("Error! Wrong Number of Null Angles.")
        print("Current Number of Nulls is ", len(null_angles))

    n_elements = len(array_th)
    if (n_elements != 6):
        print("Error! Wrong Number of Array Elements.")
    n_coef = len(freqs)

    freq_max_index = int(min([ f for f in freqs if f >= freq_max ])/freqs[1])
    freq_min_index = int(max([ f for f in freqs if f <= freq_min ])/freqs[1])

    #Compute the frequency domain representation
    Hw = np.zeros((n_elements,n_coef),dtype=np.complex128)
    for freq in range(n_coef):
        if (freqs[freq] > freq_min and freqs[freq] < freq_max):
            Hw[:,freq] = solve_cdma_sys_6(freqs[freq],r,c,null_angles,array_th)


    filter_end = freq_max_index
    filter_start = n_coef - filter_end + 1
    Hw[:,filter_start:] = np.flip(np.conj(Hw[:,1:filter_end]),axis=1)

    plt.plot(np.abs(Hw[1]),label="abs")
    hw = nf.fftshift((nf.ifft(Hw)))
    hw = hw*sig.hamming(n_coef)

    return hw

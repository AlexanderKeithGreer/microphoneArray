import numpy as np
import numpy.fft as nf
import scipy.signal as sig
import numpy.linalg as la
import matplotlib.pyplot as plt

def give_delays(source, array,c,fs,round=True):
    """ Give desired delays for a position """
    no_elements = len(array[:,0])
    delays = np.zeros(no_elements)

    for element in range(no_elements):
        distance = la.norm(source - array[element])
        if (round==True):
            delays[element] = np.int(np.round((distance/c)*fs))
        else:
            delays[element] = (distance/c)*fs

    return delays

def conv_delays_to_freq(delays, fs, freqs):
    """
    Converts the delays to a function of frequency. We could use an fft, but
    I want the samples nice and good.

    I use the identity:

    Inputs
    delays  : n*1 vector of integer (?) delay values
    fs      : sampling frequency
    freqs   : m*1 vector of frequencies

    Outputs
    delays_f: n*m sampling frequency delays
    """
    freqs_normalised = 2*np.pi*freqs/fs
    delays_f = np.zeros((len(freqs),len(delays)),dtype=np.complex128)
    print("p.shape(delays_f) =  ",np.shape(delays_f))

    for delay in range(len(delays)):
        delays_f[:,delay] = np.exp(-1j*freqs_normalised*delays[delay])
    print("freqs=",freqs)
    return delays_f

def delay_signal_across_elements(signal,delays):
    """Spread a signal across elements"""
    n_elements = len(delays)
    print("n_elements= ",n_elements)
    l_signal = len(signal)
    print("l_signal= ",l_signal)

    channels = np.zeros((n_elements,l_signal))
    for channel in range(len(delays)):
        signal_endpoint = l_signal-delays[channel]
        channels[channel,delays[channel]:] = signal[:signal_endpoint]

    return(channels)

def filter_sum(channels,filters):
    """Filter each channel and then sum them"""
    print("np.shape(channels) = ",np.shape(channels))
    print("np.shape(filters) = ",np.shape(filters))
    filteredch = sig.fftconvolve(channels,filters,mode='same',axes=1)
    fsumed = np.sum(filteredch,axis=0)
    return fsumed

def find_steering_vector(freqs,r,element_angles,look_angle,c):
    """ Finds a steering vector across a range of frequencies for a given angle.
        Note that this form requires a UCA!
    """
    steering_vector_f = np.zeros((len(element_angles),len(freqs)),dtype=np.complex128)

    #full angles, partial name for brevity
    th_full = look_angle - element_angles
    for freq in range(len(freqs)):
        steering_vector_f[:,freq] = np.exp(1j*freqs[freq]*(r/c)*np.cos(th_full))
    return steering_vector_f.T

def find_steering_vector_t(freq,r,c,element_angles,look_angle):
    """ Simple, two line, "this is the steering vector" equation
        Frequency is not radial!
    """
    th_full = look_angle - element_angles
    steering_vector = np.exp(2j*freq*(r/c)*np.pi*np.cos(th_full))
    return steering_vector

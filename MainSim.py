import numpy as np
import scipy.linalg as la
import scipy.signal as sig
import scipy.io.wavfile as wavf
import matplotlib.pyplot as plt
import SignalToArray as STA
import numpy.fft as nfft


import sdmaDesign as sd
import utility as ut

def writeout(input,fname):
    """Downsamples a 100kHz audio to 50kHz"""
    output = sig.resample_poly(input,1,2)
    filter = sig.firwin(250000,(5000/50000))
    output = np.convolve(output,filter,mode='same')

    print(min(output))
    output -= min(output)
    print(max(output))
    output = np.uint8(np.round(output*((2**8)/max(output))))

    wavf.write(fname,50000,output)
    rate,data = wavf.read(fname)
    plt.plot(10*np.log10(np.abs(nfft.fft(filter))),label='Filter')
    plt.plot(10*np.log10(np.abs(nfft.fft(data))),label='Data')
    plt.plot(10*np.log10(np.abs(nfft.fft(output))),label='Output')
    plt.legend()
    plt.show()

def delay_sum(channels, delays):
    no_channels = len(channels[:,0])
    no_samples = len(channels[0,:])
    no_samples_end = no_samples + np.int(max(delays))
    #weights = sig.hamming(no_channels)
    weights = np.ones(no_channels)

    result = np.zeros(no_samples)
    for channel in range(no_channels):

        to_position = no_samples - int(delays[channel])
        delayed = channels[channel,:to_position]
        filler = np.zeros(np.int(delays[(channel)]))
        result = result + weights[channel]*np.append(filler,delayed)

    return result

#def give_delays(source, array,c,fs):
#    """ Give desired delays for a position """
#    no_elements = len(array[:,0])
#    delays = np.zeros(no_elements)
#
#    for element in range(no_elements):
#        distance = la.norm(source - array[element])
#        delays[element] = np.int(np.round((distance/c)*fs))
#
#    return delays

def convert_to_mvdr(delays,mu,f_use,fs):
    """ Take a set of delays, and convert to mvdr"""

    d = np.zeros((len(delays),1),dtype=np.complex128)
    d[:,0] = delays[:]

    #Spatial correlation matrix is found via sdma design and copy-pasted!
    sc_top = [1.00243695+0.j,0.22441499+0.00078639j,0.02096756+0.00018244j,
    0.09272458-0.00083857j,0.02096756+0.00018244j,0.22441499+0.00078639j]
    sc_top = np.array(sc_top)/(sc_top[0])
    sc = la.toeplitz(sc_top,sc_top)

    #Solve the equation I wrote down from Microphone DSP, pg ???
    in_brack = la.inv(sc + mu*np.eye(len(sc)))
    top_line = np.matmul(in_brack,d)
    bottom_line = np.matmul((d.T).conj(),np.matmul(in_brack,d))

    coef = top_line / bottom_line


def main():
    """ Simulation goes here """
    # Start with an idealised case that does the usual two-norm based setup.
    #   Let our desired signal be at [10,3,0]
    #   Let our interference be at [1,0,0]
    c = 34.3
    fs = 100e3

    UCA6 = np.array([[0,0.057,0],[0.0494,0.0285,0],[0.0494,-0.0285,0],
                    [0,-0.057,0],[-0.0494,-0.0285,0],[-0.0494,0.0285,0]])

    channels = STA.simulate()
    #print("Channels=",np.shape(channels) )
    delays_good = ut.give_delays(np.array([10,0,0]),UCA6,c,fs)
    delays_bad = ut.give_delays(np.array([-3,-3,0]),UCA6,c,fs)


    freqs = np.arange(0,100e3,1e3)

    #I use this to save and load different sdb_coef
    if (1==0):
        sdb_coef = sd.get_sdb_coef(delays_good,UCA6,fs,c,freqs,15)
        np.save("sdb_coef_back_mu_high.npy",sdb_coef)
    else:
        sdb_coef =np.load("sdb_coef_back.npy")
        #sdb_coef_mu_high =np.load("sdb_coef_back_mu_high.npy")

    sig_good = filter_sum(channels,sdb_coef)
    #sig_bad = filter_sum(channels,sdb_coef_mu_high)

    #sig_good = delay_sum(channels,delays_good)
    sig_bad = delay_sum(channels,delays_bad)

    print(np.var(sig_good)," = var(sig_good)")
    print(np.var(sig_bad)," = var(sig_bad)")

    time = np.arange(0,5,1/fs)

    #fig,ax = plt.subplots(nrows=3, ncols=2)
    #ax[0,0].plot(time,channels[0,:])
    #ax[1,0].plot(time,channels[1,:])
    #ax[2,0].plot(time,channels[2,:])
    #ax[0,1].plot(time,channels[3,:])
    #ax[1,1].plot(time,channels[4,:])
    #ax[2,1].plot(time,channels[5,:])

    plt.plot(time,sig_good,label = "good")
    plt.plot(time,sig_bad,label = "bad")

    #writeout(sig_good,"sig_good.wav")
    #writeout(sig_bad,"sig_bad.wav")

    #plt.plot(10*np.log10(np.abs(nfft.fft(sig_good))),label='in')
    #plt.plot(10*np.log10(np.abs(nfft.fft(channels[1,:]))),label='out')
    #plt.legend()
    #plt.show()

    #plt.plot(time,channels[0,:],label = "ch0")
    #plt.plot(time,channels[1,:],label = "ch1")

    plt.legend()
    plt.show()

main()

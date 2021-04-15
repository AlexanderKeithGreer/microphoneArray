import numpy as np
import numpy.linalg as la
import numpy.random as ra
import numpy.fft as nf
import scipy.signal as sig
import scipy.io.wavfile as wavf
import matplotlib.pyplot as plt

import SignalToArray as STA
import utility as ut
import diffuse_sdb as dsdb
import sdmaDesign as sd
import cdma as cd

def generate_test_freqs(n_samp,fs,f_range):
    """We only care about a range from dc to 12kHz"""
    n_samp_int = np.int64(n_samp)
    freqs = np.zeros(n_samp_int)
    freq_max_index = np.int64(np.round((f_range/fs)*n_samp))
    freqs[:(freq_max_index+1)] = 1
    freqs[(n_samp_int-freq_max_index):] = 1
    test_data = nf.ifft(freqs)
    return test_data

def iterate_over_angles(filters,array,n_values):
    """ Evaluate the effectiveness of it all """

    r = 5 #Easily sufficient for a far field condition?? YES!!
    fs_high = 5e6 #Hopefully enough
    c = 343 #Propagation speed

    up_delay = 250
    down_display = 1
    freqs = generate_test_freqs(fs_high,fs_high,12.5e3)


    angles_l = 180
    angles = np.linspace(0,2*np.pi,angles_l,endpoint=False)
    #output_size = np.int64((fs_high/(up_delay*down_display),angles_l))
    output_size = np.int64((n_values,angles_l))
    output = np.zeros(output_size)

    for angle in range(len(angles)):

        print(angles[angle])
        #FIND POSITION
        source = np.array([r*np.cos(angles[angle]),r*np.sin(angles[angle]),0])
        #Find delays, and apply them
        delays = np.int64(ut.give_delays(source,array,c,fs_high))
        channels = ut.delay_signal_across_elements(freqs,delays)
        channels = sig.resample_poly(channels,1,up_delay,axis=1)
        spat_filtered = ut.filter_sum(channels,filters)
        spat_filtered_l = sig.resample_poly(spat_filtered,1,down_display)
        spat_filtered_ls = np.abs(nf.fft(spat_filtered_l[:np.int64(n_values)]))
        output[:,angle] = spat_filtered_ls

    output = (output/max(output.flatten()))
    output = output.T
    output = np.append(output,output,axis=0)
    output_dB = 10*np.log10(output)
    plt.imshow(output_dB)
    plt.show()

def run_sim():
    """ Work"""
    n_elem = 12
    #nb that 2.5k effective bandwidth for r=25.4mm (2k -> 4.5k) rr=69mm
    #        3.0k effective bandwidth for r=20.0mm (2.75k -> 5.75k) rr=57mm
    #        3.25k effective bandwidth for r=17.5mm (3k -> 6.25k) rr=53mm
    # 1 Inch radius is probably superior because it includes lower frequencies
    #FOR THE SUPER DIRECTIVE BEAMFORMER NOT DIFFERENTIAL
    array_r = 0.0254
    array_position_noise = 0.00005 #Fairly good. Distribution probably different
    array_angles = np.pi*np.linspace(0,2,n_elem,endpoint=False)
    array = np.zeros((len(array_angles),3))
    array[:,0] = array_r*np.cos(array_angles) + ra.randn(n_elem)*array_position_noise
    array[:,1] = array_r*np.sin(array_angles) + ra.randn(n_elem)*array_position_noise

    position = np.array([100,0,0])
    delays = ut.give_delays(position,array,343,np.int64(20e3),round=False)
    delays = delays-min(delays)
    print("Delays=",delays)
    #delay_coef = np.zeros((len(array_angles),max(delays)+1))
    #for channel in range(len(delays)):
    #    delay_coef[channel,delays[channel]] = 1

    freqs = np.linspace(0,20e3,40,endpoint=False)
    d = ut.conv_delays_to_freq(delays,20e3,freqs)

    sv_coef = ut.find_steering_vector(freqs,array_r,array_angles,0,343)
    dsdb_coef = dsdb.get_sdb_coef(sv_coef,array,20e3,343,freqs,15)

    for row in range(len(dsdb_coef[:,0])):
        dsdb_coef[row,:] = dsdb_coef[row,:] * sig.hamming(40)

    #print("\n\nDifferential Array Form")
    #nulls = np.array([10,120,170])*(np.pi/180)
    #cd_coef = cd.get_cd_coef_6(array_r,343,array_angles,nulls,freqs,0e3,4.5e3)

    #for filter in range(len(d)):
    #    d[filter,:50] = 0

    #d_filt = nf.fftshift(nf.fft(d.T))

    #for filter in range(len(d_filt))

    iterate_over_angles(dsdb_coef,array,20000)


run_sim()

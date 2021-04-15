import numpy as np
import numpy.linalg as la
import scipy.signal as sig
import scipy.io.wavfile as wavf
import matplotlib.pyplot as plt

def load_audio(f_name):
    """Loads an audio file (fs=50e3) and returns it as a numpy array"""
    rate,raw_audio = wavf.read(f_name)
    if (rate != 50e3):
        print("Strange Sampling rate???? Potential Error!")
    return raw_audio
#TEST:Natural


def prepare_audio_in(natural_audio):
    without_dc_audio = natural_audio - np.mean(natural_audio)
    prepared_audio = sig.resample_poly(without_dc_audio,200,1,axis=1)
    return prepared_audio;
#TEST:Natural


def map_to_place(array, signals, source_positions, c, fs):
    """Use the positions to add delays based on position
    For the arrays, the m/across-axis denotes dimension(x,y,z), the n/down-axis
        array element.

    """
    no_sources = len(signals[:,0])
    no_array_elements = len(array[:,0])
    no_samples = len(signals[0])

    output_channels = np.zeros([no_array_elements,no_samples])

    for source in range(no_sources):
        print("source_positions[source] = ",source_positions[source])
        for element in range(no_array_elements):
            #Determine the delay with the 2-norm
            print("array[element] = ",array[element])
            distance = la.norm(source_positions[source] - array[element])
            delay = np.int(np.round((distance/c)*fs))
            print("(Source, Element, Delay) = (",source,",",element,",",delay,")")
            output_channels[element] += np.append(signals[source,delay:],np.zeros(delay))

    return output_channels
#TEST:Output delays

def finalise_outputs(input_channels):
    output_channels = sig.resample_poly(input_channels,1,100,axis=1)
    return output_channels


def simulate():
    """Simulations are important and good."""
    #Example declarations
    sources_centre = np.array([[0,3,0],[0,0,0]])
    sources_on_end = np.array([[0,1,0],[3,1,0]])
    sources_test = np.array([[10,0,0]])

    UCA6 = np.array([[0,0.057,0],[0.0494,0.0285,0],[0.0494,-0.0285,0],
                    [0,-0.057,0],[-0.0494,-0.0285,0],[-0.0494,0.0285,0]])
    ULA4 = np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]])

    #plt.scatter(UCA6[:,0],UCA6[:,1])
    #plt.show()

    c = 34.3              # Propagation speed
    fs_in = 50e3         # Low frequency fs
    fs_up = fs_in*200    # High frequency fs

    #Simulation chain
    signal_aud = load_audio("data.wav")
    interf_aud = load_audio("interf.wav")

    # Learn multiargs and fix this later!!!
    if (len(signal_aud) != len(interf_aud)):
        print("Length Error!!!!")
    #aud = np.array([signal_aud, interf_aud])
    aud = np.array([signal_aud])

    signals_up = prepare_audio_in(aud)
    signals_up = map_to_place(UCA6,signals_up,sources_test,c,fs_up)
    per_element = finalise_outputs(signals_up)

    return per_element


#All tests use direct comparison.Correlations are for later!
#TEST:      Place source in centre for UCA and single source
#EXPECT:    No time difference

#TEST:      Place source at end for ULA and single source
#EXPECT:    Linear Time Delays

#TEST:      Place 2 sources at opposite ends for ULA and examine Delays
#EXPECT:    Equal and opposite delays for each element

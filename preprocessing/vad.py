from pyAudioAnalysis import audioFeatureExtraction as afe
from pyAudioAnalysis import audioBasicIO as abi
import matplotlib.pyplot as plt
import numpy as np
import sys

def movingAverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

    # STMA- Short term moving average
    # LTMA- Long  term moving average
if __name__ == '__main__':

    # file preprocessing - read and abs
    wav_file = sys.argv[1]

    # fs is the rate, and x.shape[0] would be number of samples we have
    [fs, x] = abi.readAudioFile(wav_file)

    # signal normalization
    x_abs = np.absolute(x)
    x_abs = x_abs - x_abs.mean()
    x_abs = x_abs / x_abs.max()

    # 1/8 s is the length of a frame for which we will extract features later
    # ltma is ~ 1/16 s
    # stma is ~ 1/32 s
    stma_framecount = int(fs * 0.03)
    ltma_framecount  = int(fs * 0.063)

    stma = movingAverage(x_abs, stma_framecount)
    ltma = movingAverage(x_abs, ltma_framecount)
    print('x_abs shape', x_abs.shape)
    print('stma  shape', stma.shape)
    print('ltma  shape', ltma.shape)
    # boosting the ltma a bit, for clearance
    ltma = ltma + 0.015

    # stma and ltma arrays are shorter, so we are filling them from the
    # beginning to get the relative position of frames in which we will detect
    # activation or lack of activation
    # we will be using np.zeros instead of np.linspace for performance reasons

    x_sample_count = x.shape[0]
    stma_end_array = np.append(np.zeros((x_sample_count - stma.shape[0],)), stma)
    ltma_end_array = np.append(np.zeros((x_sample_count - ltma.shape[0],)), ltma)
    print('stma  endsh', stma_end_array.shape)
    print('ltma  endsh', ltma_end_array.shape)

    #x_dziedzina = list(range(x.shape[0]))
    #shrt_line = plt.plot(x_dziedzina[len(x_dziedzina) - len(stma) :],
    #                     stma, label='STMA')
    #long_ling = plt.plot(x_dziedzina[len(x_dziedzina) - len(ltma) :],
    #                     ltma, label='LTMA')

    plt.legend()
    plt.show()


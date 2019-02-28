from pyAudioAnalysis import audioFeatureExtraction as afe
from pyAudioAnalysis import audioBasicIO as abi
import matplotlib.pyplot as plt
import numpy as np

def movingAverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

if __name__ == '__main__':

    #file preprocessing - read and abs
    wav_file = "sample1.wav"
    [fs, x] = abi.readAudioFile(wav_file)
    x_abs = np.absolute(x)
    x_abs = x_abs - x_abs.mean()
    x_abs = x_abs / x_abs.max()
    # counting averages for vad
    shrt_term_avg_framecount = fs / 10000
    long_term_avg_framecount  = fs / 1000
    shrt_avg = movingAverage(x_abs, shrt_term_avg_framecount)
    long_avg = movingAverage(x_abs, long_term_avg_framecount)
    long_avg = long_avg + 0.02
    # pizda jest i chuj w gratisie

    x_dziedzina = list(range(x.shape[0]))
    shrt_line = plt.plot(x_dziedzina[len(x_dziedzina) - len(shrt_avg) :],
                         shrt_avg, label='STMA')
    long_ling = plt.plot(x_dziedzina[len(x_dziedzina) - len(long_avg) :],
                         long_avg, label='LTMA')

    plt.legend()
    plt.show()




   # plt.figure(1)
   # plt.plot(x)
   # plt.figure(2)
   # plt.plot(x_abs)
   # plt.show()

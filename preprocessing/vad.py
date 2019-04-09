from pyAudioAnalysis import audioFeatureExtraction as afe
from pyAudioAnalysis import audioBasicIO as abi
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool
from operator import itemgetter
import numpy as np
import sys
import time

def movingAverage(data):
    values = data[0]
    window = data[1]
    identifier = data[2]
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return (sma, identifier)

def determine_frame_ranges(data):
    comparison_arr = data[0]
    last_element_act = data[1]
    non_silence_moments = list()
    arr_last = len(comparison_arr)-1
    comp_arr_len = len(comparison_arr)
    if comparison_arr[arr_last] == True:
        count = 0
        count = np.count_nonzero(comparison_arr[:]==True)
        if count < int(comp_arr_len * 0.95):
            non_silence_tuple = (last_element_act-comp_arr_len, last_element_act)
            if len(non_silence_moments) > 0:
                if non_silence_moments[len(non_silence_moments)-1][1] ==\
                non_silence_tuple[0]:
                    non_silence_moments[-1] = (non_silence_moments[-1][0],
                                               non_silence_tuple[1])
                else:
                    non_silence_moments.append(non_silence_tuple)
            else:
                non_silence_moments.append(non_silence_tuple)
    return non_silence_moments

def merge_frame_ranges(ranges, fs):
    merged = list()
    for i in range(0, len(ranges)):
        curr_tuple = ranges[i]
        if i==0:
            merged.append(curr_tuple)
        else:
            if curr_tuple[0]-merged[len(merged)-1][1] < int(0.125 * fs):
                merged[len(merged)-1] = (merged[len(merged)-1][0],
                                             curr_tuple[1])
            else:
                merged.append(curr_tuple)
    return merged

def perform_vad(wav_file):
    # fs is the rate, and x.shape[0] would be number of samples we have
    [fs, x] = abi.readAudioFile(wav_file)
    # signal normalization
    x_abs = np.absolute(x)
    x_abs = x_abs - x_abs.mean()
    x_abs = x_abs / x_abs.max()

    stma_framecount = int(fs * 0.001)
    ltma_framecount  = int(fs * 0.01)

    # splitting the data into chunks and passing it to threads

    pool = ThreadPool(2)
    results_wrapped = pool.map(movingAverage, \
                               [(x_abs, stma_framecount, 'stma'),
                                (x_abs, ltma_framecount, 'ltma')])
    stma = 0
    ltma = 0
    if results_wrapped[0][1] == 'stma':
        stma = results_wrapped[0][0]
        ltma = results_wrapped[1][0]
    else:
        ltma = results_wrapped[0][0]
        stma = results_wrapped[1][0]

    # boosting the ltma a bit, for clearance
    ltma = ltma + 0.015

    # stma and ltma arrays are shorter, so we are filling them from the
    # beginning to get the relative position of frames in which we will detect
    # activation or lack of activation

    x_sample_count = x.shape[0]
    stma_end_array = np.append(np.zeros((x_sample_count - stma.shape[0],)), stma)
    ltma_end_array = np.append(np.zeros((x_sample_count - ltma.shape[0],)), ltma)
    comparison_arr = np.greater(ltma_end_array, stma_end_array)

    # splitting the data into chunks and passing it to threads
    chunks = list()
    for i in range(0, len(comparison_arr), ltma_framecount):
        chunks.append((comparison_arr[i:i+ltma_framecount], i+ltma_framecount))
    pool = ThreadPool(4)
    results_wrapped = pool.map(determine_frame_ranges, chunks)
    results_unwrapped = []
    for r in results_wrapped:
        if len(r) > 0:
            results_unwrapped.append(r[0])
    results_unwrapped.sort(key=itemgetter(1))
    vad_ranges = merge_frame_ranges(results_unwrapped, fs)

    # for demo purposes
    x_dziedzina = list(range(x.shape[0]))
    shrt_line = plt.plot(x_dziedzina[len(x_dziedzina) - len(stma) :],
                         stma, label='STMA')
    long_ling = plt.plot(x_dziedzina[len(x_dziedzina) - len(ltma) :],
                         ltma, label='LTMA')
    for t in vad_ranges:
        plt.plot([t[0],t[1]], [0.02,0.02],'g-')

    plt.legend()
    plt.show()
    return vad_ranges


    # STMA- Short term moving average
    # LTMA- Long  term moving average

if __name__ == '__main__':
    perform_vad(sys.argv[1])

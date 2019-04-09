from pyAudioAnalysis import audioFeatureExtraction as afe
from pyAudioAnalysis import audioBasicIO as abi

import numpy as np

def st_fe_file(wav_file):
    [fs, x] = abi.readAudioFile(wav_file)
    win = int(0.125 * fs)
    [st_feats, st_feat_names] = afe.stFeatureExtraction(x, fs, win, win)
    print(st_feats)

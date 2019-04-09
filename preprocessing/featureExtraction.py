from pyAudioAnalysis import audioFeatureExtraction as afe
from pyAudioAnalysis import audioBasicIO as abi
from vad import VoiceActivationDetector
import numpy as np

class FeatureExtractor:
    def __short_term_feature_extraction_from_file(self, fs, x):
        win = int(0.125 * fs)
        [st_feats, st_feat_names] = afe.stFeatureExtraction(x, fs, win, win)
        return st_feats

    def extract_features(self, wav_filepath):
        voice_activation_detector = VoiceActivationDetector()
        frame_ranges = voice_activation_detector.perform_vad(wav_filepath)
        [fs, x] = abi.readAudioFile(wav_filepath)

        # now we`ll cut out the silence
        x_activated = np.zeros([], dtype = np.int16)
        first = True
        for pair in frame_ranges:
            if not first:
                x_activated = np.concatenate((x_activated, x[pair[0]:pair[1]]))
            else:
                x_activated = x[pair[0]:pair[1]]
                first = False

        # get features for non-silence fragments
        features = self.__short_term_feature_extraction_from_file(fs, x)
        return features

    # this function adds another feature
    # by labeling each frame (normal speach/ whisper)

    def extract_and_label(self, wav_filepath, label):
        # 0 - whisper
        # 1 - normal speach
        features = self.extract_features(wav_filepath)
        if label == 1:
            labels = np.ones([1,features.shape[1]])
        elif label == 0:
            labels = np.zeros([1,features.shape[1]])
        return np.concatenate((features, labels), axis=0)


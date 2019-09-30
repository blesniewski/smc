from pyAudioAnalysis import audioFeatureExtraction as afe
from pyAudioAnalysis import audioBasicIO as abi
from vad import VoiceActivationDetector
import matplotlib.pyplot as plt
from math import floor, ceil
import numpy as np

class FeatureExtractor:
    def __short_term_feature_extraction_from_file(self, fs, x):
        win = int(0.125 * fs)
        print(str(win) + " vs " + str(len(x)))
        [st_feats, st_feat_names] = afe.stFeatureExtraction(x, fs, win, win)
        return st_feats

    def extract_features(self, wav_filepath):
        # extract features for the whole file
        [fs, x] = abi.readAudioFile(wav_filepath)
        features = self.__short_term_feature_extraction_from_file(fs, x)
        return features

    # this function adds another feature
    # by labeling each frame (normal speach/ whisper/ silence)

    def extract_and_label(self, wav_filepath, label):
        # 0 - silence - assigned by VAD only
        # 1 - whisper
        # 2 - normal speech
        possible_labels = [1,2]
        # placeholder
        label_array = np.array([1,1])
        voice_activation_detector = VoiceActivationDetector()

        # frame ranges of non-silence (what it is depends on the label)
        frame_ranges = voice_activation_detector.perform_vad(wav_filepath)
        # here we fill the whole array with silence, and will only change
        # the label where VAD said there is something else
        features = self.extract_features(wav_filepath)
        if label in possible_labels:
            label_array = np.zeros([1,features.shape[1]])
            # print(label_array.shape, features.shape)
        for pair in frame_ranges:
            label_array[0][ceil(pair[0]/5512):floor(pair[1]/5512)] = label

        # [fs, x] = abi.readAudioFile(wav_filepath)
        # voice_plot = plt.plot(list(range(x.shape[0])), x, 'b', label='voice')
        # label_plot = plt.plot(list(range(0,label_array.shape[1]*5512,5512)),
        #                       label_array[0][:], 'r')
        # plt.legend()
        # plt.show()

        return np.concatenate((features, label_array), axis=0)

if __name__ == '__main__':
    fe = FeatureExtractor()
    fe.extract_and_label('sample1.wav', 1)

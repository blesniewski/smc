from featureExtraction import FeatureExtractor
from keras.models import model_from_json
from pyAudioAnalysis import audioBasicIO as abi
import sys
import numpy as np
import matplotlib.pyplot as plt
class ModelValidator:
    def validate(self, wav_filepath, model_filepath, weights_filepath):
        featureExtractor = FeatureExtractor()
        [fs, x] = abi.readAudioFile(wav_filepath)
        features = featureExtractor.extract_features(wav_filepath)
        print(features.shape)
        features = features.transpose()
        print(features.shape)
        json_file = open(model_filepath,'r')
        loaded_model_json = json_file.read()
        json_file.close()

        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(weights_filepath)
        loaded_model.compile(loss='binary_crossentropy', optimizer='adam',
                     metrics=['accuracy'])

        output = loaded_model.predict(features)
        print(output.shape)



        x_domain = list(range(len(x)))
        multiplier = int(len(x) / features.shape[0])
        ftrs = np.repeat(output, multiplier, axis=0)
        wav = plt.plot(x_domain,x)
        ftrsf_plt = plt.plot(x_domain[0:ftrs.shape[0]],ftrs[:,0]*10000, 'r', label='silence')
        ftrss_plt = plt.plot(x_domain[0:ftrs.shape[0]],ftrs[:,1]*10000, 'k', label='whisper')
        ftrst_plt = plt.plot(x_domain[0:ftrs.shape[0]],ftrs[:,2]*10000, 'g', label='normal')
        plt.legend()
        plt.show()




        return output

if __name__ == '__main__':
    mv = ModelValidator()
    mv.validate(sys.argv[1],sys.argv[2],sys.argv[3])


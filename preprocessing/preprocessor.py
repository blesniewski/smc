from featureExtraction import FeatureExtractor
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
from progress.bar import Bar
class Preprocessor:

    # sample type:
    # 0 - silence - assigned by VAD only
    # 1 - whisper
    # 2 - normal speech
    def __get_all_files_in_dir(self, dirpath):
        files = []
        dircontents = listdir(dirpath)
        for c in dircontents:
            c_path = join(dirpath, c)
            if isfile(c_path):
                files.append(c_path)
            else:
                files = files + (self.__get_all_files_in_dir(c_path))
        return files

    def process_wav_files_in_dir(self, dirpath, outfile, label):
        files = self.__get_all_files_in_dir(dirpath)
        featureExtractor = FeatureExtractor()
        label = int(label)
        # placeholder
        features = np.ones((1))

        if len(files) > 0:
            features = featureExtractor.extract_and_label(files[0], label)
        #extract features and print some info along the way
        iteration = 0
        tot_size = 0
        for f in files[1:]:
            iteration += 1
            file_features = featureExtractor.extract_and_label(f,label)
            features = np.concatenate((features, file_features), axis = 1)
            tot_size += file_features.shape[1]
            if iteration % 5 == 0:
                print(iteration," / ", len(files), "size: ",tot_size , end='\r')

        np.save(outfile, features)


    # args: beginning directory, output file, label
if __name__ == "__main__":
    pr = Preprocessor()
    if len(sys.argv) == 4:
        pr.process_wav_files_in_dir(sys.argv[1], sys.argv[2], sys.argv[3])

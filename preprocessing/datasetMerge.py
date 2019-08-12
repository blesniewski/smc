import sys
import numpy as np
class DatasetMerger:
    def merge_two_datasets(self, path_ds1, path_ds2, out_path):
        ds1 = np.load(path_ds1)
        ds2 = np.load(path_ds2)
        print("DS 1 shape: {}".format(ds1.shape))
        print("DS 2 shape: {}".format(ds2.shape))
        bigger = 0

        # we need to cut off excess of material to merge them nicely
        if ds1.shape[1] > ds2.shape[1]:
            ds1 = ds1[:,:ds2.shape[1]]
            bigger = 1
        elif ds1.shape[1] < ds2.shape[1]:
            ds2 = ds2[:,:ds1.shape[1]]
            bigger = 2

        out_size = ds1.shape[1]+ds2.shape[1]
        print(out_size)
        dso = np.empty((ds1.shape[0], out_size),\
                        dtype = ds1.dtype)

        dso[:,::2] = ds1
        dso[:,1::2] = ds2
        print(dso.shape)
        # now we can append the material that was cut off (no wasting ;))
        if bigger == 1:
            ds1 = np.load(path_ds1)
            dso = np.concatenate((dso,ds1[:,ds2.shape[1]+1:]), axis = 1)
        if bigger == 2:
            ds2 = np.load(path_ds2)
            dso = np.concatenate((dso,ds2[:,ds1.shape[1]+1:]), axis = 1)
        print(dso.shape)
        np.save(out_path, dso)

if __name__ == "__main__":
    dsm = DatasetMerger()
    dsm.merge_two_datasets(sys.argv[1], sys.argv[2], sys.argv[3])

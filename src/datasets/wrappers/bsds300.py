import os.path

import h5py
import numpy as np


class BSDS300:
    """
    A dataset of patches from BSDS300.
    """

    class Data:
        """
        Constructs the dataset.
        """

        def __init__(self, data):
            self.x = data[:]
            self.N = self.x.shape[0]

    def __init__(self, path: str):
        # load dataset
        f = h5py.File(os.path.join(path, "BSDS300", "BSDS300.hdf5"), "r")

        self.trn = self.Data(f["train"])
        self.val = self.Data(f["validation"])
        self.tst = self.Data(f["test"])

        self.n_dims = self.trn.x.shape[1]
        self.image_size = [int(np.sqrt(self.n_dims + 1))] * 2
        f.close()

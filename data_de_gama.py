import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataReview:
    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.num_feature_cols = self.features.shape[1]

    def hist_creator(self, save_file=False):
        num_df = self.features.select_dtypes(np.number)
        shape = num_df.shape
        runs = int(math.ceil(shape[1]/9.0))

        for runs in range(runs):
            num_df.iloc[:, (0 + 9*runs):(9 + 9*runs)].hist(bins=50, figsize=(30,22))
            if save_file:
                plt.savefig(f'hist_img_{runs}.jpg')
            else:
                plt.show()


    def corr_matrix(self):
        pass
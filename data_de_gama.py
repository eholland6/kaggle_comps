import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataReview:
    def __init__(self, features, target=None):
        self.features = features
        self.target = target
        self.num_feature_cols = self.features.shape[1]

    def schema_desc(self):
        num_df = self.features.select_dtypes(np.number)
        num_numeric_cols = len(num_df.columns)
        numeric_ratio = num_numeric_cols/self.num_feature_cols
        print(f'The features dataset has {self.num_feature_cols} columns')
        print(f'{num_numeric_cols} ({numeric_ratio}) of those columns are numeric')

        count_cols_nan = 0
        list_cols_nan = []
        count_cols_inf = 0
        list_cols_inf = []
        for col in num_df:
            num_null_values = num_df[col].isnull().sum()
            if num_null_values > 0:
                temp_list = []
                count_cols_nan += 1
                temp_list.append(col)
                temp_list.append(num_null_values)
                list_cols_nan.append(temp_list)
            num_inf_values = np.isinf(self.features[col]).values.sum()
            if num_inf_values > 0:
                temp_list = []
                count_cols_inf += 1
                temp_list.append(col)
                temp_list.append(num_inf_values)
                list_cols_inf.append(temp_list)

        print(f'There are {count_cols_nan} columns with at least 1 null / NaN value')
        print(f'There are {count_cols_inf} columns with at least 1 infinite value')
        if len(list_cols_nan) > 0:
            print(f'The columns with null / NaN values are:')
            for col_record in list_cols_nan:
                print(f'{col_record[0]} has {col_record[1]} null values')
        if len(list_cols_inf) > 0:
            print('The columns with infinite values are:')
            for col_record in list_cols_inf:
                print(f'{col_record[0]} has {col_record[1]} inf values')

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

    def corr_matrix(self, fig_size=(8,8)):
        fig, ax = plt.subplots(figsize=fig_size)

        corr = self.features.corr()
        ax = sns.heatmap(
            corr, 
            vmin=-1, vmax=1, center=0,
            cmap=sns.diverging_palette(20, 220, n=200),
            square=True,
            ax=ax
        )
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=45,
            horizontalalignment='right'
        )

        plt.savefig('corr_matrix.jpg')
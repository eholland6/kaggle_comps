import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import pickle

class ModelPrep():
    def __init__(self, features, target):
        self.features = features
        self.target = target

    def train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.features, self.target, test_size=.23)

    def catg_one_hot_encode(self, col_list=None, handle_unknown_setting='ignore', categories_list=None, file_name_suffix='', file_path=''):
        categorical_features_poss_ = []
        binary_num_col_list_ = []
        #encoder_list = []
        #encoder_dict = {}

        if col_list == None:
            for cols in self.X_train.columns:
                if float(self.X_train[cols].nunique()) / float(self.X_train[cols].count()) < 0.1 \
                and self.X_train[cols].nunique() < 100:
                    categorical_features_poss_.append(cols)
                else:
                    pass
                if self.X_train[cols].nunique() == 2 \
                and np.issubdtype(self.X_train[cols].dtype, np.number) == True:
                    binary_num_col_list_.append(cols)
                else:
                    pass

            self.categorical_features_poss_ = categorical_features_poss_
            self.binary_num_col_list_ = binary_num_col_list_

            handle_list_ = []

            print(f'The following features appear likely to be categorical:')
            for i in categorical_features_poss_:
                print(i)
                if i in self.binary_num_col_list_:
                    print(f'{i} is a feature with numeric data and only 2 unique values. It may not require encoding')

                react = input('Would you like to one-hot encode (1), drop (2), pass (3):')
                handle_list_.append(react)
        else:
            categorical_features_poss_ = col_list
            handle_list_ = []
            for i in range(len(col_list)):
                handle_list_.append(1)

        one_hot_df_train = pd.DataFrame()
        one_hot_df_test = pd.DataFrame()

        if categories_list == None:
            categ_list = []
            for i in range(len(handle_list_)):
                categ_list.append('auto')
        else:
            categ_list = categories_list

        for i in range(len(categorical_features_poss_)):
            if int(handle_list_[i]) == 1:
                enc = OneHotEncoder(handle_unknown=handle_unknown_setting, categories=categ_list[i])
                enc.fit((self.X_train[categorical_features_poss_[i]]).values.reshape(-1,1))

                enc_train_array = enc.transform((self.X_train[categorical_features_poss_[i]]).values.reshape(-1,1)).toarray()

                df_train = pd.DataFrame(enc_train_array)
                df_train.columns = enc.get_feature_names_out([categorical_features_poss_[i]])
                one_hot_df_train.reset_index(drop=True, inplace=True)
                df_train.reset_index(drop=True, inplace=True)
                self.X_train.reset_index(drop=True, inplace=True)
                one_hot_df_train = pd.concat([one_hot_df_train, df_train], axis=1)
                self.X_train = self.X_train.drop(categorical_features_poss_[i], axis=1)

                enc_test_array = enc.transform((self.X_test[categorical_features_poss_[i]]).values.reshape(-1,1)).toarray()

                df_test = pd.DataFrame(enc_test_array)
                df_test.columns = enc.get_feature_names_out([categorical_features_poss_[i]])
                one_hot_df_test.reset_index(drop=True, inplace=True)
                df_test.reset_index(drop=True, inplace=True)
                self.X_test.reset_index(drop=True, inplace=True)
                one_hot_df_test = pd.concat([one_hot_df_test, df_test], axis=1)
                self.X_test = self.X_test.drop(categorical_features_poss_[i], axis=1)

                # Save encoder to use on downstream datasets
                enc_file_name = file_path + categorical_features_poss_[i] + '_one_hot_' + file_name_suffix + '.pkl'
                pickle.dump(enc, open(enc_file_name, 'wb'))
                #encoder_list.append(enc_file_name)
                #encoder_dict['one_hot_encoder_'+categorical_features_poss_[i]] = enc_file_name
            elif int(handle_list_[i]) == 2:
                self.X_train.drop(categorical_features_poss_[i], axis=1)
                self.X_test.drop(categorical_features_poss_[i], axis=1)
            else:
                pass

        self.X_train = pd.concat([self.X_train, one_hot_df_train], axis=1)
        self.X_test = pd.concat([self.X_test, one_hot_df_test], axis=1)